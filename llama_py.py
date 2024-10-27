# %%
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore

from prompt_gen import generate_prompts

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
DEVICE = "cuda"

# %%
def load_model(model_name: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained_no_processing(
      model_name,
      device=DEVICE,
      dtype=torch.float16,
    )
    
    model.tokenizer.padding_side = 'left'
    return model

# %%
def get_harmful_instructions():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

# %%
def tokenize(model, instructions):
    chat = [
        {"role": "user", "content": instructions}
    ]
    text_tokens = model.tokenizer.apply_chat_template(chat, tokenize=False, padding=True, truncation=True)
    tokens = model.tokenizer.apply_chat_template(chat, tokenize=True, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    print(tokens.shape)
    return tokens

# %%
def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn,
    fwd_hooks = [],
    max_tokens_generated: int = 64,
    batch_size=4
) -> List[str]:

    generations = []
    
    raw_toks = [tokenize_instructions_fn(instructions=instruction) for instruction in instructions]
    print(raw_toks)
    all_toks = torch.Tensor(raw_toks, device=DEVICE)

    for i in tqdm(range(0, len(instructions), batch_size)):
        
        generation = _generate_with_hooks(
            model,
            all_toks[i:i+batch_size],
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations

# %%
def sample_mean_activation(model, tokenizer, prompts, batch_size):
    n_samples = len(prompts)
    n_layers = model.cfg.n_layers
    pos = -1
    
    mean_act = torch.zeros((n_layers, 1, model.cfg.d_model), dtype=model.cfg.dtype, device=DEVICE)
    
    all_toks = torch.cat([tokenizer(instructions=prompt) for prompt in prompts]).to(DEVICE)
        
    for i in range(0, n_samples, batch_size):
        logits, cache = model.run_with_cache(all_toks[i:i+batch_size], names_filter=lambda hook_name: 'resid' in hook_name)
        
        for layer_i in range(n_layers):
            mean_act[layer_i] += cache['resid_pre', layer_i][:, pos, :].sum(dim=0) / n_samples
            
        # memory management
        del cache, logits
        gc.collect(); torch.cuda.empty_cache()

    return mean_act

# %%
def find_activation_dir(desired_act, harmless_act):
    delta_dir = desired_act - harmless_act
    return delta_dir / delta_dir.norm(keepdim=True, dim=2)

# %%
def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

# %%
def direction_amplify_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    #proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation + direction#proj

# %%
def run_experiment(model, tokenizer, hook, intervention_direction, baseline_prompts, direction_layer=None):
    intervention_layers = list(range(1, model.cfg.n_layers))
    if direction_layer is None:
        fwd_hooks = [(utils.get_act_name(act_name, l), functools.partial(hook, direction=intervention_direction[l])) for l in intervention_layers for act_name in ['resid_pre']]
    else:
        fwd_hooks = [(utils.get_act_name(act_name, l), functools.partial(hook, direction=intervention_direction[direction_layer])) for l in intervention_layers for act_name in ['resid_pre']]

    intervention_generations = get_generations(model, baseline_prompts, tokenizer, fwd_hooks=fwd_hooks)
    baseline_generations = get_generations(model, baseline_prompts, tokenizer, fwd_hooks=[])
    
    for i in range(len(baseline_prompts)):
        print(f"INSTRUCTION {i}: {repr(baseline_prompts[i])}")
        print(Fore.GREEN + f"BASELINE COMPLETION:")
        print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RED + f"INTERVENTION COMPLETION:")
        print(textwrap.fill(repr(intervention_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RESET)
        
    return intervention_generations, baseline_generations


# %%
model = load_model("meta-llama/Llama-3.2-3B-Instruct")

# %%
harmful_inst_train, harmful_inst_test = get_harmful_instructions()
harmless_inst_train, harmless_inst_test = get_harmless_instructions()

# %%
n_samples = 64
batch_size = 32
n_inst_test = 8
tokenizer = functools.partial(tokenize, model=model)
harmful_mean_act = sample_mean_activation(model, tokenizer, harmful_inst_train[:n_samples], batch_size)
harmless_mean_act = sample_mean_activation(model, tokenizer, harmless_inst_train[:n_samples], batch_size)

# %%
refusal_dir_bylayer = find_activation_dir(harmful_mean_act, harmless_mean_act)
intervention_gens, baseline_gens = run_experiment(model, tokenizer, direction_ablation_hook, refusal_dir_bylayer, harmful_inst_test[:n_inst_test], direction_layer=14)

# %%
