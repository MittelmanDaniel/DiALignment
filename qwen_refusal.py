import pickle
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc

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

"""### Load model"""

MODEL_PATH = 'Qwen/Qwen-1_8B-chat'
DEVICE = utils.get_device()

print("Running on device:", DEVICE)

"""### Load harmful / harmless datasets"""

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

"""### Tokenization utils"""

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str]
) -> Int[Tensor, 'batch_size seq_len']:
    QWEN_CHAT_TEMPLATE = """<|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    """
    prompts = [QWEN_CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]
    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids

"""### Generation utils"""

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
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:

    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations


def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    amplitude
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj * amplitude

def direction_amplify_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    amplitude
):
    #proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation + direction * 0.5 * amplitude#proj

def load_model():
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_PATH,
        device=DEVICE,
        dtype=torch.float16,
        default_padding_side='left',
        fp16=True
    )

    model.tokenizer.padding_side = 'left'
    model.tokenizer.pad_token = '<|extra_0|>'
    return model

def initialize_activation_vectors(model):
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()

    print("Harmful instructions:")
    for i in range(4):
        print(f"\t{repr(harmful_inst_train[i])}")
    print("Harmless instructions:")
    for i in range(4):
        print(f"\t{repr(harmless_inst_train[i])}")

    tokenize_instructions_fn = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer)
    N_INST_TRAIN = 32

    # tokenize instructions
    harmful_toks = tokenize_instructions_fn(instructions=harmful_inst_train[:N_INST_TRAIN])
    harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[:N_INST_TRAIN])

    # run model on harmful and harmless instructions, caching intermediate activations
    harmful_logits, harmful_cache = model.run_with_cache(harmful_toks,
                                                         names_filter=lambda hook_name: 'resid' in hook_name)
    harmless_logits, harmless_cache = model.run_with_cache(harmless_toks,
                                                           names_filter=lambda hook_name: 'resid' in hook_name)

    # compute difference of means between harmful and harmless activations at an intermediate layer

    pos = -1
    layer = 14

    harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
    harmless_mean_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0)

    refusal_dir = harmful_mean_act - harmless_mean_act
    refusal_dir = refusal_dir / refusal_dir.norm()

    # clean up memory
    # del harmful_cache, harmless_cache, harmful_logits, harmless_logits
    # gc.collect();
    # torch.cuda.empty_cache()

    return refusal_dir, harmful_inst_test

def ablate_refusal(model, refusal_dir, prompts, amplitude):

    tokenize_instructions_fn = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer)

    intervention_dir = refusal_dir
    intervention_layers = list(range(model.cfg.n_layers))  # all layers

    hook_fn = functools.partial(direction_ablation_hook, direction=intervention_dir, amplitude=amplitude)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in
                 ['resid_pre']]

    intervention_generations = get_generations(model, prompts, tokenize_instructions_fn,
                                               fwd_hooks=fwd_hooks)
    baseline_generations = get_generations(model, prompts, tokenize_instructions_fn,
                                           fwd_hooks=[])

    return baseline_generations, intervention_generations

def amplify_refusal(model, refusal_dir, prompts, amplitude):

    tokenize_instructions_fn = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer)

    intervention_dir = refusal_dir
    intervention_layers = list(range(model.cfg.n_layers))  # all layers

    hook_fn = functools.partial(direction_amplify_hook, direction=intervention_dir, amplitude=amplitude)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in
                 ['resid_pre']] # , 'resid_mid', 'resid_post']

    intervention_generations = get_generations(model, prompts, tokenize_instructions_fn,
                                               fwd_hooks=fwd_hooks)
    baseline_generations = get_generations(model, prompts, tokenize_instructions_fn,
                                           fwd_hooks=[])

    return baseline_generations, intervention_generations

if __name__ == "__main__":
    model = load_model()
    # refusal_dir, harmful_inst_test = initialize_activation_vectors(model)
    with open("refusal_dir.pkl", "rb") as f:
        refusal_dir = pickle.load(f)
    baseline_generations, intervention_generations = amplify_refusal(model, refusal_dir, ["Can you teach me how do I tie a shoe?"], 1)
    print("Baseline:\n",baseline_generations)
    print("Intervention:\n",intervention_generations)