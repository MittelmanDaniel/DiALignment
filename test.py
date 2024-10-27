
import llm_surgeon

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

import pandas as pd
from datasets import load_dataset

import torch

HARMFUL_BATCH = 32

BENIGN_BATCH = 32


#GET HARMFUL STUFF
harmful_dataset = pd.read_csv('https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv')

harmful_instructions = harmful_dataset['goal'].to_list()

harmful_prompts = harmful_instructions[:HARMFUL_BATCH]

#GET BENIGN STUFF
hf_path = 'tatsu-lab/alpaca'
benign_dataset = load_dataset(hf_path)

benign_instructions = []
for i in range(len(benign_dataset['train'])):
    if benign_dataset['train'][i]['input'].strip() == '':
        benign_instructions.append(benign_dataset['train'][i]['instruction'])


benign_prompts = benign_instructions[:BENIGN_BATCH]


#RUN AND EVAL THE MODEL
torch.set_grad_enabled(False)
device = utils.get_device()

model = HookedTransformer.from_pretrained_no_processing("meta-llama/Llama-3.2-1B-Instruct", device=device, default_padding_side='left')

import llm_surgeon

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

import pandas as pd
from datasets import load_dataset

import torch

torch.set_grad_enabled(False)
device = utils.get_device()

model = HookedTransformer.from_pretrained_no_processing("meta-llama/Llama-3.2-1B-Instruct", device=device, default_padding_side='left')

def temp(prompts):
    HARMFUL_BATCH = 32

    BENIGN_BATCH = 32


  #GET HARMFUL STUFF
    harmful_dataset = pd.read_csv('https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv')

    harmful_instructions = harmful_dataset['goal'].to_list()

    harmful_prompts = harmful_instructions[:HARMFUL_BATCH]

    #GET BENIGN STUFF
    hf_path = 'tatsu-lab/alpaca'
    benign_dataset = load_dataset(hf_path)

    benign_instructions = []
    for i in range(len(benign_dataset['train'])):
        if benign_dataset['train'][i]['input'].strip() == '':
            benign_instructions.append(benign_dataset['train'][i]['instruction'])


    benign_prompts = benign_instructions[:BENIGN_BATCH]

    #RUN AND EVAL THE MODEL
    LAYER = 11
    refusal_dir = llm_surgeon.remove_vec_from_benign_to_mean(model, benign_prompts, harmful_prompts, LAYER)

    # Responses for base llm
    eval_chat = llm_surgeon.convert_prompts_to_chat_template(prompts)

    # CRUCIAL CHANGE: Use padding to ensure consistent sequence length
    unmodified_eval_tokens = model.tokenizer.apply_chat_template(eval_chat, return_tensors="pt", padding=True, tokenize=False).to(device) 
    unmodified_output = model.generate(unmodified_eval_tokens, max_new_tokens=256)

    llm_surgeon.add_refusal_hook(model, list(range(model.cfg.n_layers)), refusal_dir)

    # Apply padding here as well
    modified_eval_tokens = model.tokenizer.apply_chat_template(eval_chat, return_tensors="pt", padding=True, tokenize=False).to(device)
    modified_output = model.generate(modified_eval_tokens, max_new_tokens=256)

    return (unmodified_eval_tokens, unmodified_output, modified_eval_tokens, modified_output)


