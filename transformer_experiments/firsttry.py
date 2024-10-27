import transformer_lens.utils as utils
import torch
from transformer_lens import HookedTransformer

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import requests

import pandas as pd
import einops
import io

from datasets import load_dataset

from transformer_lens.hook_points import (
    HookPoint,
) 

harmfuldataset = pd.read_csv('https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv')

harmfulinstructions = harmfuldataset['goal'].to_list()

BATCH_SIZE = 32


harmful_prompts = harmfulinstructions[:BATCH_SIZE]

hf_path = 'tatsu-lab/alpaca'
benigndataset = load_dataset(hf_path)

# filter for instructions that do not have inputs
benigninstructions = []
for i in range(len(benigndataset['train'])):
    if benigndataset['train'][i]['input'].strip() == '':
        benigninstructions.append(benigndataset['train'][i]['instruction'])


benign_prompts = benigninstructions[:BATCH_SIZE]
#disable gradients because we aren't backpropping
torch.set_grad_enabled(False)


#get the gpu
device = utils.get_device()


#load the model into hooked transformer
model = HookedTransformer.from_pretrained_no_processing("meta-llama/Llama-3.2-1B-Instruct", device=device, default_padding_side='left')



#tokenize
harmfultokens = model.to_tokens(harmful_prompts)

print(harmfultokens)


# run and cache
logits, cache = model.run_with_cache(harmfultokens)

#get all pre-block residuals
thingy = 'resid_pre'
block = 11
#get 14th block's pre-residual
print(cache[thingy, block].shape)

#print(cache[thingy, 12])

#print(cache['blocks.14.hook_resid_pre'])


mean_rejection_activations = cache[thingy, block][:,-1,:].mean(dim=0)




refusal_dir = mean_rejection_activations / mean_rejection_activations.norm()


def refusal_ablation_hook(activation, hook: HookPoint):
    proj = einops.einsum(activation, refusal_dir.reshape(-1,1), )