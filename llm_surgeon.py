import transformer_lens.utils as utils

from typing import List
import torch

from transformer_lens import HookedTransformer

import functools

from transformer_lens.hook_points import (
    HookPoint,
) 


    #takes in a list of prompts
def convert_prompts_to_chat_template(prompts):
    return  [[{"role":"user", "content": prompt}] for prompt in prompts]



#prompt is a list of lists of dicts, where the internal list of dicts is 
def get_mean_activations(model: HookedTransformer, prompts):

    convers = convert_prompts_to_chat_template(prompts)
    tokens = model.tokenizer.apply_chat_template(convers, tokenize=False)

    logits, cache = model.run_with_cache(tokens, names_filter=lambda hook_name: 'resid' in hook_name)

    return logits, cache


def remove_vec_from_benign_to_mean(model: HookedTransformer, benign_prompts, harmful_prompts, layer, section = 'resid_pre'):
    benign_logits, benign_cache = get_mean_activations(model, benign_prompts)

    harmful_logits, harmful_cache = get_mean_activations(model, harmful_prompts)

    mean_harmful_activations = harmful_cache[section, layer][:,-1,:].mean(dim=0)

    mean_benign_activations = benign_cache[section, layer][:, -1, :].mean(dim=0)

    refusal_vec = mean_harmful_activations - mean_benign_activations

    refusal_dir = refusal_vec/refusal_vec.norm()

    return refusal_dir


def refusal_removal_hook(activation: torch.Tensor, hook: HookPoint, refusal_dir: torch.Tensor):
    elementwise_product = activation * refusal_dir.view(-1)
    projection_scalar = elementwise_product.sum(dim=-1, keepdim=True)

    proj = projection_scalar * refusal_dir

    return activation - proj

def add_refusal_hook(model: HookedTransformer, intervention_layers: List[int], refusal_dir):
    for l in intervention_layers:
        model.add_hook(utils.get_act_name('resid_pre',l), functools.partial(refusal_removal_hook, refusal_dir=refusal_dir))


        
def add_hook(activation: torch.Tensor, hook: HookPoint, refusal_dir: torch.Tensor, scaling_factor: int):
    orig_norm = activation.norm()

    new_dir = (1-scaling_factor) * activation/orig_norm +  refusal_dir * scaling_factor

    new_dir = new_dir

    return orig_norm * new_dir


def add_dir_hook(model: HookedTransformer, intervention_layers: List[int], refusal_dir, scaling_factor):
    for l in intervention_layers:
        model.add_hook(utils.get_act_name('resid_pre',l), functools.partial(add_hook, refusal_dir=refusal_dir, scaling_factor=scaling_factor))