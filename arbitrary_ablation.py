import llm_surgeon

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

import pandas as pd
from datasets import load_dataset

import torch

import google_thing

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


TARGET = "9.11 and 9.8"

target_prompts = google_thing.generate_prompts(TARGET)

target_prompts = target_prompts[:HARMFUL_BATCH]
print(target_prompts)
#print(target_prompts)

torch.set_grad_enabled(False)
device = utils.get_device()


print('based')

model = HookedTransformer.from_pretrained_no_processing("meta-llama/Llama-3.2-1B-Instruct", device=device, default_padding_side='left')

LAYER = 7
target_dir = llm_surgeon.remove_vec_from_benign_to_mean(model, benign_prompts, target_prompts, LAYER)

print("GOT")

llm_surgeon.add_refusal_hook(model, list(range(model.cfg.n_layers)), target_dir)


eval_chat = [
  {"role": "user", "content": "Is 9.11 greater than 9.8"},
]


eval_tokens = model.tokenizer.apply_chat_template(eval_chat, tokenize=False)


output = model.generate(eval_tokens, max_new_tokens=256)

print(output[len(eval_tokens):])