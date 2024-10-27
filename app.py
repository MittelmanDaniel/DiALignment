from flask import Flask, render_template, request, redirect, session
from qwen_refusal import *
import json as js

import llm_surgeon

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

import pandas as pd
from datasets import load_dataset

import torch

### LOAD QWEN
def load_qwen():
   model = load_model()
   i = 0
   # refusal_dir, harmful_inst_test = initialize_activation_vectors(model)
   with open("refusal_dir.pkl", "rb") as f:
      refusal_dir = pickle.load(f)
   return model, refusal_dir

### LOAD LLAMA
def load_llama():
   HARMFUL_BATCH = 128
   BENIGN_BATCH = 128
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
   #GET VECTOR
   LAYER = 7
   n_samples = min(len(benign_prompts), len(harmful_prompts))
   print(n_samples)
   batch_size = 32
   refusal_dir = torch.zeros((model.cfg.d_model,), dtype=model.cfg.dtype, device=device)
   for i in range(0, n_samples, batch_size):
      refusal_dir += llm_surgeon.remove_vec_from_benign_to_mean(model, benign_prompts[i:i+batch_size], harmful_prompts[i:i+batch_size], LAYER)
   WEIGHT = 0.001
   refusal_dir = refusal_dir / refusal_dir.norm()
   return llm_surgeon, model, refusal_dir

model, refusal_dir = load_qwen()
llama_surgeon, llama_model, llama_refusal = load_llama()

app = Flask(__name__)

@app.route('/test')
def test():
   return "Hello, World!"

@app.route('/ablate/qwen', methods=['POST'])
def hello():
   json = request.get_json(force=True)
   message = json['message']
   amplitude = json['amplitude']

   print(json)

   if amplitude >= 0:
      baseline_generations, intervention_generations = ablate_refusal(model, refusal_dir, [message], amplitude)
   else:
      baseline_generations, intervention_generations = amplify_refusal(model, refusal_dir, [message], abs(amplitude))
   return {"baseline": baseline_generations, "intervention": intervention_generations}

@app.route('/ablate/llama', methods=['POST'])
def llama():
   json = request.get_json(force=True)
   message = json['message']
   amplitude = json['amplitude']

   print(json)

   eval_chat = [
      {"role": "user", "content": message},
   ]
   llama_model.reset_hooks()
   eval_tokens = llama_model.tokenizer.apply_chat_template(eval_chat, tokenize=False)
   benign = llama_model.generate(eval_tokens, max_new_tokens=256)
   benign = benign[len(eval_tokens):]

   llm_surgeon.add_refusal_hook(llama_model, list(range(llama_model.cfg.n_layers)), llama_refusal)
   intervention = llama_model.generate(eval_tokens, max_new_tokens=256)
   intervention = intervention[len(eval_tokens):]

   return {"baseline": benign, "intervention": intervention}

if __name__ == '__main__':
   app.run(port=5000, debug=True)
