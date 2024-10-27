from flask import Flask, render_template, request, redirect, session
from qwen_refusal import *
import json as js

model = load_model()
i = 0
# refusal_dir, harmful_inst_test = initialize_activation_vectors(model)
with open("refusal_dir.pkl", "rb") as f:
   refusal_dir = pickle.load(f)

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


if __name__ == '__main__':
   app.run(port=5000, debug=True)
