import os
from dotenv import load_dotenv, find_dotenv, dotenv_values
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Fetch the GOOGLE_API_KEY
api_key = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_prompts(user_input):
  response = model.generate_content(f"Give me 20 general prompts about a {user_input} for a large language model. Only give to me in a python list and do not include any other text.")
  return response.text

print(generate_prompts("zebra"))