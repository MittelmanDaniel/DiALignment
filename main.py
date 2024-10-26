import google.generativeai as genai
import os

api_key = os.getenv('API_KEY')

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
def generate_prompts(user_input):
  response = model.generate_content(f"Create 100 generic prompts for an llm. Include the following topic in each prompt: {user_input}. Write it as a list of prompts only separated by newlines, and include no other text.")
  return response.text.split("\n")

print(generate_prompts("zebra"))