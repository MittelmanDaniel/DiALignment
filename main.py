# Import the Python SDK
import google.generativeai as genai

genai.configure(api_key="AIzaSyBv2e8LO9VwnBgAIYcSEJNLsMy2LyKucT4")
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_prompts(user_input):
  response = model.generate_content(f"Give me 100 general prompts about a {user_input} for a large language model. Only give to me in a python list and do not include any other text.")
  return response.text