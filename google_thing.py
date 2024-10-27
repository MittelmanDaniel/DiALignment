import google.generativeai as genai
import os


from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


api_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_prompts(user_input):
  response = model.generate_content(f"Create 100 generic prompts for an llm. Include the following topic in each prompt: {user_input}. Write it as a list of prompts only separated by newlines, and include no other text.")
  return response.text.split("\n")

def get_ratings(responses: list):
  string_responses = ''.join(responses)
  evaluations = model.generate_content(string_responses).text
  ratings = evaluations.split('\n')
  my_dict = zip(responses, ratings)
  return {response : eval for response in responses for eval in evaluations}
