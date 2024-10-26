import streamlit as st
from transformers import pipeline
import time
from anthropic import Anthropic
import google.generativeai as genai
import random
import ast
import os

# Configure the Google generative model
# Add API KEYS
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Set up Anthropic client
client = Anthropic(api_key=anthropic_api_key)

# Title and description
st.title("DiALignment")
st.write("Compare responses between two models based on a chosen topic and intensity.")

# User input for the initial topic and intensity
c1, c2 = st.columns(2)
with c1:
    init_prompt = st.text_input("Enter your initial prompt for the LLM:", "What is AI?")
with c2:
    intensity = st.slider("Choose the intensity level (0-10):", -1.0, 1.0, 0.0)

# Model selection dropdown
model_choice = st.selectbox("Choose a model to generate prompts:", ("Gemini", "Claude"))


# Define function to generate prompts using the Gemini model
def generate_with_gemini(prompt):
    # Simulating response from Gemini
    response = {
        "text": "['What is artificial intelligence?', 'Explain the concept of AI in simple terms.', 'What are the main types of AI?']"
    }

    # Clean up the response.text to remove unwanted characters
    clean_text = response['text'].strip(" \n").replace("```python", "").replace("```", "")

    # Use ast.literal_eval to safely evaluate the string
    try:
        prompts = ast.literal_eval(clean_text)

        # Ensure it is a list
        if isinstance(prompts, list):
            # Choose a random prompt or select the first one
            selected_prompt = random.choice(prompts)
            return selected_prompt
        else:
            raise ValueError("Parsed response is not a list.")
    except (SyntaxError, ValueError) as e:
        print("Error parsing prompts:", e)
        return None


# Define function to generate prompts using Claude model
def generate_with_claude(prompt, intensity=5):
    prompt_message = f"Generate a list of prompts about {prompt} suitable for a large language model. Only give the response in a Python list format and do not include any other text."

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        messages=[
            {"role": "user", "content":f"{prompt_message}"}
        ]
    )

    prompts = ast.literal_eval(response.content[0].text)

    # Check if it's a list
    if isinstance(prompts, list):
        # Access individual strings
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i + 1}: {prompt}")
    else:
        print("The content is not a list.")

    return prompts[0]



# Generate responses based on the chosen model
def generate_response(prompt, model, intensity=5):
    if model == "Gemini":
        return generate_with_gemini(prompt)
    elif model == "Claude":
        return generate_with_claude(prompt, intensity)



# Streamed response emulator
def response_display(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Button to send the initial prompt to the chatbot
if st.button("Generate Responses"):
    with st.spinner("Generating..."):
        # Generate responses for the normal and modified intensity values
        normal_response = generate_response(init_prompt, model_choice, intensity=5)
        modified_response = generate_response(init_prompt, model_choice, intensity=intensity)

    # Chat window style output
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Normal Model Response")
        normal_response_content = "".join(response_display(normal_response))
        st.markdown(normal_response_content)

    with col2:
        st.subheader("Modified Model Response")
        modified_response_content = "".join(response_display(modified_response))
        st.markdown(modified_response_content)

