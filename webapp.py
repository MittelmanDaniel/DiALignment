import streamlit as st
from transformers import pipeline
import time
from anthropic import Anthropic
import google.generativeai as genai
import random
import ast
import os
import pandas as pd
import plotly.express as px


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
    response = gemini_model.generate_content(f"Create 100 generic prompts for an llm. Include the following topic in each prompt: {prompt}. Write it as a list of prompts only separated by newlines, and include no other text.")
    print(response.text)
    print(type(response.text))

    prompts = response.text
    print(type(prompts))
    # Clean up the response.text to remove unwanted characters
    clean_text = prompts.split("\n")
    print(clean_text)
    print(type(clean_text))
    return clean_text


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

    return prompts



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
        normal_prompts = generate_response(init_prompt, model_choice, intensity=5)
        modified_prompts = generate_response(init_prompt, model_choice, intensity=intensity)

    # Chat window style output
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Normal Model Response")
        normal_response_content = "".join(response_display(normal_prompts[0]))
        st.markdown(normal_response_content)

    with col2:
        st.subheader("Modified Model Response")
        modified_response_content = "".join(response_display(modified_prompts[0]))
        st.markdown(modified_response_content)



'''
# This is the data is that is from the results from ryan and gang
df = pd.DataFrame(data)

st.dataframe(df)

sb_options = list(df.columns)
sb_options.remove('Name')

colselect1, colselect2 = st.columns(2)
with colselect1:
    sel_metric = st.selectbox('Select the Metric', options=sb_options)

# Create a new dataframe for plotting. Sort it for plotting display.
# If metric value is the same, display name by alphabetic order.
df_metric = df.copy()
df_metric = df_metric.sort_values(by=[sel_metric, 'Name'], ascending=[True, False])

plot_assym = px.bar(df_metric, x=sel_metric, y="Name", template="plotly_white", text_auto=True)
st.plotly_chart(plot_assym)
'''

