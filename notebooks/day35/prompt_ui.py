from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature = 0.5)

st.header('Research Assistant')

paper_input = st.selectbox('Select Research Paper Name',[
    'Attention Is All You Need', 
    'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', 
    'GPT-3: Language Models are Few-Shot Learners',
    'Diffusion Models Beat GANs on Image Synthesis'
])

style_input = st.selectbox('Select Explaination Style',[
    'Beginner-Friendly',
    'Technical',
    'Code-Oriented',
    'Mathematical'
])

length_input = st.selectbox('Select Length of Explaination',[
    'Short(1-2 paragraphs)',
    'Medium(3-5 paragraphs)',
    'Long(detailed explanation)'
])

template = load_prompt('templates.json')

input_variables = ['paper_input', 'style_input', 'length_input']

# fill the placeholders in the template
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input    
})

if st.button('Summarize') :
    result = model.invoke(prompt)
    st.write(result.content)