from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableParallel

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

prompt1 = PromptTemplate(
    template='Generate Short and Simple Notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short MCQ Question from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

text = """
All functionality related to Google Cloud, Google Gemini and other Google products.

Google Generative AI (Gemini API & AI Studio): Access Google Gemini models directly via the Gemini API. Use Google AI Studio for rapid prototyping and get started quickly with the langchain-google-genai package. This is often the best starting point for individual developers.
Google Cloud (Vertex AI & other services): Access Gemini models, Vertex AI Model Garden and a wide range of cloud services (databases, storage, document AI, etc.) via the Google Cloud Platform. Use the langchain-google-vertexai package for Vertex AI models and specific packages (e.g., langchain-google-cloud-sql-pg, langchain-google-community) for other cloud services. This is ideal for developers already using Google Cloud or needing enterprise features like MLOps, specific model tuning or enterprise support.
See Google's guide on migrating from the Gemini API to Vertex AI for more details on the differences.

Integration packages for Gemini models and the Vertex AI platform are maintained in the langchain-google repository. You can find a host of LangChain integrations with other Google APIs and services in the googleapis Github organization and the langchain-google-community package.

Google Generative AI (Gemini API & AI Studio)
Access Google Gemini models directly using the Gemini API, best suited for rapid development and experimentation. Gemini models are available in Google AI Studio.
"""

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model | parser,
    'quiz' : prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain
result = chain.invoke({'text' : text})
print(result)
chain.get_graph().print_ascii()