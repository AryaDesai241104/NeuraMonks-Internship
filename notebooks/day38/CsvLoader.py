from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
import os
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = GoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=api_key)

loader = CSVLoader(file_path = '..\..\datasets\Social_Network_Ads.csv')
docs = loader.load()
print(docs[0].page_content)
