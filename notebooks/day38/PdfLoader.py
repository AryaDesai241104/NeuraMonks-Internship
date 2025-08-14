from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

loader = PyPDFLoader('../../assests/Beginners_Guide_to_HTML.pdf')
docs = loader.load()

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = GoogleGenerativeAI(model = 'gemini-1.5-flash', google_api_key = api_key)

print(docs[1].page_content)