from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

prompt = PromptTemplate(
    template = 'Write a summary for the given document - \n {poem}',
    input_variables = ['poem']
)

parser = StrOutputParser()

loader = TextLoader('../../assests/football.txt')
docs = loader.load()

chain = prompt | model | parser
result  = chain.invoke({'poem' : docs[0].page_content})
print(result)

