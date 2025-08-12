from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# 1st prompt -> detailed report
temp1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
temp2 = PromptTemplate(
    template = 'Write a five pointer and crisp summary on the following {text}',
    input_variables = ['text']
)

parser = StrOutputParser()
chain = temp1 | model | parser | temp2 | model | parser

result = chain.invoke({'topic' :'black hole'})
print(result)
