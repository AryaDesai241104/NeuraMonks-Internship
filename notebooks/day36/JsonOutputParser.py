from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})

print(result)
print(type(result))