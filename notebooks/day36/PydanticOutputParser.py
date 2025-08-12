from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

class Person(BaseModel) :
    name : str = Field(description = 'Name of the person')
    age : int = Field(gt = 18, description = 'Age of the person')
    city : str = Field(description = 'Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = 'Generate the name, age and city of the fictional {place} person \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place' : 'India'})
print(result)