from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

schema = [
    ResponseSchema(name = 'fact_1', description = 'Fact 1 about the topic'),
    ResponseSchema(name = 'fact_2', description = 'Fact 2 about the topic'),
    ResponseSchema(name = 'fact_3', description = 'Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give three facts about the {topic} \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic' : 'black hole'})
print(result)
