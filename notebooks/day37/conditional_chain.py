from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

parser1 = StrOutputParser()

class Feedback(BaseModel) :
    sentiment : Literal['positive', 'negative'] = Field(description='Give me sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions' : parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = '''
    The following feedback has a negative sentiment:\n\n"
    "{feedback}\n\n"
    "Write a concise, empathetic, and professional reply in 2-3 sentences "
    "addressing the concern directly without giving general guidelines."
    ''',   
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='''
        "The following feedback has a positive sentiment:\n\n"
        "{feedback}\n\n"
        "Write a concise, warm, and professional reply in 2-3 sentences, "
        "expressing appreciation and reinforcing a positive connection."
    ''',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'negative', prompt2 | model | parser1),
    (lambda x : x.sentiment == 'positive', prompt3 | model | parser1),
    RunnableLambda(lambda X : 'could not find sentiment')
)

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback' : "The update is not adding value instead has slower down the laptop's working"})
print(result)
chain.get_graph().print_ascii()