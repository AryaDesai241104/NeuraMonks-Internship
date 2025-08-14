from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = GoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=api_key)

url = 'https://en.wikipedia.org/wiki/Rohit_Sharma'
loader = WebBaseLoader(url)
docs = loader.load()

prompt = PromptTemplate(
    template='Answer the following question based on the provided text:\n\nQuestion: {question}\n\nText: {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

chain = prompt | model | parser

print("Welcome! I can answer questions about Rohit Sharma based on the Wikipedia page.")
print("Type 'good bye', 'exit', or 'break' to stop the chat.")
print("-" * 70)

while True:
    user_question = input("You : ").strip()

    if user_question.lower() in ['good bye', 'exit', 'break']:
        print("Goodbye! Thanks for chatting.")
        break

    try:
        result = chain.invoke({
            'question': user_question,
            'text': docs[0].page_content
        })
        print(f"Answer: {result} \n")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try your question again or type an exit command.")

