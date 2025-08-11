from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=1)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant"),
]

while True:
    user_input = input('YOU : ')
    chat_history.append(HumanMessage(content = user_input))
    if user_input.lower() == 'exit':
        print("Exiting the chat. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI : ", result.content)