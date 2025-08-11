from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system" , "You are a helpful {domain} Expert."),
    ("human" , "Explain in simple terms, what is {topic}"),
])

prompt = chat_template.invoke({
    'domain': 'Football Expert',
    'topic': 'G.O.A.T'
})