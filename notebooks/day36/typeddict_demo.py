from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg", "neu"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

# Pass the TypedDict directly
structured_model = model.with_structured_output(Review)

review_text = """
My recent dining experience at "The Green Leaf Bistro" in Ahmedabad was a delightful journey for the senses...
Review by Arya Desai
"""

result = structured_model.invoke(review_text)

print('Key Themes :', result['key_themes'])
print('Summary :', result['summary'])
print('Sentiment :', result['sentiment'])
print('Pros :', result['pros'])
print('Cons :', result['cons'])
print('Name :', result['name'])
