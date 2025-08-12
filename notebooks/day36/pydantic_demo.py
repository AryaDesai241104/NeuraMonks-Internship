from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated, Optional, Literal
import os
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# The schema is now a Pydantic BaseModel, which is the recommended approach.
class Review(BaseModel):
    """A detailed review of a restaurant."""
    key_themes: Annotated[list[str], Field(description="All the key themes discussed in the review.")]
    summary: Annotated[str, Field(description="A brief summary of the review.")]
    sentiment: Annotated[Literal["pos", "neg", "neu"], Field(description="The overall sentiment of the review.")]
    pros: Annotated[Optional[list[str]], Field(description="All the pros mentioned in a list.")]
    cons: Annotated[Optional[list[str]], Field(description="All the cons mentioned in a list.")]
    name: Annotated[Optional[str], Field(description="The name of the reviewer.")]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""My recent dining experience at "The Green Leaf Bistro" in Ahmedabad was a delightful journey for the senses. From the moment I stepped in, the ambiance was fantastic, with soft, elegant lighting and a touch of greenery that created a serene and inviting atmosphere. It's the perfect spot for a quiet evening with friends or a special celebration.
The service was impeccable—our server was attentive and had an excellent knowledge of the menu, offering great recommendations. We started with the creamy mushroom soup, which was rich and flavorful, and the perfectly toasted bruschetta, topped with fresh, ripe tomatoes.
For the main course, I chose the homemade spinach and ricotta ravioli, and it was outstanding. The pasta was cooked perfectly, and the filling was delicious and well-seasoned. My partner had the paneer tikka masala, and the paneer was incredibly soft and fresh, with a flavorful, rich gravy. We ended our meal with the chocolate lava cake, and it was a decadent and satisfying finish.
While the food and service were exceptional, the portion sizes were a bit smaller than expected for the price. Additionally, the cocktails were a little on the expensive side for their size. Despite this, the quality of the food and the exceptional service make it a place I'd highly recommend for a special occasion.

Review by Arya Desai
""")

# Accessing attributes 
print('Key Themes : ',result.key_themes)
print('Summary : ', result.summary)
print('Sentiment : ', result.sentiment)
print('Pros : ', result.pros)
print('Cons : ', result.cons)
print('Result : ', result.name)