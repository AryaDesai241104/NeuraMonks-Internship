from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Schema
json_schema = {
    "title": "Review",
    "description": "A detailed review of a restaurant.",
    "type": "object",
    "properties": {
        "key_themes": {
            "title": "Key Themes",
            "description": "All the key themes discussed in the review.",
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "summary": {
            "title": "Summary",
            "description": "A brief summary of the review.",
            "type": "string"
        },
        "sentiment": {
            "title": "Sentiment",
            "description": "The overall sentiment of the review.",
            "type": "string",
            "enum": ["pos", "neg", "neu"]
        },
        "pros": {
            "title": "Pros",
            "description": "All the pros mentioned in a list.",
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                {
                    "type": "null"
                }
            ]
        },
        "cons": {
            "title": "Cons",
            "description": "All the cons mentioned in a list.",
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                {
                    "type": "null"
                }
            ]
        },
        "name": {
            "title": "Name",
            "description": "The name of the reviewer.",
            "anyOf": [
                {
                    "type": "string"
                },
                {
                    "type": "null"
                }
            ]
        }
    },
    "required": [
        "key_themes",
        "summary",
        "sentiment"
    ]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""My recent dining experience at "The Green Leaf Bistro" in Ahmedabad was a delightful journey for the senses. From the moment I stepped in, the ambiance was fantastic, with soft, elegant lighting and a touch of greenery that created a serene and inviting atmosphere. It's the perfect spot for a quiet evening with friends or a special celebration.
The service was impeccable—our server was attentive and had an excellent knowledge of the menu, offering great recommendations. We started with the creamy mushroom soup, which was rich and flavorful, and the perfectly toasted bruschetta, topped with fresh, ripe tomatoes.
For the main course, I chose the homemade spinach and ricotta ravioli, and it was outstanding. The pasta was cooked perfectly, and the filling was delicious and well-seasoned. My partner had the paneer tikka masala, and the paneer was incredibly soft and fresh, with a flavorful, rich gravy. We ended our meal with the chocolate lava cake, and it was a decadent and satisfying finish.
While the food and service were exceptional, the portion sizes were a bit smaller than expected for the price. Additionally, the cocktails were a little on the expensive side for their size. Despite this, the quality of the food and the exceptional service make it a place I'd highly recommend for a special occasion.

Review by Arya Desai
""")

data = (
    result[0]['args'] if isinstance(result, list) and result and 'args' in result[0]
    else result.get('args', result) if isinstance(result, dict)
    else {}
)

print('Key Themes : ', data['key_themes'])
print('Summary : ', data['summary'])
print('Sentiment : ', data['sentiment'])
print('Pros : ', data['pros'])
print('Cons : ', data['cons'])
print('Name : ', data['name'])