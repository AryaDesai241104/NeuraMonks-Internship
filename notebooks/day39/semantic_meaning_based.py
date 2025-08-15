from langchain_experimental.text_splitter import SemanticChunker # Still a very poor model and therefore the name experimental
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

text = """
"Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety
"""

text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key), 
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

result = text_splitter.split_text(text)
print(result)