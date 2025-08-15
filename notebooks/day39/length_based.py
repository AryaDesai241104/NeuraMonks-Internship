from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../../assests/Beginners_Guide_to_HTML.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 15, # helps to reatian context
    separator = '',
)

result = splitter.split_documents(docs)
print(result[20].page_content)