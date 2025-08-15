from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../../assests/Beginners_Guide_to_HTML.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100, # helps to reatian context
)

result = splitter.split_documents(docs)
print(result[33].page_content)
