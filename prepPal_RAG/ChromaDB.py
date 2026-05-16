from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


load_dotenv()

data = PyPDFLoader("doc_loader/lec 1.pdf")
docs = data.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)

embedding = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embedding, persist_directory="chroma_db"
)
print(vectorstore)
