from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

data = TextLoader("../temp.txt")
docs = data.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=100,
    chunk_overlap=0,
)

chunks = text_splitter.create_documents([docs[0].page_content])
for i in chunks:
    print(i)
    print()
