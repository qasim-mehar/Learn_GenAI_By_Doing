from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import TextLoader

data = TextLoader("../temp.txt")
docs = data.load()

splitter = TokenTextSplitter(
    chunk_size=10,
    chunk_overlap=1,
)

chunks = splitter.create_documents([docs[0].page_content])

for i in chunks:
    print(i.page_content)
    print()
