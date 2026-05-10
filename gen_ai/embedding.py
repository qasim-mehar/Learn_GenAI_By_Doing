from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from IPython.display import display, Markdown

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
doc = ["You are genius", "THere is much more to do", "I cant loss"]
vector = embedding.embed_documents(doc)
print(vector)
