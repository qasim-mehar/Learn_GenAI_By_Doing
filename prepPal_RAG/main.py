from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


load_dotenv()

model = ChatMistralAI(
    model="mistral-medium-3-5",
)

data = PyPDFLoader("doc_loader/lec 1.pdf")
docs = data.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
)
chunks = splitter.create_documents(docs)

embedding = MistralAIEmbeddings(model="mistral-embed")


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a great text summarizer with 15 years of experince now, you have to summarize the text",
        ),
        ("human", "{data}"),
    ]
)

final_prompt = prompt_template.format_messages(data=docs)

res = model.invoke(final_prompt)

print(res.content)
