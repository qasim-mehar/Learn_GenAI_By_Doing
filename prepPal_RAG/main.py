from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
