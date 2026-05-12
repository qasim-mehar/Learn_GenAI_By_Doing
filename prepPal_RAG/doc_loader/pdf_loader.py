from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(
    model="mistral-medium-3-5",
)

data = PyPDFLoader("Lec 1.pdf")
docs = data.load()

template_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a great text summarizer with 15 years of experince now, you have to summarize the text",
        ),
        ("human", "{data}"),
    ]
)
final_template = template_prompt.format_messages(data=docs)
res = model.invoke(final_template)
print(res.content)
