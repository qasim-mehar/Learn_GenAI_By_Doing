from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
url = "https://www.apple.com/macbook-neo/"
model = ChatMistralAI(
    model="mistral-medium-3-5",
)

data = WebBaseLoader(url)
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
final_template = template_prompt.format_messages(data=docs[0].page_content)
res = model.invoke(final_template)
print(res.content)
