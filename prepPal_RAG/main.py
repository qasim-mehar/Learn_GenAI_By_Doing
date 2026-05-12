from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatMistralAI(
    model="mistral-medium-3-5",
)

data = TextLoader("temp.txt")
docs = data.load()

# print(docs.page_content)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a great text summarizer with 15 years of experince now, you have to summarize the text",
        ),
        ("human", "{data}"),
    ]
)

final_prompt = prompt_template.format_messages(data=docs[0].page_content)

res = model.invoke(final_prompt)

print(res.content)
