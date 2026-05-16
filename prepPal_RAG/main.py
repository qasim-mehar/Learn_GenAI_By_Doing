from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma


load_dotenv()

model = ChatMistralAI(
    model="mistral-medium-3-5",
)

data = PyPDFLoader("lec 1.pdf")
docs = data.load()

embedding = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    kwargs={
        "k": 4,
        "fetch_k": 10,
    },
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a great AI assistant with 15 years of experience. Answer the user query using ONLY the given context. If the answer isn't available in the given context, just answer 'I couldn't find the answer'.\n\nContext:\n{context}",
        ),
        ("human", "{query}"),
    ]
)

print("-------------RAG-----------------")
print("Press 0 to exit.")


while True:
    query = input("You: ")
    if query == "0":
        break
    docs = retriever.invoke(query)

    context = "/n/n".join([doc.page_content for doc in docs])

    final_prompt = prompt_template.format_messages(context=context, query=query)
    res = model.invoke(final_prompt)
    print(f"\n AI: {res.content}")
