from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

llm = ChatMistralAI(
    model="mistral-medium-3-5",
)

parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a code genrator"), ("human", "write code about {topic}")]
)

explaination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a code explainer teacher"),
        ("human", "Explain the following code snippet like a expert teacher /n {code}"),
    ]
)

seq1 = code_prompt | llm | parser

seq2 = RunnableParallel(
    {"code": RunnablePassthrough(), "explaination": explaination_prompt | llm | parser}
)

chain = seq1 | seq2

res = chain.invoke("two sum in c++")

print(res["code"])
print(res["explaination"])
