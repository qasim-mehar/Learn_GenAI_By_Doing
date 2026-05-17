from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda

load_dotenv()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} breifly in 1 to 2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail")
model = ChatMistralAI(
    model="mistral-medium-3-5",
)

parser = StrOutputParser()

chain = RunnableParallel(
    {
        "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
        "detailed": RunnableLambda(lambda x: x["detailed"])
        | detailed_prompt
        | model
        | parser,
    }
)

res = chain.invoke(
    {
        "short": {"topic": "Machine learning"},
        "detailed": {"topic": "why Machine learning"},
    }
)
print(res["short"])
print(res["detailed"])
