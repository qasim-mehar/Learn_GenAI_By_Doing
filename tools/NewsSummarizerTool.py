from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

from dotenv import load_dotenv

load_dotenv()

llm = ChatMistralAI(model="mistral-medium-3-5")

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
  You are expert news summarizer with over 15 years of experince, now summarize these news in bulletpoints /n {news}
  """
)

search_agent = TavilySearch()
news_result = search_agent.invoke({"query": "Google I/O 2026"})

chain = prompt | llm | parser

summary = chain.invoke({"news": news_result})

print(summary)
