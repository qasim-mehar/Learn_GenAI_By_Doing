from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from rich import print
from langchain.tools import tool

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Return the number of characters in the text"""
    return len(text)


@tool
def get_word_count(text: str) -> int:
    """Return the number of words in the text"""
    return len(text.split())


tools_list = [get_text_length, get_word_count]
tools_dict = {t.name: t for t in tools_list}

llm = ChatMistralAI(model="mistral-medium-3-5")

llm_with_tool = llm.bind_tools(tools_list, tool_choice="any")

messages = [
    SystemMessage(
        content="You are a helpful text analysis assistant. Even if the user just types a single word, use your tools to analyze it and clearly tell them the length."
    )
]

prompt = input("You: ")
query = HumanMessage(prompt)
messages.append(query)

llm_result = llm_with_tool.invoke(messages)
messages.append(llm_result)
