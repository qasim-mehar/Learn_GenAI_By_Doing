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
