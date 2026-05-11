from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(
    model="mistral-medium-3-5",
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        """
You are an expert data extractor. Please extract the following details from the movie description provided below.

Format your output as a plain text list with the field name followed by its value. If a specific piece of information is not explicitly mentioned in the text, you must return the word "null" for that field. Do not use JSON formatting.

Required Fields:
- Title:
- Release Year:
- Director:
- Cast:
- Premise:
- Box Office:
- IMDb Rating:
- Music Composer:

Movie Description:
{text}

Extraction:
"""
    ]
)
