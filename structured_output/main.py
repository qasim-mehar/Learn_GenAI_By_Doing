from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class movieSummarySchema(BaseModel):
    title: str
    release_year: str
    director: str
    genre: List[str]
    cast: List[str]
    premise: str
    IMDb_rating: int
    music_composer: str


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
movie_paragraph = input("Enter movie info: ")
final_prompt = prompt_template.invoke({"text": movie_paragraph})
res = model.invoke(final_prompt)

print(res.content)
