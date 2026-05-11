from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()


class movieSummarySchema(BaseModel):
    title: str
    release_year: str
    director: str
    genre: List[str]
    cast: List[str]
    premise: str
    IMDb_rating: float
    music_composer: Optional[str]


parse = PydanticOutputParser(pydantic_object=movieSummarySchema)

model = ChatMistralAI(
    model="mistral-medium-3-5",
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system"
            """
      You are an expert data extractor. Please extract the following details from the movie description provided below.
      output should be in {outputSchema}
      """
        ),
        ("human", "{text}"),
    ]
)
movie_paragraph = input("Enter movie info: ")
final_prompt = prompt_template.invoke(
    {"outputSchema": parse.get_format_instructions(), "text": movie_paragraph}
)
res = model.invoke(final_prompt)

print(res.content)
