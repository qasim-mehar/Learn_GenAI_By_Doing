from IPython.display import display, Markdown
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def show(text):
    display(Markdown(text))


model = init_chat_model(model="mistral-medium-3-5", max_tokens=1000)
"""responses = model.batch(
    [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?",
    ]
)
for response in responses:
    show(response.content)"""

res = model.invoke(
    "I'm Learning gen ai now write a really brief and saturated description for my github repo 'Learn_GenAI_By_Doing' tell them what i have learn throuout gen ai journey in >350 chars "
)
show(res.content)
