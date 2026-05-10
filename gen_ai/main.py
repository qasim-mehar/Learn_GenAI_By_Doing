from IPython.display import display, Markdown
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def show(text):
    display(Markdown(text))


model = init_chat_model(model="mistral-medium-3-5", max_tokens=1000)
responses = model.batch(
    [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?",
    ]
)
for response in responses:
    show(response.content)
