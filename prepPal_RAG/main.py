from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(
    model="mistral-medium-3-5",
)
res = model.invoke("hey")
print(res.content)
