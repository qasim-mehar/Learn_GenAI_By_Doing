from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(model="mistral-medium-3-5")
print("============ Welcome ==================")
print("Press 0 to exit.")
messages = []

while True:
    prompt = input("You: ")
    messages.append(prompt)
    if prompt == "0":
        break
    res = model.invoke(messages)
    messages.append(res.content)
    print("Bot", res.content)
