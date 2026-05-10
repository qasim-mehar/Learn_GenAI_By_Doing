from langchain_mistralai import ChatMistralAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(model="mistral-medium-3-5")

print("============ Welcome ==================")
print("Press 0 to exit.")

persona = input("What persona do you want? e.g funny AI, Angry Teacher etc")

messages = [
    SystemMessage(content=f"You are a {persona}"),
]

while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    res = model.invoke(messages)
    messages.append(AIMessage(content=res.content))
    print("Bot", res.content)

print(messages)
