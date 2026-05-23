import os
import requests
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import wrap_tool_call
from tavily import TavilyClient
from langchain.tools import tool
from rich import print

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Return the current weather data of a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Error: OPENWEATHER_API_KEY not found in environment variables."

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        temp = data["main"]["temp"]
        condition = data["weather"][0]["description"]

        return f"The current weather in {city} is {temp}°C with {condition}."
    except requests.exceptions.HTTPError:
        return f"Could not find weather for '{city}'. Please check if the city name is correct."
    except Exception as e:
        return f"An error occurred while fetching the weather: {str(e)}"


@tool
def get_news(city: str) -> str:
    """Return the current top 3 top news of a city"""
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "Error: TAVILY_API_KEY not found in environment variables."

    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=f"latest news in {city}", topic="news", max_results=3
        )
        results = response.get("results", [])

        if not results:
            return f"No recent news found for {city}."

        news_items = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "No Title")
            content = item.get("content", "No summary available.")
            news_items.append(f"{i}. {title}\nSummary: {content}")

        return "\n\n".join(news_items)
    except Exception as e:
        return f"An error occurred while fetching the news: {str(e)}"


llm = ChatMistralAI(model="mistral-medium-3-5")


@wrap_tool_call
def human_approval(request, handler):
    """Simply ask human approval before using any tool"""
    tool_name = request.tool_call["name"]

    confirm = input(f"Agent wants to use the {tool_name} tool, approve yes/no: ")
    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool approval denied by user.",
            tool_call_id=request.tool_call["id"],
        )

    return handler(request)


agent = create_agent(
    model=llm,
    tools=[get_news, get_weather],
    system_prompt="You are an intelligent assistant and now you have to assist human users related to city queries using tools.",
    middleware=[human_approval],
)

print("SMART CITY AGENT\n")
print("Press 0 to exit")

while True:
    prompt = input("You: ")
    if prompt == "0":
        break

    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
    )

    print(agent_response["messages"][-1].content)
