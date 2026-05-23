import os
import requests
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import ToolMessage, HumanMessage
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

    # We use units=metric to get Celsius. Change to units=imperial for Fahrenheit.
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        response.raise_for_status()  # This will throw an error if the city isn't found
        data = response.json()

        temp = data["main"]["temp"]
        condition = data["weather"][0]["description"]

        return f"The current weather in {city} is {temp}°C with {condition}."

    except requests.exceptions.HTTPError:
        return f"Could not find weather for '{city}'. Please check if the city name is correct."
    except Exception as e:
        return f"An error occurred while fetching the weather: {str(e)}"
