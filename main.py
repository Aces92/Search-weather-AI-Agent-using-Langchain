import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

# ======================
# Setup
# ======================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("OPEN_AI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_STACK = os.getenv("WEATHER_STACK")

# ======================
# Tools
# ======================
search_tool = TavilySearch(max_results=3)

@tool
def get_weather_data(city: str) -> str:
    """Get current weather data for a given city using the weatherstack API."""
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={os.getenv('WEATHER_STACK')}&query={city}"
    )
    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not get weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

tools = [search_tool, get_weather_data]

# ======================
# Create Agent
# ======================
def create_research_agent():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=GROQ_API_KEY
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful research assistant. Use tools when you need current or real-time information."
    )
    return agent