import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

load_dotenv()
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME ="gemini-2.0-flash"


if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set BASE_URL, API_KEY, MODEL_NAME via env var or code."
    )


client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)


@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main():
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=MODEL_NAME,
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())