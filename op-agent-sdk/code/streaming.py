import os
import asyncio
from typing import cast
from dotenv import find_dotenv, load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig

load_dotenv(find_dotenv())
API_KEY  = os.getenv('GEMINI_API_KEY')
BASE_URL = os.getenv('OPENAI_BASE_URL')
MODEL = 'gemini-2.0-flash'

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# PROVIDER
provider = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# MODEL
model = OpenAIChatCompletionsModel(model=MODEL,openai_client=provider)

# CONFIG
config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# AGENT

async def main():
    try:
        agent = Agent(name="Assistant", instructions="You are a helpful assistant")
        # Run the agent with streaming enabled
        result = Runner.run_streamed(agent, input='Write a 500 words article on AI Agents and the modern world.', run_config=config)

        # Stream the response token by token
        async for event in result.stream_events():
            # if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                token = event.data.delta
                print(token, end="", flush=True)
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())