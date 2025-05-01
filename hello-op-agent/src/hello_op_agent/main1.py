import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
client = AsyncOpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


set_tracing_disabled(disabled=True)

async def main():
    '''Use OpenAI agent with Google provider gemini model at - Agent Level'''
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash",openai_client=client)
    )
    
    result = await Runner.run(agent, "How are you?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
