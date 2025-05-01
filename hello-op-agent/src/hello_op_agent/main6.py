import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.run import RunConfig

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
client = AsyncOpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

model=OpenAIChatCompletionsModel(model="gemini-2.0-flash",openai_client=client)

async def main():
    '''Use OpenAI agent with Google provider gemini model using RUN config with STREAM '''
    
    config = RunConfig(
        model=model,
        model_provider=client,
        tracing_disabled=True
    )
    
    agent = Agent(
        name ="Assistant",
        instructions="You're are helpful assistant.",
    )
    
    result = Runner.run_streamed(agent, "Hello, What you do?", run_config=config)
    # print(result.final_output)
    async for e in result.stream_events():
        print(e)


if __name__ == "__main__":
    asyncio.run(main())
