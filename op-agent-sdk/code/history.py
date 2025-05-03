import os
import asyncio
from dotenv import find_dotenv, load_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig

load_dotenv(find_dotenv())
API_KEY  = os.getenv('GEMINI_API_KEY')
BASE_URL = os.getenv('OPENAI_BASE_URL')
MODEL = 'gemini-2.0-flash'

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
agent = Agent(name='Assisstant', instructions="You're are helpful assistant that can answer the questions.")

# Initialize history variable
history = []

async def qyery_llm(message: str):
    history.append({'role':'user', 'content':message})    
    # with complete history
    result = await Runner.run(agent, input=history, run_config=config)
    print(result.final_output)
    history.append({'role':'assistant', 'content':result.final_output})

async def main():
    while True:
        user_input = input('Type your message. Press 0 to exit the conversation:\n')
        if user_input == '0':
            break
        else:
            await qyery_llm(user_input)

if __name__ == "__main__":
    asyncio.run(main())


