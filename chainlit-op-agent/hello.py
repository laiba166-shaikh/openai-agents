import chainlit as cl
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

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set('history', [])
    await cl.Message(content=f"Hello, I am your beautiful helful Assistant. How can I help you today").send()

@cl.on_message
async def main(message:cl.Message):
    history = cl.user_session.get('history')
    history.append({'role':'user', 'content':message.content})
    # RUNNER - LLM call with config
    # result = await Runner.run(agent, input=message.content, run_config=config)
    
    # with complete history
    result = await Runner.run(agent, input=history, run_config=config)
    print(result.final_output)
    history.append({'role':'assistant', 'content':result.final_output})
    cl.user_session.set('history', history)
    await cl.Message(content=f"Message Received: {result.final_output}").send()
