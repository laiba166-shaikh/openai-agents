import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_default_openai_client, set_default_openai_api

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
client = AsyncOpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# what's the point?
set_default_openai_api("chat_completions")
set_default_openai_client(client)

model=OpenAIChatCompletionsModel(model="gemini-2.0-flash",openai_client=client)

def main():
    '''Use OpenAI agent with Google provider gemini model Global'''
    
    agent = Agent(
        name ="Assistant",
        instructions="You're are helpful assistant.",
        model=model
    )
    
    result = Runner.run_sync(agent, "Hello, What you do?")
    print(result.final_output)

if __name__ == "__main__":
    main()
