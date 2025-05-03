import chainlit as cl
import os
import asyncio
from typing import cast
from dotenv import find_dotenv, load_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig

load_dotenv(find_dotenv())
API_KEY  = os.getenv('GEMINI_API_KEY')
BASE_URL = os.getenv('OPENAI_BASE_URL')
MODEL = 'gemini-2.0-flash'

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@cl.on_chat_start
async def on_chat_start():
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

    # Initialize an empty chat history in the session.    cl.user_session.set("chat_history", [])
    cl.user_session.set('history', [])
    cl.user_session.set("config", config)
    # AGENT
    agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant")
    cl.user_session.set("agent", agent)
    await cl.Message(content="Welcome to the Chainlit AI Assistant! How can I help you today?").send()

@cl.on_message
async def main(message:cl.Message):
    history = cl.user_session.get('history') or []
    history.append({'role':'user', "content":message.content})

    # Create a new message object (Assistant message) for streaming
    msg = cl.Message(content="")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        # Run the agent with streaming enabled
        result = Runner.run_streamed(agent, input=history, run_config=config)

        # Stream the response token by token
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
                token = event.data.delta
                await msg.stream_token(token)

        # Append the assistant's response to the history.
        history.append({"role": "assistant", "content": msg.content})

        # Update the session with the new history.
        cl.user_session.set("chat_history", history)

        # Optional: Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {msg.content}")

    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")
        print(f"Error: {str(e)}")
