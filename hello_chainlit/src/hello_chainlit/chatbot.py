import asyncio
import chainlit as cl

@cl.on_message
async def main(message:cl.Message):
    await cl.Message(content=f"Message Received: {message.content}").send()
    
if __name__=="__main__":
    asyncio.run(main())