import discord
import json
import os
from dotenv import load_dotenv

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from openai import AzureOpenAI

from utils.rag import RAG

load_dotenv() # load all the variables from the env file


class RagBot(discord.Bot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rag = RAG()

    async def on_ready(self):
        print(f"{self.user} is ready and online!")

    async def on_message(self, message: discord.Message):
        print("Message received:", message.content)
        if message.author == self.user:
            return
        if "seehistory" in message.content:
            print(self.rag.messages)
            return
        if "embeddings" in message.content:
            self.rag.embed_chunks()
        self.rag.add_message(message)


bot = RagBot(intents=discord.Intents.all(), debug_guilds=[331126066850824192])


@bot.slash_command(name="hello", description="Say hello to the bot")
async def hello(ctx: discord.ApplicationContext):
    await ctx.respond("Hey!")


@bot.slash_command(name="ask", description="Ask a question to the bot.")
async def ask(ctx: discord.ApplicationContext, question: str):
    rag: RAG = ctx.bot.rag
    response = rag.query(question)
    print(response.choices[0].message.content)
    await ctx.respond(f"Question: {question}\nAnswer: {response.choices[0].message.content}")


@bot.slash_command(name="history", description="Get the message history")
async def history(ctx: discord.ApplicationContext, limit: int = 5):
    rag: RAG = ctx.bot.rag
    user_messages = rag.messages.tail(limit)
    if user_messages.empty:
        await ctx.respond("No message history found.")
    else:
        history_text = "\n".join([f"{row['timestamp']}: {row['message']}" for _, row in user_messages.iterrows()])
        await ctx.respond(f"Your last {limit} messages:\n{history_text}")


@bot.slash_command(name="chunks", description="Get the message chunks")
async def chunks(ctx: discord.ApplicationContext):
    rag: RAG = ctx.bot.rag
    if rag.messages.empty:
        await ctx.respond("No messages to chunk.")
        return

    message_chunks = rag.chunk_messages(rag.messages)
    chunk_texts = ["\n".join(chunk) for chunk in message_chunks]
    response_text = "\n\n---\n\n".join(chunk_texts)
    print(response_text)
    await ctx.respond(f"Message chunks:\n{message_chunks[0]}")


bot.run(os.getenv('TOKEN')) # run the bot with the token