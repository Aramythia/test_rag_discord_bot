import discord
import os
from dotenv import load_dotenv

import pandas as pd
from openai import AzureOpenAI

load_dotenv() # load all the variables from the env file


class RagBot(discord.Bot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: pd.DataFrame = pd.DataFrame([{"user": "", "message": "", "timestamp": "", "embedding": None}])
        self.client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            base_url=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION')
        )

    def create_embeddings(self, text: str, model: str = "text-embedding-ada-002"):
        # Create embeddings for each document chunk
        embeddings = self.client.embeddings.create(input = text, model=model).data[0].embedding
        return embeddings

    async def on_ready(self):
        print(f"{self.user} is ready and online!")

    async def on_message(self, message: discord.Message):
        print("Message received:", message.content)
        if message.author == self.user:
            return
        if "seehistory" in message.content:
            print(self.messages)
            return
        if "embeddings" in message.content:
            print(self.client)
            return
        self.messages = pd.concat([
            self.messages, 
            pd.DataFrame([
                {"user": message.author.name, "message": message.content, "timestamp": message.created_at, "embedding": None}
            ])], ignore_index=True)
    
    @discord.slash_command(name="test", description="Say hello!")
    async def test(self, ctx: discord.ApplicationContext):
        await ctx.respond("Hello from RagBot!")


bot = RagBot(intents=discord.Intents.all())

@bot.slash_command(name="hello", description="Say hello to the bot")
async def hello(ctx: discord.ApplicationContext):
    await ctx.respond("Hey!")

@bot.slash_command(name="history", description="Get the message history")
async def history(ctx: discord.ApplicationContext):
    limit = 5
    user_messages = ctx.bot.messages[ctx.bot.messages["user"] == ctx.author.name].tail(limit)
    if user_messages.empty:
        await ctx.respond("No message history found.")
    else:
        history_text = "\n".join([f"{row['timestamp']}: {row['message']}" for _, row in user_messages.iterrows()])
        await ctx.respond(f"Your last {limit} messages:\n{history_text}")


bot.run(os.getenv('TOKEN')) # run the bot with the token