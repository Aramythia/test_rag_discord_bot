import discord
import os
from dotenv import load_dotenv

import pandas as pd

load_dotenv() # load all the variables from the env file


class RagBot(discord.Bot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: pd.DataFrame = pd.DataFrame([{"user": "", "message": "", "timestamp": ""}])

    async def on_ready(self):
        print(f"{self.user} is ready and online!")

    async def on_message(self, message: discord.Message):
        print("Message received:", message.content)
        if message.author == self.user:
            return
        if "seehistory" in message.content:
            print(self.messages)
            return
        self.messages = pd.concat([self.messages, pd.DataFrame([{"user": message.author.name, "message": message.content, "timestamp": message.created_at}])], ignore_index=True)
    

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