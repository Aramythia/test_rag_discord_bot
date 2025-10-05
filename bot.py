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
        self.messages: pd.DataFrame = pd.DataFrame([{"user": "", "message": "", "timestamp": "", "embedding": None}])
        self.client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version="2023-07-01-preview"
        )

    @staticmethod
    def prepare_message(message: pd.Series) -> str:
        return f"MESSAGE[({message['timestamp']}, {message['user']}): {message['message']}]"

    def create_embeddings(self, text: str, model: str = "text-embedding-ada-002"):
        # Create embeddings for each document chunk
        embeddings = self.client.embeddings.create(input=text, model=model).data[0].embedding
        return embeddings
    
    def chunk_messages(self, messages: pd.DataFrame, chunk_size: int = 10, overlap: int = 2):
        chunks = []
        n = len(messages)
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            window = messages.iloc[start:end]
            chunk = [self.prepare_message(row) for _, row in window.iterrows()]
            chunks.append(chunk)
            if end == n:
                break
            start += overlap
        return chunks

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


@bot.slash_command(name="query", description="Query the RAG system")
async def query(ctx: discord.ApplicationContext, question: str):
    rag: RAG = ctx.bot.rag
    response = rag.query(question)
    print(response.choices[0].message.content)
    await ctx.respond(f"Question: {question}\nAnswer: {response.choices[0].message.content}")


@bot.slash_command(name="ask", description="Ask a question to the bot")
async def ask(ctx: discord.ApplicationContext, question: str):
    # Load vector database
    loaded_messages = ctx.bot.messages.dropna(subset=["embedding"])
    embeddings = loaded_messages["embedding"].dropna().tolist()
    nbrs = NearestNeighbors(n_neighbors=5).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    loaded_messages['indices'] = indices.tolist()
    loaded_messages['distances'] = distances.tolist()

    # Embed the question
    query_vector = ctx.bot.create_embeddings(question)
    distances, indices = nbrs.kneighbors([query_vector])
    context = "Context:\n"
    for index in indices[0]:
        context += f"- {loaded_messages.iloc[index]['message']}\n"

    # Create the message
    prompt = [
        {'role': 'system', 'content': 'You are a helpful assistant that uses the provided context to answer the question. If you cannot answer a question with complete certainty, say "I do not know".'},
        {'role': 'system', 'content': context},
        {'role': 'user', 'content': question}
    ]
    print(prompt)

    response = ctx.bot.client.chat.completions.create(
        model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        messages=prompt,
        max_tokens=1000,
        temperature=0.7,
    )
    print("Response:", response.choices[0].message.content)

    await ctx.respond(response.choices[0].message.content)


@bot.slash_command(name="history", description="Get the message history")
async def history(ctx: discord.ApplicationContext, limit: int = 5):
    user_messages = ctx.bot.messages.tail(limit)
    if user_messages.empty:
        await ctx.respond("No message history found.")
    else:
        history_text = "\n".join([f"{row['timestamp']}: {row['message']}" for _, row in user_messages.iterrows()])
        await ctx.respond(f"Your last {limit} messages:\n{history_text}")


@bot.slash_command(name="chunks", description="Get the message chunks")
async def chunks(ctx: discord.ApplicationContext):
    if ctx.bot.messages.empty:
        await ctx.respond("No messages to chunk.")
        return

    message_chunks = ctx.bot.chunk_messages(ctx.bot.messages)
    chunk_texts = ["\n".join(chunk) for chunk in message_chunks]
    response_text = "\n\n---\n\n".join(chunk_texts)
    print(response_text)
    await ctx.respond(f"Message chunks:\n{message_chunks[0]}")


bot.run(os.getenv('TOKEN')) # run the bot with the token