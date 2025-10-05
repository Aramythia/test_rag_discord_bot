import os
import discord
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from openai import AzureOpenAI


class RAG:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version="2023-07-01-preview"
        )
        self.messages = pd.DataFrame(columns=["user", "message", "timestamp"])
        # self.chunks = pd.DataFrame(columns=["chunk", "embedding"])

    @staticmethod
    def prepare_message(message: pd.Series) -> str:
        return f"MESSAGE[({message['timestamp']}, {message['user']}): {message['message']}]"

    def add_message(self, message: discord.Message):
        self.messages = pd.concat([
            self.messages, 
            pd.DataFrame([
                {"user": message.author.name, "message": message.content, "timestamp": message.created_at}
            ])], ignore_index=True)

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
    
    def create_embeddings(self, text: str, model: str = "text-embedding-ada-002"):
        # Create embeddings for each document chunk
        embeddings = self.client.embeddings.create(input=text, model=model).data[0].embedding
        return embeddings

    def embed_chunks(self):
        chunks = self.chunk_messages(self.messages, 4, 2)
        chunk_df = pd.DataFrame({"chunk": [" ".join(chunk) for chunk in chunks], "embedding": [None]*len(chunks)})
        chunk_df["embedding"] = chunk_df['chunk'].apply(self.create_embeddings)
        print(chunk_df)
        return chunk_df
    
    def query(self, query: str):
        # Load vector embeddings
        chunk_df = self.embed_chunks()
        embeddings = chunk_df["embedding"].tolist()
        nbrs = NearestNeighbors(n_neighbors=5).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        chunk_df['indices'] = indices.tolist()
        chunk_df['distances'] = distances.tolist()

        # Embed the question
        query_vector = self.create_embeddings(query)
        distances, indices = nbrs.kneighbors([query_vector])
        context = "Context:\n"
        for index in indices[0]:
            context += f"- {chunk_df.iloc[index]['chunk']}\n"

        # Create the message
        prompt = [
            {'role': 'system', 'content': 'You are a helpful assistant that uses the provided context to answer the question. If you cannot answer a question with complete certainty, say "I do not know".'},
            {'role': 'system', 'content': context},
            {'role': 'user', 'content': query}
        ]
        print(prompt)

        response = self.client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
            messages=prompt,
            max_tokens=1000,
            temperature=0.7,
        )
        return response