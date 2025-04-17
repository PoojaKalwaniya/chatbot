from fastapi import FastAPI, Request
from pinecone_setup import get_embedding, index
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Retrieve relevant context from Pinecone
def retrieve_chunks(query, top_k=5):
    query_emb = get_embedding(query)
    res = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    return [match.metadata["text"] for match in res.matches]

# Generate answer using context
def generate_answer(query, context):
    prompt = f"""
You are an AI assistant specialized in the Transformer model, as described in the paper 'Attention is All You Need'.
Use the technical context below to answer the user's question clearly and accurately.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable AI assistant with deep understanding of Transformer architecture and deep learning concepts."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# FastAPI chat endpoint
@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    query = data.get("query", "").strip()

    if not query:
        return {"error": "Query is required."}

    context = "\n".join(retrieve_chunks(query))
    answer = generate_answer(query, context)
    return {"answer": answer}
