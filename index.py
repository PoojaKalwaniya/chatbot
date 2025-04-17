from fastapi import FastAPI, Request
from pinecone_setup import get_embedding, index  # uses shared setup
from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

# Retrieve relevant context from Pinecone
def retrieve_chunks(query, top_k=5):
    query_emb = get_embedding(query)
    res = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    return [match.metadata["text"] for match in res.matches]

# Generate answer using OpenAI chat
def generate_answer(query, context):
    prompt = f"""
You are an AI assistant specialized in the Transformer model from 'Attention is All You Need'.
Use the context to answer the user's question clearly.

Context:
{context}

Question: {query}
Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with expertise in the Transformer model."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    query = data.get("query", "").strip()

    if not query:
        return {"error": "Query is required."}

    context = "\n".join(retrieve_chunks(query))
    answer = generate_answer(query, context)
    return {"answer": answer}
