from fastapi import FastAPI, Request
from pinecone_setup import get_embedding, index
from dotenv import load_dotenv
from mangum import Mangum
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    query = data.get("query", "").strip()

    if not query:
        return {"error": "Query is required."}

    query_emb = get_embedding(query)
    res = index.query(vector=query_emb, top_k=5, include_metadata=True)
    context = "\n".join([m.metadata["text"] for m in res.matches])

    prompt = f"""
You are an AI assistant that answers questions based on the Transformer model as described in 'Attention is All You Need'.
Use the context below to help answer the question:

Context:
{context}

Question: {query}
Answer:
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return {"answer": response["choices"][0]["message"]["content"].strip()}

handler = Mangum(app)
