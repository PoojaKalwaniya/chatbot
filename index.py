from fastapi import FastAPI, Request
from pinecone_setup import get_embedding, index
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_chunks(query):
    query_emb = get_embedding(query)
    res = index.query(vector=query_emb, top_k=3, include_metadata=True)
    return [m["metadata"]["text"] for m in res["matches"]]

def generate_answer(query, context):
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.
    
Context:
{context}

Question: {query}"""
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    query = data.get("query", "")
    context = "\n".join(retrieve_chunks(query))
    answer = generate_answer(query, context)
    return {"answer": answer}
