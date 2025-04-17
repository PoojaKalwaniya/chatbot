from fastapi import FastAPI, Request
from dotenv import load_dotenv
import pinecone
import os
from openai import OpenAI

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

def get_relevant_chunks(query):
    emb_response = openai_client.embeddings.create(
        input=[query], model="text-embedding-ada-002"
    )
    query_emb = emb_response.data[0].embedding
    search_result = index.query(vector=query_emb, top_k=3, include_metadata=True)
    return [match['metadata']['text'] for match in search_result['matches']]

def ask_chatgpt(query, context):
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    query = data.get("query", "")
    context_chunks = get_relevant_chunks(query)
    context = "\n".join(context_chunks)
    answer = ask_chatgpt(query, context)
    return {"answer": answer}
