import os
import pinecone
from openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

index = pinecone.Index(index_name)

def get_embedding(text):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

def index_documents(docs):
    for i, doc in enumerate(docs):
        embedding = get_embedding(doc)
        index.upsert([(f"doc-{i}", embedding, {"text": doc})])
