import pinecone
from dotenv import load_dotenv
import openai
import os

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

index_name = os.getenv("PINECONE_INDEX_NAME", "chatbotindex")
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

index = pinecone.Index(index_name)

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding