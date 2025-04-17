import os
from PyPDF2 import PdfReader
from pinecone_setup import get_embedding, index
from dotenv import load_dotenv

load_dotenv()
filepath ="C:\chatbot\data\NIPS-2017-attention-is-all-you-need-Paper.pdf"
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def ingest_pdf(file_path):
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        index.upsert([(f"pdf-{i}", embedding, {"text": chunk})])
