import os
from PyPDF2 import PdfReader
from pinecone_setup import get_embedding, index
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix: Use raw string or escape backslashes in file path
filepath = r"C:\chatbot\data\NIPS-2017-attention-is-all-you-need-Paper.pdf"

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Fix: Handle NoneType in case a page has no text
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def ingest_pdf(file_path):
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        raise ValueError("No text extracted from the PDF.")
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        index.upsert([(f"pdf-{i}", embedding, {"text": chunk})])


