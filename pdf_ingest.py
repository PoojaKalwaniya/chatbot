from PyPDF2 import PdfReader
from pinecone_setup import get_embedding, index

filepath = "./data/NIPS-2017-attention-is-all-you-need-Paper.pdf"

def extract_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def ingest(path):
    text = extract_text(path)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        index.upsert([(f"chunk-{i}", embedding, {"text": chunk})])

if __name__ == "__main__":
    ingest(filepath)