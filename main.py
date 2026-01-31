from fastapi import FastAPI, UploadFile, File
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384
index = faiss.IndexFlatL2(dimension)
stored_chunks = []

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    text = ""

    if file.filename.endswith(".txt"):
        text = content.decode("utf-8")

    elif file.filename.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()

    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    index.add(np.array(embeddings).astype("float32"))
    stored_chunks.extend(chunks)

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_vectors_in_index": index.ntotal
    }

@app.post("/search")
async def search(query: str, top_k: int = 3):
    query_embedding = model.encode([query])
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    results = []
    for i in indices[0]:
        results.append(stored_chunks[i])

    return {
        "query": query,
        "results": results
    }
