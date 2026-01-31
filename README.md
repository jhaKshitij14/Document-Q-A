# Multi-Document Semantic Search Backend (RAG-Ready)

This project implements the **retrieval layer** of a Retrieval-Augmented Generation (RAG) system.

It allows users to upload multiple documents (PDF/TXT), chunk them, generate embeddings, and perform **semantic search** using FAISS.  
LLM-based answer generation is intentionally **decoupled** and can be integrated later.

---

This project focuses on:
- understanding embeddings
- vector databases
- document chunking
- backend API design

The architecture mirrors how real-world RAG systems are built in industry.

---

## Tech Stack

- **Python**
- **FastAPI** – backend API
- **Sentence-Transformers** – text embeddings
- **FAISS** – vector similarity search
- **PyPDF2** – PDF parsing
- **NumPy**

---

## Features

- Upload **PDF** and **TXT** documents
- Chunking with configurable overlap
- Sentence-transformer embeddings
- FAISS vector indexing
- Semantic search with top-k retrieval
- Metadata tracking (document name, page number)
- Clean and simple REST APIs
- RAG-ready design (LLM can be plugged in later)

---

## API Endpoints

### Upload a document


**Description**
- Upload a PDF or TXT file
- File is parsed, chunked, embedded, and indexed

---

### Semantic search
**Parameters**
- `question` (string)
- `top_k` (optional, default = 3)

**Returns**
- Top matching text chunks
- Source metadata (document, page number)

---

## Example Use Case

1. Upload multiple PDFs and text files
2. Ask a natural-language question
3. Retrieve the most relevant document sections
4. (Future) Pass retrieved context to an LLM for grounded answers

---

## Project Structure
---

## How to Run

### 1. Install dependencies
```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu PyPDF2 numpy