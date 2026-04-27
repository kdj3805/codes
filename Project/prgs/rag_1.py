# ============================================================
# ENTERPRISE PDF → FULL HYBRID RAG SYSTEM (ALL-IN-ONE)
# Extraction + Chunking + Embedding + Hybrid Retrieval + QA
# ============================================================

import os
import json
import re
import math
import fitz
import streamlit as st
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# ============================================================
# ENV
# ============================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# ============================================================
# EMBEDDING
# ============================================================

def embed_text(text: str):
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )
    return response.json()["embedding"]

# ============================================================
# EXTRACTION (PyMuPDF FIXED)
# ============================================================

def extract_text_tables(pdf_path: str):

    doc = fitz.open(pdf_path)
    items = []
    seq = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if len(text) < 5:
                continue
            items.append({
                "page": page_idx + 1,
                "order": seq,
                "content": text
            })
            seq += 1

        tables = page.find_tables()
        for table in tables:
            md = table.to_markdown()
            if md.strip():
                items.append({
                    "page": page_idx + 1,
                    "order": seq,
                    "content": md
                })
                seq += 1

    doc.close()
    return items

# ============================================================
# CHUNKING
# ============================================================

MAX_CHARS = 1400

def adaptive_chunk(text: str):

    sections = text.split("\n\n")
    chunks = []
    buffer = ""
    cid = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(buffer) + len(section) < MAX_CHARS:
            buffer += section + "\n\n"
        else:
            chunks.append({
                "chunk_id": f"chunk_{cid}",
                "content": buffer.strip()
            })
            cid += 1
            buffer = section + "\n\n"

    if buffer.strip():
        chunks.append({
            "chunk_id": f"chunk_{cid}",
            "content": buffer.strip()
        })

    return chunks

# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

# ============================================================
# SPARSE RETRIEVAL (Simple BM25-style)
# ============================================================

def sparse_score(query, document):
    query_terms = query.lower().split()
    doc_terms = document.lower().split()

    score = 0
    for term in query_terms:
        score += doc_terms.count(term)

    return score

# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(query, chunks, top_k=5):

    query_embedding = embed_text(query)

    results = []

    for chunk in chunks:

        dense = cosine_similarity(query_embedding, chunk["embedding"])
        sparse = sparse_score(query, chunk["content"])

        # Normalize sparse
        sparse_norm = sparse / 10

        hybrid = 0.7 * dense + 0.3 * sparse_norm

        results.append({
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "dense_score": round(dense, 4),
            "sparse_score": sparse,
            "hybrid_score": round(hybrid, 4)
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return results[:top_k]

# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(query, retrieved_chunks):

    groq = Groq(api_key=GROQ_API_KEY)

    context = "\n\n".join([c["content"] for c in retrieved_chunks])

    prompt = f"""
You are an enterprise document assistant.

Use ONLY the provided context to answer.
If not found, say: Information not found in document.

Context:
{context}

Question:
{query}
"""

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config("Hybrid RAG System", layout="wide")
st.title("🚀 Enterprise PDF → Hybrid RAG")

if "chunks" not in st.session_state:
    st.session_state.chunks = None

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf and st.session_state.chunks is None:

    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    with st.spinner("Extracting + Chunking + Embedding..."):

        items = extract_text_tables("temp.pdf")

        full_text = "\n\n".join([i["content"] for i in items])
        chunks = adaptive_chunk(full_text)

        embedded_chunks = []
        for chunk in chunks:
            emb = embed_text(chunk["content"])
            embedded_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "embedding": emb
            })

        st.session_state.chunks = embedded_chunks

    os.remove("temp.pdf")
    st.success("Ingestion complete!")

# ============================================================
# QUERY SECTION
# ============================================================

if st.session_state.chunks:

    query = st.text_input("Ask a question about the document:")

    if query:

        with st.spinner("Retrieving..."):
            retrieved = hybrid_retrieve(query, st.session_state.chunks)

        st.subheader("🔎 Retrieved Chunks + Scores")

        for r in retrieved:
            with st.expander(f"{r['chunk_id']} | Hybrid: {r['hybrid_score']}"):
                st.write(f"**Dense Score:** {r['dense_score']}")
                st.write(f"**Sparse Score:** {r['sparse_score']}")
                st.write(r["content"][:1000])

        with st.spinner("Generating Answer..."):
            answer = generate_answer(query, retrieved)

        st.subheader("🧠 Final Answer")
        st.write(answer)
