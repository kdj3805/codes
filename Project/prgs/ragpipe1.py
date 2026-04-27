# ============================================================
# ENTERPRISE PDF → DEBUGGABLE FULL RAG SYSTEM
# Extraction → Chunking → Embedding → Retrieval → Answer
# Shows Retrieved Chunks + Similarity Scores
# ============================================================

import os
import io
import json
import re
import fitz
import math
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from groq import Groq
from PIL import Image, ImageEnhance
import easyocr

# =========================
# ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

OCR_READER = easyocr.Reader(["en"], gpu=False)

# ============================================================
# -------------------- EMBEDDINGS ----------------------------
# ============================================================

def embed_text(text: str) -> List[float]:
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )
    return response.json()["embedding"]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# ============================================================
# -------------------- EXTRACTION ----------------------------
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
                "page": page_idx,
                "order": seq,
                "type": "text",
                "content": text
            })
            seq += 1

        tables = page.find_tables()
        for table in tables:
            md = table.to_markdown()
            if md.strip():
                items.append({
                    "page": page_idx,
                    "order": seq,
                    "type": "table",
                    "content": md
                })
                seq += 1

    doc.close()
    return items

# ============================================================
# -------------------- CHUNKING ------------------------------
# ============================================================

MAX_CHARS = 1400

def adaptive_chunk(md_text: str):
    sections = md_text.split("\n\n")
    chunks = []
    buffer = ""
    chunk_id = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(buffer) + len(section) < MAX_CHARS:
            buffer += section + "\n\n"
        else:
            chunks.append({
                "chunk_id": f"text_{chunk_id}",
                "chunk_type": "text",
                "content": buffer.strip()
            })
            chunk_id += 1
            buffer = section + "\n\n"

    if buffer.strip():
        chunks.append({
            "chunk_id": f"text_{chunk_id}",
            "chunk_type": "text",
            "content": buffer.strip()
        })

    return chunks

# ============================================================
# -------------------- RAG RETRIEVAL -------------------------
# ============================================================

def retrieve(query: str, chunks: List[Dict[str, Any]], top_k=5):
    query_embedding = embed_text(query)

    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[:top_k]

# ============================================================
# -------------------- ANSWERING -----------------------------
# ============================================================

def answer_query(query: str, retrieved_chunks):
    groq = Groq(api_key=GROQ_API_KEY)

    context = "\n\n".join([c["content"] for _, c in retrieved_chunks])

    prompt = f"""
You are an enterprise policy assistant.

Use ONLY the provided context to answer.
If answer not found, say:
"Information not found in document."

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
# -------------------- STREAMLIT UI --------------------------
# ============================================================

st.set_page_config("Enterprise RAG Debug System", layout="wide")
st.title("📄 Enterprise PDF → Debuggable RAG")

if "chunks" not in st.session_state:
    st.session_state.chunks = None

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    with st.spinner("Extracting document..."):
        items = extract_text_tables("temp.pdf")
        md_text = "\n\n".join([i["content"] for i in items])
        chunks = adaptive_chunk(md_text)

    st.subheader("🔍 Extracted Content Preview")
    st.text_area("", md_text[:4000], height=300)

    st.subheader("📦 Generated Chunks")
    st.json(chunks[:3])

    with st.spinner("Generating embeddings..."):
        for chunk in chunks:
            chunk["embedding"] = embed_text(chunk["content"])

    st.success("Embeddings generated.")

    st.session_state.chunks = chunks
    os.remove("temp.pdf")

# ============================================================
# -------------------- QUERY SECTION -------------------------
# ============================================================

if st.session_state.chunks:
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve(query, st.session_state.chunks, top_k=5)

        st.subheader("📊 Retrieved Chunks + Similarity Scores")

        for score, chunk in retrieved:
            st.markdown(f"**Score: {round(score,4)}**")
            st.text(chunk["content"][:1000])
            st.divider()

        with st.spinner("Generating final answer..."):
            answer = answer_query(query, retrieved)

        st.subheader("🧠 Final Answer")
        st.write(answer)
