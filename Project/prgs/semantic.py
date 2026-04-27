import streamlit as st
import fitz  # PyMuPDF
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import tempfile
from pathlib import Path
import time
import re


# ============================================================
# CONFIG
# ============================================================

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# Adaptive chunking controls
TARGET_CHARS = 1200          # soft target size
MAX_CHARS = 2000             # hard cap
MIN_CHARS = 300              # minimum meaningful chunk
SIMILARITY_THRESHOLD = 0.80  # semantic continuation

MIN_TEXT_LENGTH = 30
MIN_EMBEDDING_SIZE = 10
EMBED_SLEEP = 0.05


# ============================================================
# PDF → PARAGRAPHS (WINDOWS SAFE)
# ============================================================

def extract_paragraphs(pdf_bytes: bytes) -> List[str]:
    paragraphs = []

    temp_dir = Path(tempfile.mkdtemp())
    pdf_path = temp_dir / "input.pdf"
    pdf_path.write_bytes(pdf_bytes)

    doc = None
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text("text")
            if not text:
                continue

            # Split into paragraphs
            parts = re.split(r"\n\s*\n", text)
            for p in parts:
                p = p.strip()
                if len(p) >= MIN_TEXT_LENGTH:
                    paragraphs.append(p)

    finally:
        if doc:
            doc.close()
        try:
            pdf_path.unlink()
            temp_dir.rmdir()
        except Exception:
            pass

    return paragraphs


# ============================================================
# OLLAMA EMBEDDINGS (SAFE)
# ============================================================

def embed(text: str) -> List[float] | None:
    payload = {
        "model": EMBED_MODEL,
        "input": text
    }

    r = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=60)
    if r.status_code != 200:
        return None

    emb = r.json().get("embedding", [])
    if not emb or len(emb) < MIN_EMBEDDING_SIZE:
        return None

    time.sleep(EMBED_SLEEP)
    return emb


def similarity(e1: List[float], e2: List[float]) -> float:
    return cosine_similarity([e1], [e2])[0][0]


# ============================================================
# ADAPTIVE CHUNKER
# ============================================================

def adaptive_chunk(paragraphs: List[str]) -> List[str]:
    chunks = []
    current_chunk = []
    current_chars = 0
    current_embedding = None

    for para in paragraphs:
        para_embedding = embed(para)
        if para_embedding is None:
            continue

        # First paragraph
        if not current_chunk:
            current_chunk = [para]
            current_chars = len(para)
            current_embedding = para_embedding
            continue

        sim = similarity(current_embedding, para_embedding)

        should_continue = (
            sim >= SIMILARITY_THRESHOLD
            and current_chars + len(para) <= MAX_CHARS
        )

        if should_continue:
            current_chunk.append(para)
            current_chars += len(para)
            # update embedding as mean
            current_embedding = np.mean(
                [current_embedding, para_embedding], axis=0
            ).tolist()
        else:
            # finalize chunk
            if current_chars >= MIN_CHARS:
                chunks.append("\n\n".join(current_chunk))
            else:
                # merge small chunk forward
                if chunks:
                    chunks[-1] += "\n\n" + "\n\n".join(current_chunk)
                else:
                    chunks.append("\n\n".join(current_chunk))

            # start new chunk
            current_chunk = [para]
            current_chars = len(para)
            current_embedding = para_embedding

    # finalize last chunk
    if current_chunk:
        if current_chars >= MIN_CHARS:
            chunks.append("\n\n".join(current_chunk))
        elif chunks:
            chunks[-1] += "\n\n" + "\n\n".join(current_chunk)
        else:
            chunks.append("\n\n".join(current_chunk))

    return chunks


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(layout="wide")
    st.title("Adaptive Chunking — Production-Safe")

    st.markdown(
        """
        **Adaptive Chunking (Semantic + Size Aware)**  
        - Paragraph-level units  
        - Semantic continuation  
        - Soft + hard size limits  
        - RAG-ready output  
        """
    )

    with st.sidebar:
        st.markdown(
            """
            **Requirements**
            ```
            ollama pull nomic-embed-text
            ollama serve
            ```
            """
        )

        st.markdown("### Chunk Controls")
        st.write(f"Target chars: {TARGET_CHARS}")
        st.write(f"Max chars: {MAX_CHARS}")
        st.write(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and st.button("Run Adaptive Chunking", type="primary"):
        with st.spinner("Extracting paragraphs..."):
            paragraphs = extract_paragraphs(uploaded.getvalue())

        if not paragraphs:
            st.warning("No extractable text found.")
            return

        st.success(f"Extracted {len(paragraphs)} paragraphs")

        with st.spinner("Running adaptive chunking..."):
            chunks = adaptive_chunk(paragraphs)

        st.success(f"Generated {len(chunks)} adaptive chunks")

        tab1, tab2 = st.tabs(["Chunks", "Raw Paragraphs"])

        with tab1:
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i + 1} ({len(chunk)} chars)"):
                    st.write(chunk)

        with tab2:
            for i, p in enumerate(paragraphs):
                with st.expander(f"Paragraph {i + 1}"):
                    st.write(p)

    elif not uploaded:
        st.info("Upload a PDF to begin")


if __name__ == "__main__":
    main()
