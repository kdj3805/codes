import os
import io
import re
import math
import hashlib
import fitz
import streamlit as st
import requests
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from groq import Groq
import easyocr
from rank_bm25 import BM25Okapi

# =========================
# CONFIG
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

OCR_READER = easyocr.Reader(["en"], gpu=False)

# =========================
# UTILITIES
# =========================

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def embed_text(text: str) -> List[float]:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        return resp.json()["embedding"]
    except Exception:
        return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def reciprocal_rank_fusion(ranked_lists: List[List[Tuple[int, float]]], k: int = 60):
    scores = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


# =========================
# IMAGE PROCESSING
# =========================

def preprocess_image(img_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = img.convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return img_bytes


def run_ocr(img_bytes: bytes) -> str:
    processed = preprocess_image(img_bytes)
    result = OCR_READER.readtext(processed, detail=0, paragraph=True)
    return "\n".join(result).strip() if result else ""


# =========================
# PDF EXTRACTION
# =========================

def extract_pdf(pdf_path: str, groq_client: Groq):
    doc = fitz.open(pdf_path)

    chunks = []
    chunk_id = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # TEXT BLOCKS
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if len(text) < 5:
                continue

            chunks.append({
                "chunk_id": f"text_{chunk_id}",
                "chunk_type": "section",
                "heading": "",
                "content": text,
                "page": page_idx + 1
            })
            chunk_id += 1

        # TABLES
        for table in page.find_tables():
            md = table.to_markdown()
            if md.strip():
                chunks.append({
                    "chunk_id": f"table_{chunk_id}",
                    "chunk_type": "table",
                    "heading": f"Table — Page {page_idx + 1}",
                    "content": md,
                    "page": page_idx + 1
                })
                chunk_id += 1

        # IMAGES
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                ocr = run_ocr(base["image"])
            except Exception:
                continue

            if len(ocr) < 3:
                continue

            desc = ocr

            chunks.append({
                "chunk_id": f"image_{chunk_id}",
                "chunk_type": "image",
                "heading": f"Image — Page {page_idx + 1}",
                "content": f"OCR:\n{ocr}\n\nDescription:\n{desc}",
                "page": page_idx + 1
            })
            chunk_id += 1

    doc.close()
    return chunks


# =========================
# INTENT CLASSIFIER
# =========================

def classify_intent(query: str) -> str:
    q = query.lower()

    if "table of contents" in q or "main sections" in q:
        return "toc"
    if any(x in q for x in ["difference", "compare", "comparison", "include"]):
        return "table"
    if any(x in q for x in ["screen", "ui", "panel", "visible", "settings"]):
        return "image"
    if "best practice" in q:
        return "section"

    return "general"


def intent_boost(chunk_type: str, intent: str) -> float:
    if intent == "table" and chunk_type == "table":
        return 5.0
    if intent == "image" and chunk_type == "image":
        return 5.0
    if intent == "section" and chunk_type == "section":
        return 3.0
    return 0.0


# =========================
# HYBRID RETRIEVAL
# =========================

def hybrid_retrieve(query: str, chunks: List[Dict[str, Any]], intent: str, top_k=5):

    query_embedding = embed_text(query)
    query_tokens = tokenize(query)

    dense_scores = []
    for idx, chunk in enumerate(chunks):
        sim = cosine_similarity(query_embedding, chunk["embedding"])
        dense_scores.append((idx, sim))

    dense_ranked = sorted(dense_scores, key=lambda x: x[1], reverse=True)

    tokenized_corpus = [tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    raw_bm25 = bm25.get_scores(query_tokens)

    bm25_ranked = sorted(
        [(i, float(s)) for i, s in enumerate(raw_bm25)],
        key=lambda x: x[1],
        reverse=True,
    )

    rrf_scores = reciprocal_rank_fusion([dense_ranked, bm25_ranked])

    final_scores = []

    for idx, rrf in rrf_scores.items():
        boost = intent_boost(chunks[idx]["chunk_type"], intent)
        final = rrf + boost
        final_scores.append((idx, final))

    final_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, final in final_scores[:top_k]:
        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "chunk_type": chunks[idx]["chunk_type"],
            "content": chunks[idx]["content"],
            "page": chunks[idx]["page"],
            "final_score": round(final, 4),
        })

    return results


# =========================
# ANSWER GENERATION
# =========================

def generate_answer(query: str, retrieved: List[Dict[str, Any]]):
    groq = Groq(api_key=GROQ_API_KEY)

    context = "\n\n---\n\n".join([c["content"] for c in retrieved])

    prompt = f"""
Use ONLY the context below to answer the question.
If not found, say: Information not found in document.

Context:
{context}

Question:
{query}

Answer:
"""

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Enterprise Hybrid RAG", layout="wide")
st.title("🏢 Enterprise Hybrid RAG")

if "chunks" not in st.session_state:
    st.session_state.chunks = None

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file and st.session_state.chunks is None:

    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    groq_client = Groq(api_key=GROQ_API_KEY)

    with st.spinner("Extracting PDF..."):
        raw_chunks = extract_pdf("temp.pdf", groq_client)

        progress = st.progress(0)
        embedded = []

        for i, chunk in enumerate(raw_chunks):
            emb = embed_text(chunk["content"])
            chunk["embedding"] = emb
            embedded.append(chunk)
            progress.progress((i + 1) / len(raw_chunks))

        progress.empty()

        st.session_state.chunks = embedded

    os.remove("temp.pdf")

    st.success(f"Ingested {len(embedded)} chunks successfully.")

if st.session_state.chunks:

    query = st.text_input("Ask a question")

    if query:
        intent = classify_intent(query)

        retrieved = hybrid_retrieve(
            query,
            st.session_state.chunks,
            intent=intent,
            top_k=5,
        )

        st.subheader("Retrieved Chunks")
        for r in retrieved:
            with st.expander(f"{r['chunk_type']} | Page {r['page']} | Score {r['final_score']}"):
                st.text(r["content"][:1000])

        answer = generate_answer(query, retrieved)

        st.subheader("Final Answer")
        st.success(answer)
