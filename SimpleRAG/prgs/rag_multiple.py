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

# ===============================
# Setup
# ===============================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

OCR_READER = easyocr.Reader(["en"], gpu=False)

IMAGE_SYSTEM_PROMPT = """You are an enterprise compliance assistant.

Describe the UI screenshot using ONLY the OCR text provided.

Rules:
- Identify screen type
- List visible fields or options
- Explain policy purpose
- Do not guess
- If unclear, say so
- 3-5 sentences max
"""

# ===============================
# Embedding
# ===============================

def embed_text(text: str) -> List[float]:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        return response.json().get("embedding", [])
    except Exception:
        return []

# ===============================
# Image Processing
# ===============================

def preprocess_image(img_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if min(img.size) < 300:
            scale = 300 / min(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = img.convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return img_bytes

def ocr_image(img_bytes: bytes) -> str:
    img_bytes = preprocess_image(img_bytes)
    result = OCR_READER.readtext(img_bytes, detail=0, paragraph=True)
    return "\n".join(result).strip() if result else ""

def is_logo_image(ocr_text: str) -> bool:
    text = ocr_text.lower().strip()
    return len(text) < 20 and ("ibm" in text or "maas360" in text)

def extract_images(pdf_path: str, groq_client: Groq):
    doc = fitz.open(pdf_path)
    page_images = {}
    seen_logos = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_images[page_idx] = []

        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            ocr = ocr_image(base["image"])

            if len(ocr) < 3:
                continue

            if is_logo_image(ocr):
                h = hashlib.md5(ocr.encode()).hexdigest()
                if h in seen_logos:
                    continue
                seen_logos.add(h)

            try:
                res = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    temperature=0.1,
                    max_tokens=400,
                    messages=[
                        {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                        {"role": "user", "content": ocr}
                    ]
                )
                desc = res.choices[0].message.content.strip()
            except Exception:
                desc = f"Image contains: {ocr[:80]}"

            page_images[page_idx].append({
                "ocr": ocr,
                "description": desc,
                "img_no": i + 1
            })

    doc.close()
    return page_images

# ===============================
# Text + Table Extraction
# ===============================

def extract_text_tables(pdf_path: str):
    doc = fitz.open(pdf_path)
    items = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if len(text) < 5:
                continue
            items.append({
                "page": page_idx,
                "type": "text",
                "content": text,
                "y0": b[1]
            })

        tables = page.find_tables()
        for table in tables:
            md = table.to_markdown()
            if md.strip():
                items.append({
                    "page": page_idx,
                    "type": "table",
                    "content": md,
                    "y0": table.bbox[1] if hasattr(table, 'bbox') else 0
                })

    doc.close()
    return items

# ===============================
# Chunking
# ===============================

def normalize_heading(heading: str) -> str:
    heading = re.sub(r'\s+', ' ', heading).strip()
    heading = re.sub(r'[:\.]$', '', heading)
    return heading

def is_heading(text: str) -> bool:
    text = text.strip()
    return len(text) < 100 and text.endswith(":") and text[0].isupper()

def section_level_chunk(items, page_images):
    chunks = []
    current_heading = None
    current_content = []
    current_page = None
    section_id = 0

    items_sorted = sorted(items, key=lambda x: (x["page"], x["y0"]))

    for item in items_sorted:
        content = item["content"].strip()
        page = item["page"]

        if is_heading(content):
            if current_heading and current_content:
                section_text = "\n\n".join(current_content)
                chunks.append({
                    "chunk_id": f"section_{section_id}",
                    "heading": normalize_heading(current_heading),
                    "content": f"{normalize_heading(current_heading)}\n\n{section_text}",
                    "page": current_page + 1
                })
                section_id += 1

            current_heading = content
            current_content = []
            current_page = page
        else:
            if current_heading:
                current_content.append(content)

    if current_heading and current_content:
        section_text = "\n\n".join(current_content)
        chunks.append({
            "chunk_id": f"section_{section_id}",
            "heading": normalize_heading(current_heading),
            "content": f"{normalize_heading(current_heading)}\n\n{section_text}",
            "page": current_page + 1
        })

    return chunks

# ===============================
# Retrieval
# ===============================

def tokenize_text(text: str):
    return re.findall(r'\b\w+\b', text.lower())

def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def reciprocal_rank_fusion(dense_ranks, bm25_ranks, k=60):
    rrf_scores = {}
    for rank, (chunk_idx, _) in enumerate(dense_ranks):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1 / (k + rank + 1)
    for rank, (chunk_idx, _) in enumerate(bm25_ranks):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retrieve(query, chunks, top_k=5):
    query_embedding = embed_text(query)

    dense_scores = [(i, cosine_similarity(query_embedding, c["embedding"]))
                    for i, c in enumerate(chunks)]
    dense_scores.sort(key=lambda x: x[1], reverse=True)

    tokenized_corpus = [tokenize_text(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenize_text(query))
    bm25_ranked = sorted([(i, s) for i, s in enumerate(bm25_scores)],
                         key=lambda x: x[1], reverse=True)

    hybrid_scores = reciprocal_rank_fusion(dense_scores, bm25_ranked)

    return [chunks[i] for i, _ in hybrid_scores[:top_k]]

# ===============================
# Answer Generation
# ===============================

def generate_answer(query, retrieved_chunks):
    groq = Groq(api_key=GROQ_API_KEY)
    context = "\n\n".join(
        [f"[Source: {c.get('source_file','Unknown')} | Page {c.get('page')}]\n{c['content']}"
         for c in retrieved_chunks]
    )

    prompt = f"""You are an enterprise document assistant.

The context may contain excerpts from multiple documents.
If relevant, mention the source file.

Use ONLY the provided context.
If not found, say: "Information not found in document."

Context:
{context}

Question:
{query}

Answer:"""

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# ===============================
# Streamlit UI
# ===============================

st.title("Multi-Document RAG Pipeline")

if "chunks" not in st.session_state:
    st.session_state.chunks = None

pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if pdfs and st.session_state.chunks is None:
    groq_client = Groq(api_key=GROQ_API_KEY)
    all_chunks = []

    with st.spinner("Processing PDFs..."):
        for pdf in pdfs:
            temp_path = f"temp_{pdf.name}"
            with open(temp_path, "wb") as f:
                f.write(pdf.read())

            text_items = extract_text_tables(temp_path)
            page_images = extract_images(temp_path, groq_client)
            chunks = section_level_chunk(text_items, page_images)

            for chunk in chunks:
                chunk["source_file"] = pdf.name
                chunk["embedding"] = embed_text(chunk["content"])
                all_chunks.append(chunk)

            os.remove(temp_path)

    st.session_state.chunks = all_chunks
    st.success(f"Created {len(all_chunks)} chunks from {len(pdfs)} files.")

if st.session_state.chunks:
    st.divider()
    query = st.text_input("Ask a question:")

    if query:
        retrieved = hybrid_retrieve(query, st.session_state.chunks)
        answer = generate_answer(query, retrieved)

        st.subheader("Answer")
        st.success(answer)

        st.subheader("Retrieved Chunks")
        for r in retrieved:
            with st.expander(f"{r.get('source_file')} | Page {r.get('page')}"):
                st.write(r["content"])
