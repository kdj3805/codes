# ============================================================
# ENTERPRISE PDF → FULL RAG SYSTEM
# TEXT + TABLES (PyMuPDF)
# IMAGES (OCR + Groq)
# ADAPTIVE CHUNKING
# OLLAMA EMBEDDINGS
# CHROMADB STORAGE
# RETRIEVAL + QA
# ============================================================

import os
import io
import json
import re
import hashlib
import fitz
import requests
import chromadb
import streamlit as st
from chromadb.config import Settings
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from typing import List, Dict, Any
from groq import Groq
import easyocr

# =========================
# ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "openai/gpt-oss-120b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "enterprise_docs"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

OCR_READER = easyocr.Reader(["en"], gpu=False)

# ============================================================
# IMAGE DESCRIPTION PROMPT
# ============================================================
IMAGE_SYSTEM_PROMPT = """You are an enterprise compliance assistant.

Describe the UI screenshot using ONLY the OCR text provided.

Rules:
- Identify screen type
- List visible fields or options
- Explain policy purpose
- Do not guess
- If unclear, say so
- 3–5 sentences max
"""

# ============================================================
# IMAGE OCR
# ============================================================

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

# ============================================================
# IMAGE EXTRACTION
# ============================================================

def is_logo_image(ocr_text: str) -> bool:
    text = ocr_text.lower().strip()
    return len(text) < 20 and ("ibm" in text or "maas360" in text)


def extract_images(pdf_path: str):
    doc = fitz.open(pdf_path)
    images = []
    seen_logos = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
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

            images.append({
                "page": page_idx,
                "order": i,
                "type": "image",
                "ocr": ocr,
                "img_no": i + 1
            })

    doc.close()
    return images

# ============================================================
# TEXT + TABLE EXTRACTION
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
# GROQ IMAGE DESCRIPTION
# ============================================================

def describe_image(ocr_text: str, client: Groq) -> str:
    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=400,
            messages=[
                {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                {"role": "user", "content": ocr_text}
            ]
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return f"Image contains: {ocr_text[:80]}"

# ============================================================
# ADAPTIVE CHUNKING
# ============================================================

MAX_CHARS = 1400

def adaptive_chunk(md_text: str):
    chunks = []
    buffer = ""
    chunk_id = 0

    sections = md_text.split("\n\n")

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if section.startswith("<imagedesc>"):
            if buffer:
                chunks.append({
                    "chunk_id": f"text_{chunk_id}",
                    "chunk_type": "text",
                    "content": buffer.strip()
                })
                chunk_id += 1
                buffer = ""

            chunks.append({
                "chunk_id": f"image_{chunk_id}",
                "chunk_type": "image",
                "content": section
            })
            chunk_id += 1
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
# OLLAMA EMBEDDINGS
# ============================================================

def get_embedding(text: str):
    res = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
    )
    return res.json()["embedding"]

def embed_chunks(chunks):
    for chunk in chunks:
        chunk["embedding"] = get_embedding(chunk["content"])
    return chunks

# ============================================================
# CHROMADB
# ============================================================

def store_in_chroma(chunks):
    client = chromadb.Client(
        Settings(persist_directory=CHROMA_DB_DIR)
    )

    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = []
    docs = []
    embeds = []
    metas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        docs.append(chunk["content"])
        embeds.append(chunk["embedding"])
        metas.append({"type": chunk["chunk_type"]})

    collection.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)


def retrieve(query: str, top_k=5):
    client = chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    chunks_with_scores = []
    for i, doc in enumerate(results["documents"][0]):
        chunks_with_scores.append({
            "content": doc,
            "distance": results["distances"][0][i],
            "id": results["ids"][0][i]
        })
    
    return chunks_with_scores

# ============================================================
# RAG ANSWER
# ============================================================

def answer_query(query: str):
    groq = Groq(api_key=GROQ_API_KEY)

    chunks_with_scores = retrieve(query)
    context = "\n\n".join([chunk["content"] for chunk in chunks_with_scores])

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

    res = groq.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = res.choices[0].message.content.strip()
    
    return answer, chunks_with_scores

# ============================================================
# MAIN PROCESSING
# ============================================================

def process_pdf(pdf_path: str):
    groq = Groq(api_key=GROQ_API_KEY)

    text_items = extract_text_tables(pdf_path)
    image_items = extract_images(pdf_path)

    for img in image_items:
        img["description"] = describe_image(img["ocr"], groq)

    pages = {}
    for item in text_items + image_items:
        pages.setdefault(item["page"], []).append(item)

    md_blocks = []

    for page in sorted(pages.keys()):
        page_items = pages[page]

        for t in sorted([i for i in page_items if i["type"] == "text"], key=lambda x: x["order"]):
            md_blocks.append(t["content"])

        for t in sorted([i for i in page_items if i["type"] == "table"], key=lambda x: x["order"]):
            md_blocks.append(t["content"])

        for img in sorted([i for i in page_items if i["type"] == "image"], key=lambda x: x["order"]):
            md_blocks.append(f"""<imagedesc>
Page {page+1}, Image {img["img_no"]}

OCR Text:
{img["ocr"]}

Description:
{img["description"]}
</imagedesc>""")

    md_text = "\n\n".join(md_blocks)
    chunks = adaptive_chunk(md_text)
    chunks = embed_chunks(chunks)
    store_in_chroma(chunks)

    return chunks

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config("Enterprise RAG System", layout="wide")
st.title("📄 Enterprise PDF → Full RAG System")

if "processed" not in st.session_state:
    st.session_state.processed = False

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf and not st.session_state.processed:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    with st.spinner("Processing and storing in ChromaDB..."):
        chunks = process_pdf("temp.pdf")

    st.success("Document stored in ChromaDB!")
    st.session_state.processed = True

    os.remove("temp.pdf")

if st.session_state.processed:
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Generating answer..."):
            answer, chunks = answer_query(query)

        st.subheader("Answer")
        st.write(answer)
        
        st.subheader("Retrieved Chunks")
        st.caption(f"Top {len(chunks)} most relevant chunks")
        
        for idx, chunk in enumerate(chunks):
            distance = chunk["distance"]
            similarity = 1 / (1 + distance)
            
            with st.expander(f"Chunk {idx + 1}: {chunk['id']} (Similarity: {similarity:.2%})"):
                st.markdown(f"**Distance Score:** {distance:.4f} (lower = more relevant)")
                st.markdown(f"**Similarity Score:** {similarity:.2%}")
                st.markdown("---")
                st.text(chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"])


