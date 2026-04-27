import os
import io
import json
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

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
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


def embed_text(text: str) -> List[float]:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        return response.json()["embedding"]
    except Exception:
        return []


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


def extract_images(pdf_path: str, groq_client: Groq) -> Dict[int, List[Dict[str, Any]]]:
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
                if len(desc) < 10:
                    desc = f"Image contains: {ocr[:80]}"
            except Exception:
                desc = f"Image contains: {ocr[:80]}"

            page_images[page_idx].append({
                "ocr": ocr,
                "description": desc,
                "img_no": i + 1
            })

    doc.close()
    return page_images


def extract_text_tables(pdf_path: str) -> List[Dict[str, Any]]:
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


def is_heading(text: str) -> bool:
    heading_patterns = [
        r"^Best Practice #\d+:",
        r"^Introduction\s*$",
        r"^Our Recommendations\s*$",
        r"^How MaaS360 Helps\s*$",
        r"^The Options\s*$",
        r"^Chapter \d+[:\.]",
        r"^About this publication\s*$",
        r"^Table of Contents\s*$"
    ]
    
    for pattern in heading_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    
    if len(text) < 80 and text.strip().endswith(":") and text.strip()[0].isupper():
        return True
    
    return False


def section_level_chunk(items: List[Dict[str, Any]], page_images: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    chunks = []
    current_section = None
    current_content = []
    section_id = 0

    items_sorted = sorted(items, key=lambda x: (x["page"], x["y0"]))

    for item in items_sorted:
        content = item["content"]
        
        if is_heading(content):
            if current_section and current_content:
                page = items_sorted[0]["page"] if current_content else 0
                for idx, c in enumerate(current_content):
                    if isinstance(c, dict) and "page" in c:
                        page = c["page"]
                        break
                
                images_in_section = page_images.get(page, [])
                image_blocks = []
                for img in images_in_section:
                    image_blocks.append(f"""<imagedesc>
Page {page + 1}, Image {img["img_no"]}

OCR Text:
{img["ocr"]}

Description:
{img["description"]}
</imagedesc>""")
                
                section_text = "\n\n".join([
                    c if isinstance(c, str) else c.get("content", "")
                    for c in current_content
                ])
                
                if image_blocks:
                    section_text += "\n\n" + "\n\n".join(image_blocks)
                
                chunks.append({
                    "chunk_id": f"section_{section_id}",
                    "chunk_type": "section",
                    "heading": current_section,
                    "content": f"{current_section}\n\n{section_text}",
                    "page": page + 1
                })
                section_id += 1
            
            current_section = content
            current_content = []
        else:
            current_content.append(item)

    if current_section and current_content:
        page = current_content[0]["page"] if current_content else 0
        
        images_in_section = page_images.get(page, [])
        image_blocks = []
        for img in images_in_section:
            image_blocks.append(f"""<imagedesc>
Page {page + 1}, Image {img["img_no"]}

OCR Text:
{img["ocr"]}

Description:
{img["description"]}
</imagedesc>""")
        
        section_text = "\n\n".join([
            c if isinstance(c, str) else c.get("content", "")
            for c in current_content
        ])
        
        if image_blocks:
            section_text += "\n\n" + "\n\n".join(image_blocks)
        
        chunks.append({
            "chunk_id": f"section_{section_id}",
            "chunk_type": "section",
            "heading": current_section,
            "content": f"{current_section}\n\n{section_text}",
            "page": page + 1
        })

    return chunks


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def reciprocal_rank_fusion(dense_ranks: List[Tuple[int, float]], bm25_ranks: List[Tuple[int, float]], k: int = 60) -> List[Tuple[int, float]]:
    rrf_scores = {}
    
    for rank, (chunk_idx, _) in enumerate(dense_ranks):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1 / (k + rank + 1)
    
    for rank, (chunk_idx, _) in enumerate(bm25_ranks):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1 / (k + rank + 1)
    
    sorted_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


def hybrid_retrieve(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = embed_text(query)
    
    dense_scores = []
    for idx, chunk in enumerate(chunks):
        sim = cosine_similarity(query_embedding, chunk["embedding"])
        dense_scores.append((idx, sim))
    
    dense_scores.sort(key=lambda x: x[1], reverse=True)
    
    tokenized_corpus = [chunk["content"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    bm25_ranked = [(idx, score) for idx, score in enumerate(bm25_scores)]
    bm25_ranked.sort(key=lambda x: x[1], reverse=True)
    
    hybrid_scores = reciprocal_rank_fusion(dense_scores, bm25_ranked, k=60)
    
    results = []
    for chunk_idx, hybrid_score in hybrid_scores[:top_k]:
        dense_score = next((s for i, s in dense_scores if i == chunk_idx), 0.0)
        bm25_score = bm25_scores[chunk_idx]
        
        results.append({
            "chunk_id": chunks[chunk_idx]["chunk_id"],
            "heading": chunks[chunk_idx].get("heading", ""),
            "content": chunks[chunk_idx]["content"],
            "page": chunks[chunk_idx].get("page"),
            "dense_score": round(dense_score, 4),
            "bm25_score": round(bm25_score, 4),
            "hybrid_score": round(hybrid_score, 4)
        })
    
    return results


def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    groq = Groq(api_key=GROQ_API_KEY)
    
    context = "\n\n".join([c["content"] for c in retrieved_chunks])
    
    prompt = f"""You are an enterprise document assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say: "Information not found in document."

Context:
{context}

Question:
{query}

Answer:"""

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


def stage_1_extraction(pdf_path: str, groq_client: Groq) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    st.info("Stage 1: Extracting text, tables, and images...")
    text_items = extract_text_tables(pdf_path)
    page_images = extract_images(pdf_path, groq_client)
    return text_items, page_images


def stage_2_chunking(text_items: List[Dict[str, Any]], page_images: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    st.info("Stage 2: Section-level chunking...")
    chunks = section_level_chunk(text_items, page_images)
    return chunks


def stage_3_embedding(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    st.info("Stage 3: Generating embeddings...")
    embedded_chunks = []
    progress_bar = st.progress(0)
    
    for idx, chunk in enumerate(chunks):
        embedding = embed_text(chunk["content"])
        embedded_chunk = dict(chunk)
        embedded_chunk["embedding"] = embedding
        embedded_chunks.append(embedded_chunk)
        progress_bar.progress((idx + 1) / len(chunks))
    
    progress_bar.empty()
    return embedded_chunks


st.set_page_config("Hybrid RAG System", layout="wide")
st.title("🚀 Enterprise PDF → Hybrid RAG System")

st.markdown("""
**Features:**
- Section-level chunking (heading-based, preserves context)
- Hybrid retrieval: Dense (Ollama embeddings) + BM25 + RRF fusion
- Image descriptions embedded in sections
- Full debug visibility with all scores
""")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf and st.session_state.chunks is None:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    with st.spinner("Processing PDF..."):
        text_items, page_images = stage_1_extraction("temp.pdf", groq_client)
        
        chunks = stage_2_chunking(text_items, page_images)
        
        embedded_chunks = stage_3_embedding(chunks)
        
        st.session_state.chunks = embedded_chunks
    
    os.remove("temp.pdf")
    
    st.success(f"✅ Ingestion complete! Created {len(embedded_chunks)} section-level chunks.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Chunks", len(embedded_chunks))
    with col2:
        st.metric("Text Items", len(text_items))
    
    with st.expander("Preview Chunks"):
        for chunk in embedded_chunks[:3]:
            st.markdown(f"**{chunk['chunk_id']}**: {chunk.get('heading', 'N/A')}")
            st.text_area("", chunk["content"][:500], height=150, key=chunk['chunk_id'])

if st.session_state.chunks:
    st.divider()
    st.subheader("🔍 Ask Questions")
    
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = hybrid_retrieve(query, st.session_state.chunks, top_k=5)
        
        st.subheader("📊 Retrieved Chunks + Scores")
        
        for r in retrieved:
            with st.expander(f"**{r['chunk_id']}** | Hybrid: {r['hybrid_score']} | Page: {r.get('page', 'N/A')}"):
                st.markdown(f"**Heading**: {r.get('heading', 'N/A')}")
                st.write(f"**Dense Score (Cosine)**: {r['dense_score']}")
                st.write(f"**BM25 Score**: {r['bm25_score']}")
                st.write(f"**Hybrid Score (RRF)**: {r['hybrid_score']}")
                st.divider()
                st.text_area("Content", r["content"], height=200, key=f"content_{r['chunk_id']}")
        
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, retrieved)
        
        st.subheader("🧠 Final Answer")
        st.success(answer)