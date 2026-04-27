import os
import io
import re
import math
import json
import hashlib
import fitz
import streamlit as st
import requests
from typing import List, Dict, Any, Tuple, Optional
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
- Identify screen type (configuration panel, policy screen, dashboard, etc.)
- List every visible field, label, setting, option, or button
- Explain the purpose of each setting in enterprise policy context
- Do not guess or hallucinate
- If unclear, explicitly say so
- 4-6 sentences max"""

SECTION_HEADING_PATTERNS = [
    r'^Best\s+Practice\s*#?\d+',
    r'^Introduction\s*:?\s*$',
    r'^Chapter\s+\d+',
    r'^About\s+this\s+publication\s*$',
    r'^Conclusion\s*:?\s*$',
    r'^Summary\s*:?\s*$',
    r'^Overview\s*:?\s*$',
]

SUBSECTION_HEADING_PATTERNS = [
    r'^Our\s+Recommendations?\s*:?\s*$',
    r'^How\s+MaaS360\s+Helps?\s*:?\s*$',
    r'^The\s+Options?\s*:?\s*$',
    r'^Types?\s+of\s+Passcodes?\s*:?\s*$',
    r'^Minimum\s+Length\s*:?\s*$',
    r'^Passcode\s+Expiration\s*:?\s*$',
    r'^Passcode\s+Reuse\s*:?\s*$',
]

INTENT_TOC = "toc"
INTENT_TABLE = "table"
INTENT_IMAGE = "image"
INTENT_SECTION = "section"
INTENT_SUBSECTION = "subsection"
INTENT_GENERAL = "general"

TOC_QUERY_KEYWORDS = [
    "main contents", "table of contents", "list sections",
    "all sections", "list headings", "what topics", "what sections",
    "contents page", "document structure"
]
TABLE_QUERY_KEYWORDS = [
    "difference", "compare", "comparison", "types of", "include",
    "table", "columns", "rows", "list of", "options available"
]
IMAGE_QUERY_KEYWORDS = [
    "screen", "screenshot", "configure", "configuration screen",
    "settings screen", "panel", "interface", "dialog", "ui",
    "what does the", "wizard", "button", "field", "checkbox",
    "dropdown", "visible", "shown", "displayed", "passcode policy screen",
    "policy screen"
]
SECTION_QUERY_KEYWORDS = [
    "best practice #", "best practice number",
    "what is best practice", "explain best practice"
]
SUBSECTION_QUERY_KEYWORDS = [
    "recommendations", "recommend", "how does maas360",
    "how maas360 helps", "what does maas360"
]


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


def embed_text(text: str) -> List[float]:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
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


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60
) -> Dict[int, float]:
    rrf: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return rrf


def preprocess_image_bytes(img_bytes: bytes) -> bytes:
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


def run_ocr(img_bytes: bytes) -> str:
    processed = preprocess_image_bytes(img_bytes)
    result = OCR_READER.readtext(processed, detail=0, paragraph=True)
    return "\n".join(result).strip() if result else ""


def is_logo_image(ocr_text: str) -> bool:
    text = ocr_text.lower().strip()
    return len(text) < 20 and ("ibm" in text or "maas360" in text)


def is_toc_page(page_text: str) -> bool:
    text_lower = page_text.lower()
    if "table of contents" in text_lower:
        return True
    lines = page_text.splitlines()
    dotted_lines = sum(1 for l in lines if re.search(r'\.{4,}', l))
    if dotted_lines >= 4:
        return True
    return False


def is_section_heading(text: str) -> bool:
    text = text.strip()
    for pattern in SECTION_HEADING_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def is_subsection_heading(text: str) -> bool:
    text = text.strip()
    for pattern in SUBSECTION_HEADING_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def normalize_heading(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[:\.]$', '', text)
    return text


def classify_query_intent(query: str) -> str:
    q_lower = query.lower()

    for kw in TOC_QUERY_KEYWORDS:
        if kw in q_lower:
            return INTENT_TOC

    for kw in IMAGE_QUERY_KEYWORDS:
        if kw in q_lower:
            return INTENT_IMAGE

    for kw in TABLE_QUERY_KEYWORDS:
        if kw in q_lower:
            return INTENT_TABLE

    for kw in SECTION_QUERY_KEYWORDS:
        if kw in q_lower:
            return INTENT_SECTION

    for kw in SUBSECTION_QUERY_KEYWORDS:
        if kw in q_lower:
            return INTENT_SUBSECTION

    return INTENT_GENERAL


def extract_raw_content(pdf_path: str) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    doc = fitz.open(pdf_path)
    items: List[Dict[str, Any]] = []
    page_full_text: Dict[int, str] = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        full_text = page.get_text("text")
        page_full_text[page_idx] = full_text

        for b in page.get_text("blocks"):
            text = b[4].strip()
            if len(text) < 5:
                continue
            items.append({
                "page": page_idx,
                "type": "text",
                "content": text,
                "y0": float(b[1])
            })

        for tbl in page.find_tables():
            md = tbl.to_markdown()
            if md.strip():
                y0 = float(tbl.bbox[1]) if hasattr(tbl, 'bbox') else 0.0
                items.append({
                    "page": page_idx,
                    "type": "table",
                    "content": md,
                    "y0": y0
                })

    doc.close()
    items.sort(key=lambda x: (x["page"], x["y0"]))
    return items, page_full_text


def extract_image_chunks(pdf_path: str, groq_client: Groq) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    image_chunks: List[Dict[str, Any]] = []
    seen_logo_hashes: set = set()
    img_chunk_id = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                ocr_text = run_ocr(base["image"])
            except Exception:
                continue

            if len(ocr_text) < 3:
                continue

            if is_logo_image(ocr_text):
                h = hashlib.md5(ocr_text.encode()).hexdigest()
                if h in seen_logo_hashes:
                    continue
                seen_logo_hashes.add(h)

            try:
                res = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    temperature=0.1,
                    max_tokens=400,
                    messages=[
                        {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                        {"role": "user", "content": ocr_text}
                    ]
                )
                desc = res.choices[0].message.content.strip()
                if len(desc) < 10:
                    desc = f"Image contains: {ocr_text[:120]}"
            except Exception:
                desc = f"Image contains: {ocr_text[:120]}"

            content = (
                f"<imagedesc>\n"
                f"Page {page_idx + 1}, Image {img_idx + 1}\n\n"
                f"OCR Text:\n{ocr_text}\n\n"
                f"Description:\n{desc}\n"
                f"</imagedesc>"
            )

            image_chunks.append({
                "chunk_id": f"image_{img_chunk_id}",
                "chunk_type": "image",
                "heading": f"Image — Page {page_idx + 1}",
                "content": content,
                "page": page_idx + 1
            })
            img_chunk_id += 1

    doc.close()
    return image_chunks


def build_toc_chunk(page_full_text: Dict[int, str]) -> Optional[Dict[str, Any]]:
    toc_lines: List[str] = []
    toc_page: Optional[int] = None

    for page_idx, full_text in page_full_text.items():
        if is_toc_page(full_text):
            toc_page = page_idx
            for line in full_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "table of contents" in line.lower():
                    continue
                clean_line = re.sub(r'\.{4,}', '', line)
                clean_line = re.sub(r'\s{2,}', ' ', clean_line).strip()
                if clean_line:
                    page_match = re.search(r'(\d+)\s*$', clean_line)
                    entry_text = re.sub(r'\d+\s*$', '', clean_line).strip()
                    if entry_text:
                        if page_match:
                            toc_lines.append(f"- {entry_text} (page {page_match.group(1)})")
                        else:
                            toc_lines.append(f"- {entry_text}")
            break

    if not toc_lines:
        return None

    content = "## Table of Contents\n\n" + "\n".join(toc_lines)

    return {
        "chunk_id": "toc_0",
        "chunk_type": "toc",
        "heading": "Table of Contents",
        "content": content,
        "page": (toc_page or 0) + 1
    }


def build_section_and_subsection_chunks(items: List[Dict[str, Any]], page_full_text: Dict[int, str]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    toc_pages: set = set()
    for page_idx, full_text in page_full_text.items():
        if is_toc_page(full_text):
            toc_pages.add(page_idx)

    section_heading: Optional[str] = None
    section_page: Optional[int] = None
    section_blocks: List[str] = []

    subsection_heading: Optional[str] = None
    subsection_page: Optional[int] = None
    subsection_blocks: List[str] = []

    section_id = 0
    subsection_id = 0

    def flush_subsection() -> None:
        nonlocal subsection_id
        if not subsection_heading or not subsection_blocks:
            return
        body = "\n\n".join(subsection_blocks)
        parent = normalize_heading(section_heading) if section_heading else ""
        content = f"[Section: {parent}]\n\n{normalize_heading(subsection_heading)}\n\n{body}"
        chunks.append({
            "chunk_id": f"subsection_{subsection_id}",
            "chunk_type": "subsection",
            "heading": normalize_heading(subsection_heading),
            "parent_heading": parent,
            "content": content,
            "page": (subsection_page or 0) + 1
        })
        subsection_id += 1

    def flush_section() -> None:
        nonlocal section_id
        if not section_heading or not section_blocks:
            return
        body = "\n\n".join(section_blocks)
        content = f"{normalize_heading(section_heading)}\n\n{body}"
        chunks.append({
            "chunk_id": f"section_{section_id}",
            "chunk_type": "section",
            "heading": normalize_heading(section_heading),
            "parent_heading": "",
            "content": content,
            "page": (section_page or 0) + 1
        })
        section_id += 1

    for item in items:
        if item["page"] in toc_pages:
            continue
        if item["type"] == "table":
            continue

        text = item["content"].strip()
        page = item["page"]

        if is_section_heading(text):
            flush_subsection()
            flush_section()
            section_heading = text
            section_page = page
            section_blocks = []
            subsection_heading = None
            subsection_blocks = []
            subsection_page = None
            continue

        if is_subsection_heading(text):
            flush_subsection()
            subsection_heading = text
            subsection_page = page
            subsection_blocks = []
            continue

        if subsection_heading is not None:
            subsection_blocks.append(text)
        elif section_heading is not None:
            section_blocks.append(text)

    flush_subsection()
    flush_section()

    return chunks


def build_table_chunks(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    table_chunks: List[Dict[str, Any]] = []
    tbl_id = 0

    for item in items:
        if item["type"] != "table":
            continue
        md = item["content"].strip()
        if not md:
            continue
        table_chunks.append({
            "chunk_id": f"table_{tbl_id}",
            "chunk_type": "table",
            "heading": f"Table — Page {item['page'] + 1}",
            "parent_heading": "",
            "content": md,
            "page": item["page"] + 1
        })
        tbl_id += 1

    return table_chunks


def build_all_chunks(pdf_path: str, groq_client: Groq) -> List[Dict[str, Any]]:
    raw_items, page_full_text = extract_raw_content(pdf_path)

    all_chunks: List[Dict[str, Any]] = []

    toc_chunk = build_toc_chunk(page_full_text)
    if toc_chunk:
        all_chunks.append(toc_chunk)

    section_chunks = build_section_and_subsection_chunks(raw_items, page_full_text)
    all_chunks.extend(section_chunks)

    table_chunks = build_table_chunks(raw_items)
    all_chunks.extend(table_chunks)

    image_chunks = extract_image_chunks(pdf_path, groq_client)
    all_chunks.extend(image_chunks)

    return all_chunks


def embed_all_chunks(chunks: List[Dict[str, Any]], progress_callback=None) -> List[Dict[str, Any]]:
    embedded: List[Dict[str, Any]] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        emb = embed_text(chunk["content"])
        ec = dict(chunk)
        ec["embedding"] = emb
        embedded.append(ec)
        if progress_callback:
            progress_callback((idx + 1) / total)
    return embedded


def build_index_for_type(chunks: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[int]]:
    tokenized = [tokenize_for_bm25(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25


def type_filtered_retrieve(
    query: str,
    all_embedded_chunks: List[Dict[str, Any]],
    allowed_types: List[str],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    candidates = [c for c in all_embedded_chunks if c.get("chunk_type") in allowed_types]

    if not candidates:
        return []

    query_embedding = embed_text(query)
    query_tokens = tokenize_for_bm25(query)

    dense_scores: List[Tuple[int, float]] = []
    for local_idx, chunk in enumerate(candidates):
        sim = cosine_similarity(query_embedding, chunk["embedding"])
        dense_scores.append((local_idx, sim))
    dense_ranked = sorted(dense_scores, key=lambda x: x[1], reverse=True)

    tokenized_corpus = [tokenize_for_bm25(c["content"]) for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    raw_bm25 = bm25.get_scores(query_tokens)
    bm25_ranked = sorted(
        [(i, float(s)) for i, s in enumerate(raw_bm25)],
        key=lambda x: x[1], reverse=True
    )

    rrf_scores = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60)
    final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for local_idx, rrf_score in final_ranked[:top_k]:
        dense_score = next((s for i, s in dense_ranked if i == local_idx), 0.0)
        bm25_score = float(raw_bm25[local_idx])

        results.append({
            "chunk_id": candidates[local_idx]["chunk_id"],
            "chunk_type": candidates[local_idx]["chunk_type"],
            "heading": candidates[local_idx].get("heading", ""),
            "parent_heading": candidates[local_idx].get("parent_heading", ""),
            "content": candidates[local_idx]["content"],
            "page": candidates[local_idx].get("page"),
            "dense_score": round(dense_score, 4),
            "bm25_score": round(bm25_score, 4),
            "rrf_score": round(rrf_score, 6),
        })

    return results


def route_and_retrieve(
    query: str,
    all_embedded_chunks: List[Dict[str, Any]],
    top_k: int = 5
) -> Tuple[str, List[Dict[str, Any]]]:
    intent = classify_query_intent(query)

    if intent == INTENT_TOC:
        toc_chunks = [c for c in all_embedded_chunks if c.get("chunk_type") == "toc"]
        if toc_chunks:
            return intent, [{
                "chunk_id": toc_chunks[0]["chunk_id"],
                "chunk_type": "toc",
                "heading": toc_chunks[0]["heading"],
                "parent_heading": "",
                "content": toc_chunks[0]["content"],
                "page": toc_chunks[0].get("page"),
                "dense_score": 1.0,
                "bm25_score": 1.0,
                "rrf_score": 1.0,
            }]
        else:
            sections = [c for c in all_embedded_chunks if c.get("chunk_type") == "section"]
            headings = "\n".join([f"- {c['heading']}" for c in sections])
            return intent, [{
                "chunk_id": "synthesized_toc",
                "chunk_type": "toc",
                "heading": "Document Sections",
                "parent_heading": "",
                "content": f"## Document Sections\n\n{headings}",
                "page": None,
                "dense_score": 1.0,
                "bm25_score": 1.0,
                "rrf_score": 1.0,
            }]

    elif intent == INTENT_IMAGE:
        retrieved = type_filtered_retrieve(query, all_embedded_chunks, ["image"], top_k=top_k)
        return intent, retrieved

    elif intent == INTENT_TABLE:
        retrieved = type_filtered_retrieve(query, all_embedded_chunks, ["table"], top_k=top_k)
        return intent, retrieved

    elif intent == INTENT_SECTION:
        retrieved = type_filtered_retrieve(query, all_embedded_chunks, ["section"], top_k=top_k)
        return intent, retrieved

    elif intent == INTENT_SUBSECTION:
        retrieved = type_filtered_retrieve(query, all_embedded_chunks, ["subsection"], top_k=top_k)
        return intent, retrieved

    else:
        retrieved = type_filtered_retrieve(query, all_embedded_chunks, ["section", "subsection"], top_k=top_k)
        return intent, retrieved


def generate_answer(
    query: str,
    intent: str,
    retrieved_chunks: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]]
) -> str:
    groq = Groq(api_key=GROQ_API_KEY)

    if intent == INTENT_TOC and retrieved_chunks:
        return retrieved_chunks[0]["content"]

    context_parts = []
    for c in retrieved_chunks:
        header = f"[{c['chunk_type'].upper()} | {c['heading']} | Page {c.get('page', '?')}]"
        context_parts.append(f"{header}\n{c['content']}")

    context = "\n\n---\n\n".join(context_parts)

    intent_instruction = {
        INTENT_IMAGE: "Focus on the image descriptions, OCR text, visible UI fields, and settings described.",
        INTENT_TABLE: "Focus on the table content and answer based on the rows and columns present.",
        INTENT_SECTION: "Focus on the section heading and its body content.",
        INTENT_SUBSECTION: "Focus on recommendations, guidance, or how MaaS360 helps.",
        INTENT_GENERAL: "Use all provided context to answer comprehensively.",
    }.get(intent, "Use all provided context to answer comprehensively.")

    prompt = f"""You are an enterprise document assistant.

{intent_instruction}

Use ONLY the provided context to answer the question.
If the answer is not found in the context, say exactly: "Information not found in document."

Context:
{context}

Question:
{query}

Answer:"""

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


st.set_page_config(page_title="Enterprise Hybrid RAG", layout="wide")
st.title("🏢 Enterprise PDF → Hierarchical Hybrid RAG")

st.markdown("""
**Chunk Taxonomy**: `toc` | `section` | `subsection` | `table` | `image`

**Retrieval**: Intent routing → Type-filtered Dense + BM25 + RRF
""")

if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks = None
if "raw_chunks" not in st.session_state:
    st.session_state.raw_chunks = []

col_upload, col_reset = st.columns([5, 1])
with col_reset:
    if st.button("🔄 Reset"):
        st.session_state.embedded_chunks = None
        st.session_state.raw_chunks = []
        st.rerun()

with col_upload:
    pdf_file = st.file_uploader("Upload Enterprise PDF", type=["pdf"])

if pdf_file and st.session_state.embedded_chunks is None:
    with open("temp_enterprise_rag.pdf", "wb") as f:
        f.write(pdf_file.read())

    groq_client = Groq(api_key=GROQ_API_KEY)

    with st.status("Processing PDF...", expanded=True) as status:
        st.write("Stage 1 — Extracting text, tables, images...")
        all_chunks = build_all_chunks("temp_enterprise_rag.pdf", groq_client)

        n_toc = sum(1 for c in all_chunks if c["chunk_type"] == "toc")
        n_section = sum(1 for c in all_chunks if c["chunk_type"] == "section")
        n_subsection = sum(1 for c in all_chunks if c["chunk_type"] == "subsection")
        n_table = sum(1 for c in all_chunks if c["chunk_type"] == "table")
        n_image = sum(1 for c in all_chunks if c["chunk_type"] == "image")
        st.write(f"Built {len(all_chunks)} chunks: {n_toc} TOC | {n_section} sections | {n_subsection} subsections | {n_table} tables | {n_image} images")

        st.write("Stage 2 — Generating embeddings...")
        progress_bar = st.progress(0.0)
        embedded_chunks = embed_all_chunks(all_chunks, progress_callback=lambda p: progress_bar.progress(p))
        progress_bar.empty()

        st.session_state.embedded_chunks = embedded_chunks
        st.session_state.raw_chunks = all_chunks
        status.update(label="✅ Ingestion complete!", state="complete")

    try:
        os.remove("temp_enterprise_rag.pdf")
    except Exception:
        pass

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("TOC", n_toc)
    col2.metric("Sections", n_section)
    col3.metric("Subsections", n_subsection)
    col4.metric("Tables", n_table)
    col5.metric("Images", n_image)

    with st.expander("📦 View All Chunks"):
        type_icons = {"toc": "📑", "section": "📄", "subsection": "📋", "table": "📊", "image": "🖼️"}
        for c in all_chunks:
            icon = type_icons.get(c["chunk_type"], "📄")
            label = f"{icon} [{c['chunk_type']}] {c['chunk_id']} — {c['heading']} (Page {c.get('page', '?')})"
            with st.expander(label):
                st.text_area("", c["content"], height=160, key=f"view_{c['chunk_id']}")

if st.session_state.embedded_chunks:
    st.divider()
    st.subheader("🔍 Ask Questions")

    query = st.text_input(
        "Enter your question:",
        placeholder="e.g. What settings are in the passcode policy screen? / What are the recommendations under Best Practice #2?"
    )

    if query:
        with st.spinner("Classifying intent and retrieving..."):
            intent, retrieved = route_and_retrieve(query, st.session_state.embedded_chunks, top_k=5)

        intent_colors = {
            "toc": "🟦",
            "table": "🟧",
            "image": "🟪",
            "section": "🟩",
            "subsection": "🟨",
            "general": "⬜"
        }
        st.info(f"{intent_colors.get(intent, '⬜')} **Query Intent Detected**: `{intent.upper()}` — Retrieval filtered to `{intent}` chunk type(s)")

        st.subheader("📊 Retrieved Chunks + Scores")

        type_icons = {"toc": "📑", "section": "📄", "subsection": "📋", "table": "📊", "image": "🖼️"}

        for r in retrieved:
            icon = type_icons.get(r["chunk_type"], "📄")
            label = (
                f"{icon} [{r['chunk_type']}] {r['heading']} "
                f"| RRF: {r['rrf_score']} | Page: {r.get('page', 'N/A')}"
            )
            with st.expander(label):
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Dense Score", r["dense_score"])
                sc2.metric("BM25 Score", r["bm25_score"])
                sc3.metric("Hybrid (RRF)", r["rrf_score"])
                if r.get("parent_heading"):
                    st.caption(f"Parent section: {r['parent_heading']}")
                st.text_area("Content", r["content"], height=220, key=f"ret_{r['chunk_id']}")

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, intent, retrieved, st.session_state.raw_chunks)

        st.subheader("🧠 Final Answer")
        st.success(answer)