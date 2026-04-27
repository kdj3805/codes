# ============================================================
# ENTERPRISE PDF PIPELINE
# TEXT + TABLES (PyMuPDF) + IMAGES (OCR) + GROQ + CLEAN ADAPTIVE CHUNKING
# ============================================================

import os
import io
import json
import re
import hashlib
import fitz
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from typing import List, Dict, Any, Tuple
from groq import Groq
import easyocr

# =========================
# ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

OCR_READER = easyocr.Reader(["en"], gpu=False)

# =========================
# IMAGE PROMPT
# =========================
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

# =========================
# IMAGE OCR
# =========================
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


# =========================
# IMAGE EXTRACTION
# =========================
def is_logo_image(ocr_text: str) -> bool:
    text = ocr_text.lower().strip()
    return len(text) < 20 and ("ibm" in text or "maas360" in text)


def extract_images(pdf_path: str) -> List[Dict[str, Any]]:
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


# =========================
# TEXT + TABLE EXTRACTION
# =========================
def extract_text_tables(pdf_path: str) -> List[Dict[str, Any]]:
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


# =========================
# IMAGE DESCRIPTION
# =========================
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
        desc = res.choices[0].message.content.strip()
        if len(desc) < 10:
            return f"Image contains: {ocr_text[:80]}"
        return desc
    except Exception:
        return f"Image contains: {ocr_text[:80]}"


# =========================
# TOC NORMALIZATION
# =========================
def build_toc_chunk(md_text: str) -> Tuple[str, str]:
    toc_entries = []
    new_lines = []

    for line in md_text.splitlines():
        if re.search(r"\.{4,}\s*\d+$", line):
            title = re.sub(r"\.{4,}.*", "", line).strip()
            page = re.findall(r"(\d+)$", line)
            if title:
                toc_entries.append((title, page[0] if page else None))
        else:
            new_lines.append(line)

    if not toc_entries:
        return "", md_text

    seen = set()
    unique = []
    for t, p in toc_entries:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append((t, p))

    toc_lines = ["## Table of Contents"]
    for t, p in unique:
        if p:
            toc_lines.append(f"- {t} (page {p})")
        else:
            toc_lines.append(f"- {t}")

    toc_chunk = "\n".join(toc_lines)
    return toc_chunk, "\n".join(new_lines)


# =========================
# TABLE DETECTION (FIXED)
# =========================
def looks_like_table(text: str) -> bool:
    lines = text.splitlines()

    if "|" in text and "---" in text:
        return True

    if len(lines) >= 3:
        lengths = [len(l.split()) for l in lines if l.strip()]
        if len(set(lengths)) == 1:
            return True

    numeric_lines = sum(1 for l in lines if re.match(r"^\d+[\.\)]", l.strip()))
    if numeric_lines >= 3:
        return True

    return False


# =========================
# ADAPTIVE CHUNKING
# =========================
MAX_CHARS = 1400


def split_blocks(md_text: str):
    blocks = []
    pattern = r"(<imagedesc>.*?</imagedesc>)"
    parts = re.split(pattern, md_text, flags=re.DOTALL)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("<imagedesc>"):
            blocks.append(("image", part))
        else:
            sections = part.split("\n\n")
            buffer = ""

            for s in sections:
                s = s.strip()
                if not s:
                    continue

                if looks_like_table(s):
                    if buffer:
                        blocks.append(("text", buffer.strip()))
                        buffer = ""
                    blocks.append(("table", s))
                else:
                    buffer += s + "\n\n"

            if buffer.strip():
                blocks.append(("text", buffer.strip()))

    return blocks


def adaptive_chunk(md_text: str):
    toc_chunk, md_text = build_toc_chunk(md_text)
    blocks = split_blocks(md_text)
    chunks = []

    text_buffer = ""
    text_id = 0
    table_id = 0
    image_id = 0

    if toc_chunk:
        chunks.append({
            "chunk_id": "toc_0",
            "chunk_type": "toc",
            "content": toc_chunk
        })

    for block_type, content in blocks:

        if block_type == "image":
            if text_buffer:
                chunks.append({
                    "chunk_id": f"text_{text_id}",
                    "chunk_type": "text",
                    "content": text_buffer.strip()
                })
                text_id += 1
                text_buffer = ""

            chunks.append({
                "chunk_id": f"image_{image_id}",
                "chunk_type": "image",
                "content": content
            })
            image_id += 1
            continue

        if block_type == "table":
            if text_buffer:
                chunks.append({
                    "chunk_id": f"text_{text_id}",
                    "chunk_type": "text",
                    "content": text_buffer.strip()
                })
                text_id += 1
                text_buffer = ""

            chunks.append({
                "chunk_id": f"table_{table_id}",
                "chunk_type": "table",
                "content": content
            })
            table_id += 1
            continue

        if len(text_buffer) + len(content) < MAX_CHARS:
            text_buffer += content + "\n\n"
        else:
            chunks.append({
                "chunk_id": f"text_{text_id}",
                "chunk_type": "text",
                "content": text_buffer.strip()
            })
            text_id += 1
            text_buffer = content + "\n\n"

    if text_buffer.strip():
        chunks.append({
            "chunk_id": f"text_{text_id}",
            "chunk_type": "text",
            "content": text_buffer.strip()
        })

    return chunks


# =========================
# MAIN PIPELINE
# =========================
def process_pdf(pdf_path: str) -> Tuple[str, str, Dict[str, int]]:
    client = Groq(api_key=GROQ_API_KEY)

    text_items = extract_text_tables(pdf_path)
    image_items = extract_images(pdf_path)

    for img in image_items:
        img["description"] = describe_image(img["ocr"], client)

    pages: Dict[int, List[Dict[str, Any]]] = {}
    for item in text_items + image_items:
        pages.setdefault(item["page"], []).append(item)

    raw_blocks = []
    md_blocks = []

    for page in sorted(pages.keys()):
        page_items = pages[page]

        for t in sorted([i for i in page_items if i["type"] == "text"], key=lambda x: x["order"]):
            raw_blocks.append(t["content"])
            md_blocks.append(t["content"])

        for t in sorted([i for i in page_items if i["type"] == "table"], key=lambda x: x["order"]):
            raw_blocks.append(t["content"])
            md_blocks.append(t["content"])

        for img in sorted([i for i in page_items if i["type"] == "image"], key=lambda x: x["order"]):
            raw_blocks.append(f"[IMAGE Page {page+1}]")
            raw_blocks.append(img["ocr"])
            md_blocks.append(f"""<imagedesc>
Page {page+1}, Image {img["img_no"]}

OCR Text:
{img["ocr"]}

Description:
{img["description"]}
</imagedesc>""")

    stats = {
        "text_blocks": len([i for i in text_items if i["type"] == "text"]),
        "tables": len([i for i in text_items if i["type"] == "table"]),
        "images": len(image_items)
    }

    return "\n\n".join(raw_blocks), "\n\n".join(md_blocks), stats


# =========================
# STREAMLIT UI
# =========================
st.set_page_config("Enterprise PDF Extractor", layout="wide")
st.title("📄 Enterprise PDF Extraction + Adaptive Chunking")

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    with st.spinner("Extracting document…"):
        raw, md, stats = process_pdf("temp.pdf")
        chunks = adaptive_chunk(md)

    st.success("PDF processing completed successfully!")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Text Blocks", stats["text_blocks"])
    c2.metric("Tables", stats["tables"])
    c3.metric("Images", stats["images"])
    c4.metric("Final Chunks", len(chunks))

    st.download_button("Download raw_extracted.txt", raw, "raw_extracted.txt")
    st.download_button("Download enriched_output.md", md, "enriched_output.md")
    st.download_button(
        "Download final_chunks.json",
        data=json.dumps(chunks, indent=2, ensure_ascii=False),
        file_name="final_chunks.json"
    )

    st.subheader("Chunk Preview")
    st.json(chunks[:5])

    os.remove("temp.pdf")
