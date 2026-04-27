# ============================================================
# Enterprise PDF → Structured Extraction → Adaptive Chunking
# FIXED VERSION
#
# - Reliable text + table extraction via Docling
# - OCR + LLM description for images (Groq gpt-oss-120b)
# - Logo / header image deduplication
# - Proper Table of Contents normalization
# - Adaptive chunking (text / table / image kept separate)
# - <imagedesc> preserved as atomic chunks
#
# REQUIREMENTS
# pip install streamlit docling pymupdf pillow easyocr groq python-dotenv
# ============================================================

import os
import re
import io
import json
import fitz  # PyMuPDF
import hashlib
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
from PIL import Image
import easyocr
from groq import Groq

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

# =========================
# ENV + CONFIG
# =========================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

IMAGE_MODEL = "openai/gpt-oss-120b"
MIN_LOGO_TEXT_LEN = 15

# =========================
# IMAGE DESCRIPTION PROMPT
# =========================

IMAGE_DESCRIPTION_PROMPT = """
You are analyzing OCR text extracted from a screenshot or diagram in an enterprise policy document.

Rules:
- Use ONLY the OCR text
- Do NOT guess
- Be factual and concise
- If the image is just a logo or repeated header, say so

OCR TEXT:
{ocr_text}
"""

# =========================
# OCR + IMAGE EXTRACTION
# =========================

def extract_images_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    reader = easyocr.Reader(["en"], gpu=False)
    pdf = fitz.open(pdf_path)

    images = []

    for page_idx in range(len(pdf)):
        page = pdf[page_idx]
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = pdf.extract_image(xref)
            img_bytes = base["image"]

            pil = Image.open(io.BytesIO(img_bytes)).convert("L")
            buf = io.BytesIO()
            pil.save(buf, format="PNG")

            ocr_lines = reader.readtext(buf.getvalue(), detail=0, paragraph=True)
            ocr_text = "\n".join(ocr_lines).strip()

            if not ocr_text:
                continue

            images.append({
                "page": page_idx + 1,
                "image_id": f"p{page_idx+1}_i{img_idx+1}",
                "ocr_text": ocr_text
            })

    pdf.close()
    return images

# =========================
# IMAGE DEDUP (LOGOS)
# =========================

def is_repeated_logo(ocr_text: str) -> bool:
    text = ocr_text.lower().strip()
    return (
        len(text) < MIN_LOGO_TEXT_LEN and
        ("maas360" in text or "ibm" in text)
    )

# =========================
# GROQ IMAGE DESCRIPTION
# =========================

def describe_image(ocr_text: str, groq: Groq) -> str:
    prompt = IMAGE_DESCRIPTION_PROMPT.format(ocr_text=ocr_text)
    resp = groq.chat.completions.create(
        model=IMAGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()

# =========================
# DOCLING TEXT + TABLES
# =========================

def extract_text_tables(pdf_path: str) -> List[Dict[str, Any]]:
    opts = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=False,
        do_picture_description=False,
    )

    doc = DocumentConverter().convert(pdf_path).document
    items = []

    for item, _ in doc.iterate_items():
        label = str(getattr(item, "label", "")).upper()
        
        # Skip picture items - handled separately with OCR
        if "PICTURE" in label:
            continue
        
        # Get page number safely
        page = None
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(prov, "page_no"):
                page = prov.page_no

        if "TABLE" in label and hasattr(item, "export_to_markdown"):
            try:
                md = item.export_to_markdown()
                if md and md.strip():
                    items.append({
                        "type": "table",
                        "page": page,
                        "content": md.strip()
                    })
            except Exception:
                pass

        else:
            # Use getattr to safely access text attribute
            text = getattr(item, "text", None)
            if text and text.strip():
                items.append({
                    "type": "text",
                    "page": page,
                    "content": text.strip()
                })

    return items

# =========================
# TOC NORMALIZATION
# =========================

def normalize_toc(text: str) -> str:
    lines = text.splitlines()
    toc = []

    for l in lines:
        if re.search(r"\.{5,}", l):
            title = re.sub(r"\.{5,}.*", "", l).strip()
            page = re.findall(r"(\d+)$", l)
            toc.append(f"- {title} (page {page[0]})" if page else f"- {title}")

    return "\n".join(toc)

# =========================
# ADAPTIVE CHUNKING
# =========================

def chunk_content(
    text_items: List[Dict],
    image_items: List[Dict],
    groq: Groq
) -> List[Dict]:

    chunks = []
    seen_logos = set()

    # ---- TEXT / TABLES ----
    buffer = ""
    chunk_id = 0

    for item in text_items:
        content = item["content"]

        if "table of contents" in content.lower():
            content = "## Table of Contents\n" + normalize_toc(content)

        buffer += content + "\n\n"

        if len(buffer) > 1200:
            chunks.append({
                "chunk_type": "text",
                "chunk_id": f"text_{chunk_id}",
                "content": buffer.strip()
            })
            chunk_id += 1
            buffer = ""

    if buffer.strip():
        chunks.append({
            "chunk_type": "text",
            "chunk_id": f"text_{chunk_id}",
            "content": buffer.strip()
        })

    # ---- IMAGES ----
    img_id = 0
    for img in image_items:
        if is_repeated_logo(img["ocr_text"]):
            h = hashlib.md5(img["ocr_text"].encode()).hexdigest()
            if h in seen_logos:
                continue
            seen_logos.add(h)

        desc = describe_image(img["ocr_text"], groq)

        chunks.append({
            "chunk_type": "image",
            "chunk_id": f"image_{img_id}",
            "content": (
                f"Page {img['page']}\n\n"
                f"OCR Text:\n{img['ocr_text']}\n\n"
                f"Description:\n{desc}"
            )
        })
        img_id += 1

    return chunks

# =========================
# STREAMLIT APP
# =========================

def main():
    st.set_page_config(layout="wide")
    st.title("Enterprise PDF → Clean RAG Chunks")

    pdf = st.file_uploader("Upload Enterprise PDF", type=["pdf"])

    if not pdf:
        return

    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY")
        return

    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    groq = Groq(api_key=GROQ_API_KEY)

    with st.spinner("Extracting text & tables..."):
        text_items = extract_text_tables("temp.pdf")

    with st.spinner("Extracting images + OCR..."):
        image_items = extract_images_with_ocr("temp.pdf")

    with st.spinner("Chunking content..."):
        chunks = chunk_content(text_items, image_items, groq)

    with open("final_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    st.success("Extraction complete")

    st.metric("Text Blocks", sum(1 for c in chunks if c["chunk_type"] == "text"))
    st.metric("Image Blocks", sum(1 for c in chunks if c["chunk_type"] == "image"))

    st.download_button(
        "Download final_chunks.json",
        data=json.dumps(chunks, indent=2, ensure_ascii=False),
        file_name="final_chunks.json",
        mime="application/json"
    )

    st.subheader("Preview")
    st.json(chunks[:3])

    os.remove("temp.pdf")


if __name__ == "__main__":
    main()
