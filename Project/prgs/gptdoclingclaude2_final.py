# ============================================================
# FINAL, WORKING ENTERPRISE PDF EXTRACTION PIPELINE
# TEXT + TABLES (PyMuPDF) + IMAGES (OCR) + GROQ gpt-oss-120b
# THIS FIXES THE ROOT CAUSE PERMANENTLY
# ============================================================

import os
import io
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from typing import List, Dict, Any, Tuple
from groq import Groq

# =========================
# ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# =========================
# OCR
# =========================
import easyocr
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
def extract_images(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    images = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            ocr = ocr_image(base["image"])
            if len(ocr) < 3:
                continue
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
# TEXT + TABLE EXTRACTION (THE FIX)
# =========================
def extract_text_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Docling FAILS on this PDF.
    PyMuPDF text extraction WORKS.
    This is the authoritative fix.
    """
    doc = fitz.open(pdf_path)
    items = []
    seq = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # -------- TEXT BLOCKS --------
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

        # -------- TABLE DETECTION (HEURISTIC) --------
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
# IMAGE DESCRIPTION (GROQ)
# =========================
def describe_image(ocr_text: str, client: Groq) -> str:
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
st.title("📄 Enterprise PDF Extraction (TEXT FIXED)")

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    with st.spinner("Extracting document…"):
        raw, md, stats = process_pdf("temp.pdf")

    st.success("PDF processing completed successfully!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Text Blocks", stats["text_blocks"])
    c2.metric("Tables", stats["tables"])
    c3.metric("Images", stats["images"])

    st.download_button("Download raw_extracted.txt", raw, "raw_extracted.txt")
    st.download_button("Download enriched_output.md", md, "enriched_output.md")

    st.subheader("Preview (Markdown)")
    st.text_area("", md[:6000], height=500)

    os.remove("temp.pdf")
