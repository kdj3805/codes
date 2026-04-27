import streamlit as st
import tempfile
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Any
import io
import base64
import requests
import hashlib
from PIL import Image
import warnings

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import TableItem, PictureItem


# =========================
# SILENCE NON-CRITICAL WARNINGS
# =========================

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MOONDREAM_MODEL = "moondream"

MIN_IMAGE_AREA = 150 * 150      # skip tiny images
MAX_IMAGES_PER_PAGE = 2         # hard cap
ENABLE_IMAGE_DESCRIPTION = True # default ON


# =========================
# IMAGE DESCRIPTION CACHE
# =========================

IMAGE_DESC_CACHE: Dict[str, str] = {}


def image_hash(img: Image.Image) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()


# =========================
# MOONDREAM IMAGE DESCRIPTION
# =========================

def describe_image_with_moondream(img_pil: Image.Image) -> str:
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")

    payload = {
        "model": MOONDREAM_MODEL,
        "prompt": (
            "You are analyzing an image for detailed documentation and retrieval.\n\n"
            "Write a detailed, multi-sentence paragraph describing exactly what is visible in the image. "
            "Do NOT give a short or generic summary.\n\n"
            "You MUST include:\n"
            "- The type of page or screen shown (for example: a configuration form, registration page, dashboard, or settings screen).\n"
            "- The layout and structure of the page, including how sections are arranged vertically or horizontally.\n"
            "- The kinds of elements present (such as headers, explanatory text, form fields, tables, and buttons) and where they appear on the page.\n"
            "- What a user is expected to do on this page, based on the visible controls and layout.\n\n"
            "Describe relationships between elements, such as forms being followed by tables, "
            "or buttons being placed after user input fields.\n\n"
            "If text is too small or unclear to read exactly, do NOT guess the wording. "
            "Instead, describe the role or purpose of that text (for example: 'an explanatory paragraph', "
            "'a labeled input field', or 'a multi-row table').\n\n"
            "Write in complete sentences. Avoid vague phrases such as "
            "'this image shows', 'appears to be', or 'some text'. "
            "Provide a rich, specific description suitable for search and retrieval."
        ),
        "images": [base64.b64encode(buffered.getvalue()).decode()],
        "stream": False
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=45)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        return ""


# =========================
# DOCLING CONVERTER (NO OCR)
# =========================

def create_converter() -> DocumentConverter:
    pipeline_opts = PdfPipelineOptions(
        do_table_structure=True,
        generate_picture_images=True,
        do_picture_description=False,
        do_ocr=False
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_opts
            )
        }
    )


# =========================
# IMAGE DESCRIPTION EXTRACTION (FAST)
# =========================

def extract_image_descriptions(pdf_path: str, page_no: int):
    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]

    descriptions = []
    count = 0

    for img_index, img in enumerate(page.get_images(full=True)):
        if count >= MAX_IMAGES_PER_PAGE:
            break

        xref = img[0]

        try:
            pix = fitz.Pixmap(doc, xref)
            area = pix.width * pix.height

            if pix.n < 5 and area >= MIN_IMAGE_AREA:
                mode = "L" if pix.n == 1 else "RGB"
                img_pil = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

                h = image_hash(img_pil)
                if h in IMAGE_DESC_CACHE:
                    desc = IMAGE_DESC_CACHE[h]
                else:
                    desc = describe_image_with_moondream(img_pil)
                    IMAGE_DESC_CACHE[h] = desc

                if desc:
                    descriptions.append(desc)
                    count += 1

            pix = None

        except Exception:
            continue

    doc.close()
    return descriptions


# =========================
# MAIN PDF PROCESSOR
# =========================

def process_pdf(pdf_bytes: bytes, max_pages: int | None) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        pdf_path = f.name

    try:
        converter = create_converter()
        result = converter.convert(pdf_path)
        doc = result.document

        raw_text_parts = []
        structured_parts = []

        stats = {
            "total_pages": 0,
            "text_items": 0,
            "tables": 0,
            "images_described": 0
        }

        processed_pages = set()
        current_page = 0

        for item, _ in doc.iterate_items():

            if hasattr(item, "prov") and item.prov:
                page_no = item.prov[0].page_no

                if page_no != current_page:
                    current_page = page_no
                    stats["total_pages"] = current_page

                    if max_pages and current_page > max_pages:
                        break

                    structured_parts.append(f"\n\n## Page {current_page}\n")
                    raw_text_parts.append(f"\n\n=== PAGE {current_page} ===\n")

                    if current_page not in processed_pages and ENABLE_IMAGE_DESCRIPTION:
                        images = extract_image_descriptions(pdf_path, current_page)
                        for desc in images:
                            structured_parts.append(
                                f"\n<imagedesc>\n{desc}\n</imagedesc>\n"
                            )
                            raw_text_parts.append(desc)
                            stats["images_described"] += 1

                        processed_pages.add(current_page)

            if isinstance(item, TableItem):
                stats["tables"] += 1
                try:
                    table_md = item.export_to_markdown(doc=doc)
                except Exception:
                    table_md = item.export_to_markdown()

                structured_parts.append(f"\n<tableinfo>\n{table_md}\n</tableinfo>\n")
                raw_text_parts.append(table_md)

            elif isinstance(item, PictureItem):
                continue

            else:
                stats["text_items"] += 1
                try:
                    text = item.export_to_markdown()
                except Exception:
                    text = getattr(item, "text", "")

                if text and text.strip():
                    structured_parts.append(text)
                    raw_text_parts.append(text)

        return {
            "raw_text": "\n\n".join(raw_text_parts),
            "structured_output": "\n\n".join(structured_parts),
            "diagnostics": stats
        }

    finally:
        Path(pdf_path).unlink(missing_ok=True)


# =========================
# STREAMLIT UI
# =========================

def main():
    st.set_page_config("Multimodal RAG Ingestion (No OCR)", layout="wide")
    st.title("Multimodal RAG PDF Ingestion — Fast Mode")

    with st.sidebar:
        st.header("Performance Settings")
        max_pages = st.number_input("Limit pages (0 = all)", 0, 100, 0)

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and st.button("Process Document", type="primary"):
        with st.spinner("Processing (CPU-friendly)…"):
            result = process_pdf(
                uploaded.getvalue(),
                None if max_pages == 0 else max_pages
            )

        diag = result["diagnostics"]

        st.success(f"Processed {diag['total_pages']} pages")

        col1, col2, col3 = st.columns(3)
        col1.metric("Text Items", diag["text_items"])
        col2.metric("Tables", diag["tables"])
        col3.metric("Images Described", diag["images_described"])

        tab1, tab2 = st.tabs(["RAW TEXT (RAG)", "STRUCTURED OUTPUT"])

        with tab1:
            st.text_area("Raw Text", result["raw_text"], height=600)

        with tab2:
            st.code(result["structured_output"], language="markdown")

    elif not uploaded:
        st.info("Upload a PDF to begin")


if __name__ == "__main__":
    main()
