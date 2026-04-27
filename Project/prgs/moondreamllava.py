import streamlit as st
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
import base64
import requests
import time
from typing import List, Dict
import hashlib


# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava:7b"

RENDER_DPI = 110           # safe DPI
SLEEP_BETWEEN_PAGES = 1.0  # seconds (CRITICAL for thermals)

VISION_PROMPT = (
    "You are analyzing a full page from an enterprise document.\n\n"
    "Write a detailed paragraph describing:\n"
    "- The overall type and purpose of the page\n"
    "- The layout and structure of the page\n"
    "- Any forms, tables, UI elements, diagrams, or workflows visible\n"
    "- What a reader or user is expected to understand or do\n\n"
    "Do not guess unreadable text. Focus on structure and intent."
)


# =========================
# SIMPLE CACHE
# =========================

PAGE_CACHE: Dict[str, str] = {}


def hash_page(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


# =========================
# LLaVA CALL
# =========================

def describe_page_with_llava(image_bytes: bytes) -> str:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": MODEL_NAME,
        "prompt": VISION_PROMPT,
        "images": [image_base64],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()

    return response.json().get("response", "").strip()


# =========================
# PROCESS ENTIRE PDF
# =========================

@st.cache_data(show_spinner=False)
def process_pdf_all_pages(pdf_bytes: bytes) -> List[Dict]:
    temp_dir = Path(tempfile.mkdtemp())
    pdf_path = temp_dir / "input.pdf"
    pdf_path.write_bytes(pdf_bytes)

    doc = fitz.open(pdf_path)
    results = []

    for page_index in range(len(doc)):
        page_num = page_index + 1
        page = doc[page_index]

        pix = page.get_pixmap(dpi=RENDER_DPI)
        image_bytes = pix.tobytes("png")

        page_hash = hash_page(image_bytes)

        if page_hash in PAGE_CACHE:
            description = PAGE_CACHE[page_hash]
        else:
            description = describe_page_with_llava(image_bytes)
            PAGE_CACHE[page_hash] = description

        results.append({
            "page": page_num,
            "description": description
        })

        # IMPORTANT: throttle to avoid overheating
        time.sleep(SLEEP_BETWEEN_PAGES)

    return results


# =========================
# STREAMLIT UI
# =========================

def main():
    st.set_page_config(layout="wide")
    st.title("Multimodal PDF Ingestion — Full Document (Page-Level LLaVA)")

    st.markdown(
        "**Local-only | Page-level vision | LLaVA | Safe sequential processing**"
    )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Analyze Entire Document", type="primary"):
            with st.spinner("Processing pages safely..."):
                results = process_pdf_all_pages(uploaded_file.getvalue())

            st.success(f"Processed {len(results)} pages")

            # Combined RAG-ready output
            st.subheader("Combined Page Descriptions (RAG-ready)")
            combined_text = "\n\n".join(
                f"## Page {r['page']}\n{r['description']}"
                for r in results
            )

            st.text_area(
                "Full Document Description",
                combined_text,
                height=500
            )

            st.divider()

            # Per-page display
            for r in results:
                with st.expander(f"Page {r['page']}"):
                    st.write(r["description"])

    else:
        st.info("Upload a PDF to begin")


if __name__ == "__main__":
    main()
