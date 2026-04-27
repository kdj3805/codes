import streamlit as st
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image
import tempfile
from pathlib import Path
import requests


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "llama3"  # or qwen3, deepseek-r1

LLM_PROMPT_TEMPLATE = """
You are analyzing OCR-extracted text from an enterprise policy document page.

Tasks:
1. Understand the content
2. Explain what this page represents
3. Mention any tables, settings, or workflows
4. Summarize clearly and concisely

OCR TEXT:
{ocr_text}
"""


# -------------------------------------------------
# INITIALIZE OCR (ONCE)
# -------------------------------------------------

ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)


# -------------------------------------------------
# OCR + LLM
# -------------------------------------------------

def run_ocr(image_path: str) -> str:
    result = ocr_engine.ocr(image_path, cls=True)
    lines = []

    for page in result:
        for line in page:
            text = line[1][0]
            lines.append(text)

    return "\n".join(lines)


def summarize_with_llm(ocr_text: str) -> str:
    prompt = LLM_PROMPT_TEMPLATE.format(ocr_text=ocr_text)

    payload = {
        "model": TEXT_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    return response.json().get("response", "").strip()


# -------------------------------------------------
# PDF PROCESSING (SINGLE PAGE)
# -------------------------------------------------

@st.cache_data(show_spinner=True)
def process_single_page(pdf_bytes: bytes, page_number: int):
    temp_dir = Path(tempfile.mkdtemp())
    pdf_path = temp_dir / "input.pdf"
    pdf_path.write_bytes(pdf_bytes)

    doc = fitz.open(pdf_path)

    page_index = page_number - 1
    page = doc[page_index]

    pix = page.get_pixmap(dpi=200)
    image_path = temp_dir / f"page_{page_number}.png"
    pix.save(image_path)

    ocr_text = run_ocr(str(image_path))
    summary = summarize_with_llm(ocr_text)

    return {
        "page_num": page_number,
        "image_path": image_path,
        "ocr_text": ocr_text,
        "summary": summary,
        "total_pages": len(doc),
    }


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

def main():
    st.set_page_config(layout="wide")
    st.title("OCR + Text LLM PDF Ingestion (No Admin Required)")

    st.markdown(
        "**Pure Python pipeline** | PyMuPDF + PaddleOCR + Ollama (Text LLM)"
    )

    with st.sidebar:
        st.header("Runtime Check")
        st.info(f"**Text LLM**: {TEXT_MODEL}")
        st.code(
            "ollama list\nollama run llama3",
            language="bash"
        )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        total_pages = len(doc)

        page_number = st.number_input(
            "Select page number to analyze",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )

        if st.button("Analyze Page", type="primary"):
            with st.spinner("Running OCR and summarization..."):
                result = process_single_page(
                    uploaded_file.getvalue(),
                    page_number
                )

            st.success(
                f"Processed page {result['page_num']} of {result['total_pages']}"
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Rendered Page")
                st.image(
                    str(result["image_path"]),
                    use_column_width=True
                )

            with col2:
                st.subheader("Generated Description")
                st.info(result["summary"])

            with st.expander("Raw OCR Text"):
                st.text(result["ocr_text"])

    else:
        st.info("Upload a PDF to begin")


if __name__ == "__main__":
    main()
