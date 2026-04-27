import streamlit as st
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
import base64
import requests


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava:7b"

VISION_PROMPT = """
You are analyzing a page from an enterprise policy document.

Describe clearly:
- Any screenshots or UI elements
- Tables or forms visible
- Diagrams or workflows
- The purpose of the page
Be factual and concise.
"""


# -------------------------------------------------
# OLLAMA VISION CALL
# -------------------------------------------------

def describe_image_with_llava(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": MODEL_NAME,
        "prompt": VISION_PROMPT,
        "images": [image_base64],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
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

    pix = page.get_pixmap(dpi=110)
    image_path = temp_dir / f"page_{page_number}.png"
    pix.save(image_path)

    description = describe_image_with_llava(str(image_path))

    return {
        "page_num": page_number,
        "image_path": image_path,
        "description": description,
        "total_pages": len(doc),
    }


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

def main():
    st.set_page_config(layout="wide")
    st.title("Multimodal PDF Ingestion (Single Page)")

    st.markdown(
        "**Local-only pipeline** | PyMuPDF + Ollama (LLaVA) | Selective vision processing"
    )

    with st.sidebar:
        st.header("Runtime Check")
        st.code(
            "ollama list\nollama run llava:7b",
            language="bash"
        )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        # Read PDF to get page count
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        total_pages = len(doc)

        page_number = st.number_input(
            "Select page number to analyze",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )

        if st.button("Analyze Selected Page", type="primary"):
            with st.spinner("Rendering page and generating description..."):
                result = process_single_page(
                    uploaded_file.getvalue(),
                    page_number
                )

            st.success(
                f"Processed page {result['page_num']} of {result['total_pages']}"
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader(f"Page {result['page_num']}")
                st.image(
                    str(result["image_path"]),
                    use_column_width=True
                )

            with col2:
                st.subheader("Generated Description")
                st.info(result["description"])

    else:
        st.info("Upload a PDF to begin")


if __name__ == "__main__":
    main()
