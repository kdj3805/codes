#TRIAL
import os
import io
import fitz  # PyMuPDF
import time
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# -----------------------------
# OCR CONFIG
# -----------------------------
USE_EASYOCR = True  # False → RapidOCR

if USE_EASYOCR:
    import easyocr
else:
    from rapidocr_onnxruntime import RapidOCR

# -----------------------------
# DOCLING
# -----------------------------
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat

# -----------------------------
# GROQ
# -----------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# ENV
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# -----------------------------
# STREAMLIT
# -----------------------------
st.set_page_config(
    page_title="PDF Extraction (Ordered Images)",
    layout="wide"
)
st.title("📄 PDF Extraction + Ordered Image Descriptions (Groq)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

OUTPUT_DIR = "extraction_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# OCR HELPER
# -----------------------------
def ocr_image_bytes(image_bytes: bytes) -> str:
    if USE_EASYOCR:
        reader = easyocr.Reader(["en"], gpu=False)
        text = reader.readtext(image_bytes, detail=0, paragraph=True)
        return "\n".join(text).strip()
    else:
        ocr = RapidOCR()
        result, _ = ocr(image_bytes)
        return " ".join(r[1] for r in result).strip()

# -----------------------------
# IMAGE DESCRIPTION PROMPT
# -----------------------------
IMAGE_DESC_PROMPT = ChatPromptTemplate.from_template(
    """
You are given OCR-extracted text from a screenshot in an enterprise policy document.

Describe:
- What screen or feature this represents
- What settings or options are visible
- What purpose it serves in the policy context

Be factual, concise, and do NOT hallucinate.

OCR text:
{ocr_text}
"""
)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if uploaded_file:
    pdf_path = os.path.join(OUTPUT_DIR, "input.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # -----------------------------
    # DOCLING CONFIG
    # -----------------------------
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.generate_picture_images = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    st.info("🔍 Running Docling extraction…")
    result = converter.convert(pdf_path)
    doc = result.document

    # -----------------------------
    # GROUP DOCLING ITEMS BY PAGE
    # -----------------------------
    page_text_map = {}

    for item, _ in doc.iterate_items():
        if not hasattr(item, "text") or not item.text:
            continue

        label_name = item.label.name if hasattr(item.label, "name") else str(item.label)

        if label_name not in {
            "TITLE",
            "SECTION_HEADER",
            "PARAGRAPH",
            "TEXT",
            "LIST_ITEM",
            "CAPTION",
        }:
            continue

        page_no = 0
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(prov, "page_no"):
                page_no = prov.page_no - 1

        page_text_map.setdefault(page_no, []).append(item.text.strip())

    # -----------------------------
    # INIT GROQ LLM
    # -----------------------------
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b"
    )

    # -----------------------------
    # PAGE-BY-PAGE ASSEMBLY
    # -----------------------------
    pdf = fitz.open(pdf_path)
    extracted_blocks = []
    image_counter = 0

    for page_idx in range(len(pdf)):
        # --- Page Text ---
        if page_idx in page_text_map:
            extracted_blocks.extend(page_text_map[page_idx])

        # --- Page Images ---
        page = pdf[page_idx]
        images = page.get_images(full=True)

        for img in images:
            xref = img[0]
            base = pdf.extract_image(xref)
            image_bytes = base["image"]

            ocr_text = ocr_image_bytes(image_bytes)
            if len(ocr_text) < 10:
                continue

            image_counter += 1

            with st.spinner(f"Describing image {image_counter} (Groq)…"):
                response = llm.invoke(
                    IMAGE_DESC_PROMPT.format_messages(ocr_text=ocr_text)
                )

            image_desc = response.content.strip()

            extracted_blocks.append(
                f"""<imagedesc>
Page {page_idx + 1}, Image {image_counter}
{image_desc}
</imagedesc>"""
            )

    pdf.close()

    # -----------------------------
    # OUTPUT
    # -----------------------------
    final_text = "\n\n".join(extracted_blocks)

    st.subheader("📄 Extracted Content (Correct Order)")
    st.text_area(
        "Final extracted document",
        final_text,
        height=650
    )

    st.success(f"✅ Extraction complete — {image_counter} images described in correct order")
