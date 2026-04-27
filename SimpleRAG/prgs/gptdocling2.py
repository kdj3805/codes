#TRIAL 2

import os
import io
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# -----------------------------
# OCR CONFIG
# -----------------------------
USE_EASYOCR = True  # False → RapidOCR

if USE_EASYOCR:
    import easyocr
    OCR_READER = easyocr.Reader(["en"], gpu=False)
else:
    from rapidocr_onnxruntime import RapidOCR
    OCR_READER = RapidOCR()

# -----------------------------
# DOCLING
# -----------------------------
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DocItemLabel

# -----------------------------
# GROQ (TEXT-ONLY, SAFE)
# -----------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# ENV
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# STREAMLIT
# -----------------------------
st.set_page_config(page_title="PDF Extraction (Docling + OCR + Groq)", layout="wide")
st.title("📄 Clean Multimodal PDF Extraction (Correct Order)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

OUTPUT_DIR = "extraction_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# OCR HELPER
# -----------------------------
def ocr_image_bytes(image_bytes: bytes) -> str:
    try:
        if USE_EASYOCR:
            result = OCR_READER.readtext(image_bytes, detail=0, paragraph=True)
            return "\n".join(result).strip()
        else:
            result, _ = OCR_READER(image_bytes)
            return " ".join(r[1] for r in result).strip()
    except Exception:
        return ""

# -----------------------------
# IMAGE DESCRIPTION PROMPT
# -----------------------------
IMAGE_DESC_PROMPT = ChatPromptTemplate.from_template(
    """
You are given OCR-extracted text from a screenshot in an enterprise IT / security policy document.

Your task:
- Identify what screen, feature, or configuration this represents
- List the visible settings or fields (if any)
- Explain its purpose in the policy context

Rules:
- Use ONLY the OCR text
- Do NOT guess missing UI
- Do NOT hallucinate features
- Be concise, factual, and structured
- If OCR text is unclear, say so explicitly

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
    opts.generate_picture_images = False  # IMPORTANT

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    st.info("🔍 Extracting text & tables with Docling…")
    result = converter.convert(pdf_path)
    doc = result.document

    # -----------------------------
    # GROUP TEXT BY PAGE (ORDERED)
    # -----------------------------
    page_text = {}

    for item, _ in doc.iterate_items():
        if not hasattr(item, "text") or not item.text:
            continue

        if item.label not in {
            DocItemLabel.TITLE,
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.TEXT,
            DocItemLabel.LIST_ITEM,
            DocItemLabel.CAPTION,
            DocItemLabel.TABLE,
        }:
            continue

        page_no = 0
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(prov, "page_no"):
                page_no = prov.page_no - 1

        content = item.text.strip()

        if item.label == DocItemLabel.TABLE and hasattr(item, "export_to_markdown"):
            try:
                content = f"<table>\n{item.export_to_markdown(doc=doc)}\n</table>"
            except Exception:
                pass

        page_text.setdefault(page_no, []).append(content)

    # -----------------------------
    # INIT GROQ LLM (TEXT ONLY)
    # -----------------------------
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b",
        temperature=0.0
    )

    # -----------------------------
    # PAGE-BY-PAGE MERGE (TEXT + IMAGES)
    # -----------------------------
    pdf = fitz.open(pdf_path)
    output_blocks = []
    image_counter = 0

    for page_idx in range(len(pdf)):
        # ---- Page Text ----
        for block in page_text.get(page_idx, []):
            output_blocks.append(block)

        # ---- Page Images ----
        page = pdf[page_idx]
        images = page.get_images(full=True)

        for img in images:
            xref = img[0]
            base = pdf.extract_image(xref)
            image_bytes = base["image"]

            ocr_text = ocr_image_bytes(image_bytes)
            if len(ocr_text) < 20:
                continue

            image_counter += 1
            with st.spinner(f"🖼️ Describing image {image_counter}…"):
                response = llm.invoke(
                    IMAGE_DESC_PROMPT.format_messages(ocr_text=ocr_text)
                )

            output_blocks.append(
                f"""<imagedesc>
Page {page_idx + 1}, Image {image_counter}
{response.content.strip()}
</imagedesc>"""
            )

    pdf.close()

    # -----------------------------
    # OUTPUT
    # -----------------------------
    final_text = "\n\n".join(output_blocks)

    st.subheader("📄 Raw Extracted Content (Ordered)")
    st.text_area("Raw Text", final_text, height=650)

    st.subheader("📄 Markdown Preview")
    st.markdown(final_text, unsafe_allow_html=True)

    st.success(f"✅ Done — {image_counter} images described in correct document order")
