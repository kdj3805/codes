#TRIAL ONE 
import os
import io
import fitz  # PyMuPDF
import ollama
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# -----------------------------
# OCR CONFIG (CHOOSE ONE)
# -----------------------------
USE_EASYOCR = True   # set False to use RapidOCR

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
from docling.datamodel.document import DocItemLabel

# -----------------------------
# ENV
# -----------------------------
load_dotenv()

# -----------------------------
# STREAMLIT SETUP
# -----------------------------
st.set_page_config(page_title="Docling + NuMarkdown (No Chunking)", layout="wide")
st.title("PDF → Markdown (Docling + PyMuPDF + NuMarkdown-Thinking-8B)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

OUTPUT_DIR = Path("extraction_debug")
IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# PRODUCTION PROMPT (YOUR REQUEST)
# -----------------------------
NU_MARKDOWN_PROMPT = """
You are given content extracted from a PDF document, including:
- Text blocks
- Tables
- Image descriptions or screenshots

Your task is to reconstruct the document faithfully in GitHub-Flavored Markdown.

Follow these rules strictly:

DOCUMENT STRUCTURE
- Use `#` for the document title
- Use `##` for major sections (e.g., “Best Practice #1”)
- Use `###` for subsections (e.g., “Our Recommendations”, “How MaaS360 Helps”)
- Preserve original section names and numbering exactly

TABLE OF CONTENTS
- If a Table of Contents exists, include it
- Represent it as a Markdown list
- Preserve section order and hierarchy

TEXT AND LISTS
- Preserve paragraph boundaries
- Convert bullet points and numbered lists to Markdown lists
- Do not merge unrelated paragraphs

TABLES
- Reconstruct tables using Markdown table syntax
- Preserve headers, rows, and multi-line cells
- Do not convert tables into prose

IMAGES AND SCREENSHOTS
- Treat screenshots as informational content
- Summarize visible UI elements, labels, and settings accurately
- Do not hallucinate unseen content

FOOTERS AND BOILERPLATE
- Consolidate repeated footers or legal text into a single final section unless unique

OUTPUT RULES
- Output only the final Markdown
- Do not include explanations, reasoning, or meta commentary
- Do not omit any meaningful content

Produce the reconstructed document now.
"""

# -----------------------------
# OCR TEXT EXTRACTION FROM IMAGES
# -----------------------------
def ocr_image_bytes(image_bytes: bytes) -> str:
    """Extract text from image bytes using OCR (not vision LLM)"""
    try:
        if USE_EASYOCR:
            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(image_bytes, detail=0, paragraph=True)
            return "\n".join(results) if results else ""
        else:
            ocr = RapidOCR()
            result, _ = ocr(image_bytes)
            if result:
                return " ".join(r[1] for r in result)
            return ""
    except Exception as e:
        return f"[OCR failed: {str(e)[:80]}]"

# -----------------------------
# MAIN PIPELINE (NO CHUNKING)
# -----------------------------
if uploaded_file:
    OUTPUT_DIR.mkdir(exist_ok=True)

    pdf_path = OUTPUT_DIR / "temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # -----------------------------
    # DOCLING CONFIG
    # -----------------------------
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.generate_picture_images = True
    opts.images_scale = 2.0

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    st.info("Running Docling extraction…")
    result = converter.convert(str(pdf_path))
    doc = result.document

    extracted_blocks = []

    # -----------------------------
    # TEXT + TABLE EXTRACTION
    # -----------------------------
    for item, _ in doc.iterate_items():
        if hasattr(item, "text") and item.text:
            if item.label in {
                DocItemLabel.TITLE,
                DocItemLabel.SECTION_HEADER,
                DocItemLabel.PARAGRAPH,
                DocItemLabel.LIST_ITEM,
                DocItemLabel.CAPTION,
            }:
                extracted_blocks.append(item.text.strip())

        if item.label == DocItemLabel.TABLE:
            try:
                table_md = item.export_to_markdown(doc=doc)
                extracted_blocks.append(
                    f"<table>\n{table_md}\n</table>"
                )
            except:
                pass

    # -----------------------------
    # IMAGE OCR EXTRACTION
    # -----------------------------
    st.info("🖼️ Extracting text from images with OCR…")
    pdf = fitz.open(str(pdf_path))
    img_counter = 0

    for page_idx in range(len(pdf)):
        page = pdf[page_idx]
        for img in page.get_images(full=True):
            xref = img[0]
            base = pdf.extract_image(xref)
            image_bytes = base["image"]

            # Run OCR on image bytes
            ocr_text = ocr_image_bytes(image_bytes)
            
            # Skip images with no text or very little text
            if len(ocr_text.strip()) < 5:
                continue

            img_counter += 1
            
            # Optionally save image for debugging
            img_path = IMAGES_DIR / f"image_{img_counter}.png"
            Image.open(io.BytesIO(image_bytes)).save(img_path)

            extracted_blocks.append(
                f"""<imagedesc>
Page {page_idx + 1}, Image {img_counter}
{ocr_text.strip()}
</imagedesc>"""
            )

    pdf.close()

    # -----------------------------
    # SHOW RAW EXTRACTED CONTENT
    # -----------------------------
    extracted_text = "\n\n".join(extracted_blocks)

    st.subheader(" Extracted Content (Before NuMarkdown)")
    st.text_area(
        "Raw extracted content",
        extracted_text,
        height=400
    )

    # -----------------------------
    # RUN NUMARKDOWN RECONSTRUCTION
    # -----------------------------
    if st.button("Generate Markdown with NuMarkdown-Thinking-8B"):
        st.info("Running NuMarkdown reconstruction…")
        
        # Check input length (NuMarkdown has limited context window)
        input_length = len(extracted_text)
        st.caption(f"Input length: {input_length:,} characters")
        
        # Warn if input is very large
        MAX_CHARS = 100000  # Conservative limit
        if input_length > MAX_CHARS:
            st.warning(f"⚠️ Input is very large ({input_length:,} chars). Truncating to {MAX_CHARS:,} chars to prevent crashes.")
            truncated_text = extracted_text[:MAX_CHARS] + "\n\n[... content truncated due to length ...]"
        else:
            truncated_text = extracted_text
        
        # Try NuMarkdown first, fallback to llama3.3 if it fails
        try:
            with st.spinner("Running NuMarkdown-Thinking-8B..."):
                response = ollama.chat(
                    model="maternion/NuMarkdown-Thinking:8b",
                    messages=[
                        {
                            "role": "user",
                            "content": NU_MARKDOWN_PROMPT + "\n\n" + truncated_text
                        }
                    ],
                    options={
                        "num_ctx": 8192,  # Context window size
                    }
                )
                markdown_output = response["message"]["content"]
                st.success("✅ NuMarkdown reconstruction completed!")
                
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ NuMarkdown failed: {error_msg[:200]}")
            st.warning("🔄 Falling back to llama3.3...")
            
            try:
                response = ollama.chat(
                    model="llama3.3",
                    messages=[
                        {
                            "role": "user",
                            "content": NU_MARKDOWN_PROMPT + "\n\n" + truncated_text
                        }
                    ]
                )
                markdown_output = response["message"]["content"]
                st.info("✅ Fallback model (llama3.3) completed successfully")
            except Exception as e2:
                st.error(f"Both models failed. Error: {e2}")
                markdown_output = truncated_text  # Last resort fallback

        st.subheader("📄 Reconstructed Markdown")
        st.text_area(
            "Final Markdown Output",
            markdown_output,
            height=600
        )
