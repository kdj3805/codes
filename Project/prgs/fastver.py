import streamlit as st
import tempfile
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
from PIL import Image
import cv2
import numpy as np

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling_core.types.doc import TableItem, PictureItem, TextItem

from rapidocr_onnxruntime import RapidOCR


# =====================================================
# OCR ENGINE
# =====================================================
ocr_engine = RapidOCR()


# =====================================================
# DOCLING CONVERTER
# =====================================================
def create_converter():
    ocr_opts = EasyOcrOptions(
        force_full_page_ocr=False,
        use_gpu=False
    )

    pipeline = PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=False,                 # IMPORTANT
        generate_picture_images=True,
        ocr_options=ocr_opts
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline
            )
        }
    )


# =====================================================
# IMAGE TEXT LIKELIHOOD CHECK
# =====================================================
def image_likely_contains_text(img: Image.Image) -> bool:
    img_np = np.array(img.convert("L"))
    edges = cv2.Canny(img_np, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # UI screenshots usually large + edge dense
    return (
        img_np.shape[0] > 400 and
        img_np.shape[1] > 400 and
        edge_density > 0.01
    )


# =====================================================
# IMAGE OCR WITH FALLBACK
# =====================================================
def extract_image_text_with_detection(pdf_path: str, page_no: int) -> List[str]:
    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]
    outputs = []

    for img in page.get_images(full=True):
        xref = img[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_pil = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
            )

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img_pil.save(tmp.name)

            result, _ = ocr_engine(tmp.name)
            Path(tmp.name).unlink(missing_ok=True)

            if result:
                lines = [r[1] for r in result if r[1].strip()]
                if lines:
                    outputs.append("\n".join(lines))
                    continue

            # OCR FAILED → CHECK IF TEXT LIKELY PRESENT
            if image_likely_contains_text(img_pil):
                outputs.append(
                    "Textual image detected (UI/screenshot), but OCR could not "
                    "reliably extract text. Image likely contains form fields, "
                    "labels, or configuration options."
                )

            pix = None

        except Exception:
            continue

    doc.close()
    return outputs


# =====================================================
# MAIN PROCESSOR
# =====================================================
def process_pdf(pdf_bytes: bytes, max_pages: int | None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        pdf_path = f.name

    try:
        converter = create_converter()
        result = converter.convert(pdf_path)
        doc = result.document

        structured = []
        raw_text = []

        current_page = 0

        for item, _ in doc.iterate_items():

            if hasattr(item, "prov") and item.prov:
                page_no = item.prov[0].page_no
                if page_no != current_page:
                    current_page = page_no
                    if max_pages and current_page > max_pages:
                        break

                    structured.append(f"\n\n## Page {current_page}\n")
                    raw_text.append(f"\n\n=== PAGE {current_page} ===\n")

                    image_texts = extract_image_text_with_detection(
                        pdf_path, current_page
                    )
                    for txt in image_texts:
                        structured.append(
                            f"\n<imagedesc>\n{txt}\n</imagedesc>\n"
                        )
                        raw_text.append(txt)

            if isinstance(item, TableItem):
                md = item.export_to_markdown(doc=doc)
                structured.append(f"\n<tableinfo>\n{md}\n</tableinfo>\n")
                raw_text.append(md)

            elif isinstance(item, TextItem):
                if item.text.strip():
                    structured.append(item.text)
                    raw_text.append(item.text)

        return {
            "structured": "\n\n".join(structured),
            "raw_text": "\n\n".join(raw_text)
        }

    finally:
        Path(pdf_path).unlink(missing_ok=True)


# =====================================================
# STREAMLIT UI
# =====================================================
def main():
    st.set_page_config("Multimodal RAG Ingestion", layout="wide")
    st.title("Multimodal RAG PDF Ingestion")

    max_pages = st.number_input(
        "Limit pages (0 = all)", min_value=0, value=0
    )

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and st.button("Process Document", type="primary"):
        with st.spinner("Processing…"):
            result = process_pdf(
                uploaded.getvalue(),
                None if max_pages == 0 else max_pages
            )

        st.success("Extraction complete")

        st.subheader("Unified RAG-Ready Output")
        st.markdown(result["structured"], unsafe_allow_html=True)

        with st.expander("Raw Text (for embeddings)"):
            st.text_area("Raw Text", result["raw_text"], height=400)


if __name__ == "__main__":
    main()
