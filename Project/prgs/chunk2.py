import streamlit as st
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling_core.types.doc import TableItem

from rapidocr_onnxruntime import RapidOCR

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# =========================
# OCR ENGINE (IN-MEMORY)
# =========================
ocr_engine = RapidOCR()


# =========================
# DOCLING CONFIG
# =========================
def create_converter():
    ocr_opts = EasyOcrOptions(
        force_full_page_ocr=False,
        use_gpu=False
    )

    pipeline_opts = PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=False,
        generate_picture_images=True,
        ocr_options=ocr_opts
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )


# =========================
# IMAGE OCR (ONLY IF IMAGE EXISTS)
# =========================
def extract_image_text(pdf_path: str, page_no: int):
    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]

    texts = []

    for img in page.get_images(full=True):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)

        if pix.n < 5:
            mode = "RGB" if pix.n == 3 else "L"
            img_pil = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img_pil.save(tmp.name)
                result, _ = ocr_engine(tmp.name)

            if result:
                text = "\n".join([line[1] for line in result])
                if text.strip():
                    texts.append(text.strip())

        pix = None

    doc.close()
    return texts


# =========================
# ONE-TIME PDF INGESTION
# =========================
def process_pdf_once(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        pdf_path = f.name

    try:
        converter = create_converter()
        result = converter.convert(pdf_path)
        doc = result.document

        unified_text = []
        structured_text = []

        current_page = 0

        for item, _ in doc.iterate_items():
            if hasattr(item, "prov") and item.prov:
                page_no = item.prov[0].page_no
                if page_no != current_page:
                    current_page = page_no
                    unified_text.append(f"\n\n=== PAGE {current_page} ===\n")
                    structured_text.append(f"\n\n## Page {current_page}\n")

                    image_texts = extract_image_text(pdf_path, current_page)
                    for t in image_texts:
                        structured_text.append(f"\n<imagedesc>\n{t}\n</imagedesc>\n")
                        unified_text.append(t)

            if isinstance(item, TableItem):
                try:
                    md = item.export_to_markdown(doc=doc)
                    structured_text.append(f"\n<tableinfo>\n{md}\n</tableinfo>\n")
                    unified_text.append(md)
                except:
                    pass
            else:
                try:
                    if hasattr(item, "export_to_markdown"):
                        text = item.export_to_markdown()
                    else:
                        text = getattr(item, "text", "")

                    if text.strip():
                        unified_text.append(text)
                        structured_text.append(text)
                except:
                    pass

        return {
            "raw_text": "\n".join(unified_text),
            "structured_text": "\n".join(structured_text)
        }

    finally:
        Path(pdf_path).unlink(missing_ok=True)


# =========================
# CHUNKING
# =========================
def chunk_text(text: str, strategy: str):
    if strategy == "Recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    elif strategy == "Character":
        splitter = CharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
    elif strategy == "Token":
        splitter = TokenTextSplitter(
            chunk_size=1000,
            chunk_overlap=64
        )
    else:
        raise ValueError("Unknown strategy")

    return splitter.split_text(text)


# =========================
# STREAMLIT UI
# =========================
def main():
    st.set_page_config("Multimodal RAG Ingestion", layout="wide")
    st.title("📄 Multimodal RAG Ingestion + Chunking Lab")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and "doc_data" not in st.session_state:
        with st.spinner("Processing PDF (one-time extraction)..."):
            st.session_state.doc_data = process_pdf_once(uploaded.getvalue())
        st.success("PDF loaded once. Ready for chunking experiments.")

    if "doc_data" in st.session_state:
        raw_text = st.session_state.doc_data["raw_text"]

        strategy = st.selectbox(
            "Choose Chunking Strategy",
            ["Recursive", "Token", "Character"]
        )

        chunks = chunk_text(raw_text, strategy)

        st.metric("Total Characters", len(raw_text))
        st.metric("Total Chunks", len(chunks))

        st.divider()

        st.subheader("All Chunks (Expandable)")
        for i, c in enumerate(chunks):
            with st.expander(f"Chunk {i + 1} ({len(c)} chars)"):
                st.code(c)

    else:
        st.info("Upload a PDF to begin")


if __name__ == "__main__":
    main()
