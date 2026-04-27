import streamlit as st
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    PictureDescriptionApiOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import ImageRefMode



OLLAMA_URL = "http://localhost:11434"
VLM_MODEL = "qwen3-vl:4b"

PAGE_BREAK = "\n\n--- PAGE BREAK ---\n\n"
IMAGE_DESC_START = "\n\n### Image Description\n"
IMAGE_DESC_END = "\n"




def create_picture_description_options() -> PictureDescriptionApiOptions:
    return PictureDescriptionApiOptions(
        api_base=OLLAMA_URL,
        model=VLM_MODEL,
        prompt=(
            "Describe the image clearly and factually. "
            "If the image contains UI elements, tables, diagrams, workflows, "
            "or screenshots, explain what is shown and its purpose."
        ),
        max_completion_tokens=256,
        timeout=90,
        seed=42,
    )


def create_pdf_pipeline_options() -> PdfPipelineOptions:
    return PdfPipelineOptions(
        enable_remote_services=True,
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=True,
        do_picture_descriptions=True,
        table_structure_options=TableStructureOptions(
            mode="accurate"
        ),
        picture_description_options=create_picture_description_options(),
    )



@st.cache_data(show_spinner=True)
def process_pdf(file_bytes: bytes) -> str:
    temp_path = Path("temp_uploaded.pdf")
    temp_path.write_bytes(file_bytes)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=create_pdf_pipeline_options(),
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    result = converter.convert(str(temp_path))
    doc = result.document

    markdown = doc.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
        image_placeholder="",
        page_break_placeholder=PAGE_BREAK,
        include_annotations=True,
        mark_annotations=True,
    )

    # Make image descriptions human-readable
    markdown = markdown.replace(
        '<!--<annotation kind="description">-->',
        IMAGE_DESC_START
    )
    markdown = markdown.replace(
        '<!--</annotation>-->',
        IMAGE_DESC_END
    )

    return markdown



st.set_page_config(layout="wide")
st.title("Multimodal PDF Ingestion (Tables + Image Descriptions)")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file:
    markdown_output = process_pdf(uploaded_file.getvalue())

    pages = markdown_output.split(PAGE_BREAK)

    st.sidebar.header("Navigation")
    page_num = st.sidebar.number_input(
        "Select page",
        min_value=1,
        max_value=len(pages),
        value=1,
        step=1,
    )

    st.subheader(f"Page {page_num}")
    st.markdown(pages[page_num - 1])
