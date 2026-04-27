# ============================================================
# END-TO-END RAG PIPELINE: VISION EXTRACTION + SEMANTIC CHUNKING
# Uses PyMuPDF + Groq Vision + LangChain
# ============================================================

import os
import json
import base64
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# =========================
# ENV & INIT
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# MUST be a Vision model to avoid the 400 Array/String error
GROQ_VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct" 

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set. Please check your .env file.")

# =========================
# SYSTEM PROMPT
# =========================
VISION_PROMPT = """You are an expert data extraction assistant.
I am providing you with an image of a document page.

Your task:
1. Extract ALL the text and tables from this page accurately.
2. Format the output entirely in clean, structural Markdown.
3. For tables: Reconstruct them perfectly. If a cell has multiple lines or bullet points, combine them into a single line within the cell using `<br>` or spaces so the Markdown table formatting does not break.
4. Do not include any conversational filler (e.g., "Here is the extracted text"). Just return the Markdown.
"""

# =========================
# STEP 1: VISION EXTRACTION
# =========================
def extract_page_with_vision(b64_image: str, client: Groq) -> str:
    """Passes the base64 image to Groq Vision for extraction."""
    try:
        res = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Error extracting page: {str(e)}"

def process_pdf_vision(pdf_path: str) -> list:
    """Renders PDF pages to images and extracts Markdown via Vision LLM."""
    doc = fitz.open(pdf_path)
    client = Groq(api_key=GROQ_API_KEY)
    document_data = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        
        # Render page to image
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        b64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        # Extract via Vision
        extracted_md = extract_page_with_vision(b64_image, client)
        
        document_data.append({
            "page_number": page_idx + 1,
            "content": extracted_md
        })

    doc.close()
    return document_data

# =========================
# STEP 2: LANGCHAIN CHUNKING
# =========================
def chunk_extracted_data(document_data: list) -> list:
    """Chunks the extracted Markdown semantically while preserving tables and metadata."""
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 1. Split semantically by Markdown headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False 
    )

    # 2. THE FIX: Markdown-aware recursive splitting with larger chunk sizes
    recursive_splitter = RecursiveCharacterTextSplitter.from_language(
        language="markdown",
        chunk_size=2500,     # Increased from 1000 to keep tables intact
        chunk_overlap=250    # Increased overlap to maintain context
    )

    final_chunks = []

    for page in document_data:
        page_num = page["page_number"]
        content = page["content"]
        
        # Split semantically by Markdown headers
        md_header_splits = markdown_splitter.split_text(content)
        
        for split in md_header_splits:
            # Inject precise page metadata
            split.metadata["page"] = page_num
            
            # Fallback split for massive sections (respects token limits & Markdown bounds)
            smaller_chunks = recursive_splitter.split_documents([split])
            final_chunks.extend(smaller_chunks)

    return final_chunks

# =========================
# STREAMLIT UI
# =========================
st.set_page_config("End-to-End RAG Pipeline", layout="wide")
st.title("🚀 End-to-End RAG Extraction & Chunking")
st.write("1. Renders PDF $\\rightarrow$ 2. Groq Vision Markdown $\\rightarrow$ 3. LangChain Semantic Chunking")

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    with st.spinner("Extracting pages via Groq Vision... (This may take a moment)"):
        # Step 1: Extraction
        extracted_data = process_pdf_vision("temp.pdf")
        
    with st.spinner("Chunking data semantically via LangChain..."):
        # Step 2: Chunking
        langchain_chunks = chunk_extracted_data(extracted_data)

    st.success("Pipeline executed successfully!")

    # Display Metrics
    c1, c2 = st.columns(2)
    c1.metric("Pages Processed", len(extracted_data))
    c2.metric("RAG Chunks Generated", len(langchain_chunks))

    # Prepare export data
    json_output = json.dumps(extracted_data, indent=2)
    
    # Prepare chunks for display/download
    chunks_export = []
    for i, chunk in enumerate(langchain_chunks):
        chunks_export.append({
            "chunk_id": i + 1,
            "metadata": chunk.metadata,
            "content": chunk.page_content
        })
    chunks_json_output = json.dumps(chunks_export, indent=2)

    # Download Buttons
    st.download_button(
        label="Download Extracted Document (JSON)",
        data=json_output,
        file_name="vision_extracted_document.json",
        mime="application/json"
    )
    st.download_button(
        label="Download Final RAG Chunks (JSON)",
        data=chunks_json_output,
        file_name="rag_ready_chunks.json",
        mime="application/json"
    )

    st.subheader("Pipeline Output Preview")
    
    tab1, tab2 = st.tabs(["🧩 RAG Chunks (Final Output)", "📄 Raw Page Extractions"])
    
    with tab1:
        st.write("These chunks are ready to be embedded into a vector database.")
        for chunk in chunks_export:
            with st.expander(f"Chunk {chunk['chunk_id']} | Page: {chunk['metadata'].get('page', 'N/A')} | Headers: {chunk['metadata']}"):
                st.markdown(chunk['content'])
                
    with tab2:
        st.write("The raw Markdown generated by the Vision model before chunking.")
        for page in extracted_data:
            with st.expander(f"Page {page['page_number']}"):
                st.markdown(page['content'])

    os.remove("temp.pdf")