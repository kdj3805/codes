import streamlit as st
import warnings

# Remove docling warnings
warnings.filterwarnings("ignore", message=".*langchain_docling.*")

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders.parsers import TesseractBlobParser

# Specific imports with safety checks
try:
    from langchain_pymupdf4llm import PyMuPDF4LLMLoader
except ImportError:
    PyMuPDF4LLMLoader = None

try:
    from langchain_docling.loader import DoclingLoader
except ImportError:
    DoclingLoader = None

# --- CONFIGURATION ---
PDF_FILE = "D:\\trial\\data\\Disaster_Management_Unit_1.pdf"  # Change this to your PDF path

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="PDF Loader Studio")
st.title("PDF Loaders")

# Display the current PDF file being processed
st.info(f" Processing: {PDF_FILE}")

# --- LOADER SELECTION ---
st.sidebar.header("Loader Configuration")

loader_option = st.sidebar.radio(
    "Select Loader Strategy",
    [
        "PyPDFLoader",
        "PDFPlumberLoader",
        "PyMuPDFLoader",
        "PyPDFium2Loader",
        "PyMuPDF4LLMLoader",
        "PDFMinerLoader",
        "DoclingLoader"
    ]
)

# --- EXECUTION ---
st.divider()
st.header(f"Output: {loader_option}")

documents = []

try:
    # Initializing loaders
    if loader_option == "PyPDFLoader":
        loader = PyPDFLoader(PDF_FILE,mode="page",images_inner_format="markdown-img",images_parser=TesseractBlobParser())
        
    elif loader_option == "PDFPlumberLoader":
        loader = PDFPlumberLoader(PDF_FILE)
        
    elif loader_option == "PyMuPDFLoader":
        loader = PyMuPDFLoader(PDF_FILE, mode="page", images_inner_format="markdown-img", images_parser=RapidOCRBlobParser(), extract_tables="markdown")
        
    elif loader_option == "PyPDFium2Loader":
        loader = PyPDFium2Loader(PDF_FILE, mode="page", images_inner_format="markdown-img", images_parser=RapidOCRBlobParser())
        
    elif loader_option == "PyMuPDF4LLMLoader":
        if PyMuPDF4LLMLoader:
            loader = PyMuPDF4LLMLoader(PDF_FILE, mode="page", extract_images=True, images_parser=RapidOCRBlobParser(), table_strategy="lines_strict")
        else:
            st.error("Library 'langchain_pymupdf4llm' not found.")
            loader = None

    elif loader_option == "PDFMinerLoader":
        loader = PDFMinerLoader(PDF_FILE, mode="page", images_inner_format="markdown-img", images_parser=RapidOCRBlobParser())
        
    elif loader_option == "DoclingLoader":
        if DoclingLoader:
            loader = DoclingLoader(file_path=PDF_FILE)
        else:
            st.error("Library 'langchain_docling' not found.")
            loader = None
    
    # Load content
    if 'loader' in locals() and loader:
        documents = loader.load()

        if documents:
            st.success(f" Successfully loaded {len(documents)} units.")
            st.markdown("---")
            
            # --- DISPLAY OUTPUT ---
            for i, doc in enumerate(documents):
                # Header for the page/unit
                st.subheader(f" Page {i+1}")
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Markdown", "Raw Text"])
                
                with tab1:
                    st.markdown(doc.page_content)
                
                with tab2:
                    st.text(doc.page_content)
                
                # Metadata - Small footprint
                with st.expander("Show Metadata"):
                    st.json(doc.metadata)
                
                st.divider()
        else:
            st.warning(" Loader returned no documents.")

except Exception as e:
    st.error(f" Error during loading: {e}")