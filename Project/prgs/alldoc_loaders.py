import streamlit as st
import tempfile
import time
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="PDF Loader Explorer")

st.title("PDF Loader Explorer (Full Output)")
st.markdown("""
**Educational Tool:** Compare the raw extraction outputs of different Python PDF libraries.
**Note:** This mode displays ALL extracted units. Large documents may take longer to render.
""")

# --- HELPER: DEPENDENCY CHECK ---
def get_loader_class(import_path, class_name):
    try:
        module = __import__(import_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None

# --- 1. FILE MANAGEMENT ---
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None

uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

if uploaded_file:
    if st.session_state.temp_file_path is None or st.session_state.temp_file_path != f"temp_{uploaded_file.name}":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.temp_file_path = tmp_file.name
            st.success(f"File successfully uploaded: {uploaded_file.name}")

    file_path = st.session_state.temp_file_path

    # --- 2. LIBRARY SELECTION ---
    st.sidebar.header("Configuration")
    
    loader_type = st.sidebar.radio(
        "Select Parsing Library:",
        [
            "PyPDF (pypdf)",
            "PDFPlumber (pdfplumber)",
            "PyMuPDF (pymupdf)",
            "PyMuPDF4LLM (pymupdf4llm)",
            "PyPDFium2 (pypdfium2)",
            "PDFMiner (pdfminer.six)",
            "Unstructured (unstructured)",
            "Docling (docling)"
        ]
    )

    # --- 3. EXECUTION LOGIC ---
    st.divider()
    st.subheader(f"Analysis: {loader_type}")
    
    start_time = time.time()
    raw_output = []
    error_message = None

    try:
        # A. PyPDF
        if "PyPDF (" in loader_type:
            LoaderClass = get_loader_class("langchain_community.document_loaders", "PyPDFLoader")
            if LoaderClass:
                loader = LoaderClass(file_path)
                raw_output = loader.load()
            else:
                error_message = "Library 'pypdf' not found."

        # B. PDFPlumber
        elif "PDFPlumber" in loader_type:
            LoaderClass = get_loader_class("langchain_community.document_loaders", "PDFPlumberLoader")
            if LoaderClass:
                loader = LoaderClass(file_path)
                raw_output = loader.load()
            else:
                error_message = "Library 'pdfplumber' not found."

        # C. PyMuPDF
        elif "PyMuPDF (" in loader_type:
            LoaderClass = get_loader_class("langchain_community.document_loaders", "PyMuPDFLoader")
            if LoaderClass:
                loader = LoaderClass(file_path)
                raw_output = loader.load()
            else:
                error_message = "Library 'pymupdf' not found."

        # D. PyMuPDF4LLM
        elif "PyMuPDF4LLM" in loader_type:
            try:
                import pymupdf4llm
                from langchain_core.documents import Document
                md_text = pymupdf4llm.to_markdown(file_path)
                raw_output = [Document(page_content=md_text, metadata={"source": file_path, "type": "markdown_full"})]
            except ImportError:
                error_message = "Library 'pymupdf4llm' not found."

        # E. PyPDFium2
        elif "PyPDFium2" in loader_type:
            LoaderClass = get_loader_class("langchain_community.document_loaders", "PyPDFium2Loader")
            if LoaderClass:
                loader = LoaderClass(file_path)
                raw_output = loader.load()
            else:
                error_message = "Library 'pypdfium2' not found."

        # F. PDFMiner
        elif "PDFMiner" in loader_type:
            LoaderClass = get_loader_class("langchain_community.document_loaders", "PDFMinerLoader")
            if LoaderClass:
                loader = LoaderClass(file_path)
                raw_output = loader.load()
            else:
                error_message = "Library 'pdfminer.six' not found."

        # G. Unstructured
        elif "Unstructured" in loader_type:
            LoaderClass = get_loader_class("langchain_community.document_loaders", "UnstructuredPDFLoader")
            if LoaderClass:
                loader = LoaderClass(file_path, mode="elements")
                raw_output = loader.load()
            else:
                error_message = "Library 'unstructured' not found."

        # H. Docling
        elif "Docling" in loader_type:
            try:
                from docling.document_converter import DocumentConverter
                from langchain_core.documents import Document
                converter = DocumentConverter()
                result = converter.convert(file_path)
                md_output = result.document.export_to_markdown()
                raw_output = [Document(page_content=md_output, metadata={"source": file_path, "type": "docling_markdown"})]
            except ImportError:
                error_message = "Library 'docling' not found."
            except Exception as e:
                error_message = f"Docling error: {e}"

    except Exception as e:
        error_message = f"Error: {e}"

    end_time = time.time()

    # --- 4. OUTPUT DISPLAY ---
    if error_message:
        st.error(error_message)
    else:
        duration = end_time - start_time
        st.success(f"Extraction Complete | Time: {duration:.4f}s | Output Units: {len(raw_output)}")

        view_tab1, view_tab2, view_tab3 = st.tabs(["Rendered View", "Raw Content (Text)", "Metadata"])

        # Loop through ALL output items without slicing/limits
        for i, doc in enumerate(raw_output):
            label = f"Unit {i+1}"
            
            with view_tab1:
                with st.expander(label, expanded=(i==0)):
                    st.markdown(doc.page_content)
            
            with view_tab2:
                with st.expander(label, expanded=(i==0)):
                    st.text(doc.page_content)
            
            with view_tab3:
                with st.expander(label, expanded=(i==0)):
                    st.json(doc.metadata)

else:
    st.info("Upload a PDF to begin.")