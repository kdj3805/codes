import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_docling.loader import DoclingLoader
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders.parsers import TesseractBlobParser
import pytesseract # Ensure pytesseract is imported for OCR parsing

st.title("PDF Loader PyMuPDF Example")

#File_path = "D:\\trial\\data\\GRE Math Formulas.pdf"
File_path = "D:\\trial\\data\\Disaster_Management_Unit_1.pdf"
folder_path = "D:\\trial\\data\\"

#loader = PyPDFLoader(File_path,mode="page",images_inner_format="markdown-img",images_parser=TesseractBlobParser())
#loader = PDFPlumberLoader(File_path)
# #loader = PyMuPDFLoader(File_path,mode="page",extract_tables="markdown",images_inner_format="markdown-img",images_parser=TesseractBlobParser())
#loader = PyPDFDirectoryLoader(File_path)
loader = PyPDFium2Loader(File_path,mode="page",images_inner_format="markdown-img",images_parser=TesseractBlobParser())
#loader = PyMuPDF4LLMLoader(File_path)
#loader = PDFMinerLoader(File_path)
#loader = DoclingLoader(file_path = File_path,export_type="markdown")


documents = loader.load()

if documents:
    for i, doc in enumerate(documents):
        st.subheader(f"Page {i+1}")
        st.write(doc.page_content)
        st.divider()

    st.subheader("Metadata")
    st.json(documents[0].metadata)
else:
    st.warning("No documents loaded")
