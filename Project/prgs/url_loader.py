import streamlit as st
from langchain_docling.loader import DoclingLoader

st.title("PPT Loader Example")

File_path = "https://docs.langchain.com/oss/python/integrations/document_loaders/docling"

loader = DoclingLoader(File_path,export_type="markdown")

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
