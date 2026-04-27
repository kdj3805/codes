import streamlit as st
from langchain_community.document_loaders import UnstructuredPowerPointLoader

st.title("PPT Loader Example")

#File_path = "D:\\trial\\data\\GRE Math Formulas.pdf"
File_path = "D:\\trial\\data\\Deadlock_Unit_4_OS.pptx"
folder_path = "D:\\trial\\data\\"

loader = UnstructuredPowerPointLoader(File_path,mode="paged")

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
