import streamlit as st
from langchain_community.document_loaders import CSVLoader
import os

# --- CONFIGURATION ---
FILE_PATH = "D:\\trial\\data\\warehouse_messy_data.csv"

st.set_page_config(layout="wide")
st.title("CSV Data Loader")

# --- CHECK FILE EXISTENCE ---
if not os.path.exists(FILE_PATH):
    st.error(f"File not found at: {FILE_PATH}")
    st.stop()

# --- LLM VIEW EXECUTION ---
st.subheader(f"LLM Document View: {os.path.basename(FILE_PATH)}")

try:
    # Initialize Loader
    loader = CSVLoader(
        file_path=FILE_PATH,
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
        }
    )
    documents = loader.load()

    st.success(f"Generated {len(documents)} Document Objects")
    st.divider()

    # Display Loop (Raw Content)
    for i, doc in enumerate(documents):
        st.markdown(f"**Record {i+1}**")
        
        # 1. The Content (What the LLM reads)
        st.text(doc.page_content)
        
        # 2. The Metadata (Source info)
        st.caption("Metadata:")
        st.json(doc.metadata)
        
        # Separator
        st.divider()

except Exception as e:
    st.error(f"Error loading Documents: {e}")