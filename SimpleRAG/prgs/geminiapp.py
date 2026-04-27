import streamlit as st
import os
import tempfile
from geminiingest2 import ingest_documents
from geminiretrieve1 import HybridRetriever

# st.set_page_config(page_title="Enterprise Hybrid RAG", layout="wide")
st.title("RAG Pipeline")

# ── Session state ──────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = (
        HybridRetriever() if os.path.exists("./chroma_db") else None
    )

# =============================================================================
# SIDEBAR — INGESTION
# =============================================================================
with st.sidebar:
    st.header("1. Ingest Documents")

    # Goal 2: accept multiple PDFs at once
    pdf_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process & Embed PDFs"):
        if pdf_files:
            with st.spinner(
                f"Extracting {len(pdf_files)} file(s) with Vision, Chunking, and Embedding…"
            ):
                pdf_infos: list[dict] = []
                tmp_paths: list[str] = []

                for uploaded_file in pdf_files:
                    pdf_bytes = uploaded_file.getvalue()

                    if len(pdf_bytes) < 100:
                        st.error(
                            f"'{uploaded_file.name}' appears empty or corrupt — skipping."
                        )
                        continue

                    # Write to a named temp file so fitz can open it by path
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf", prefix="upload_"
                    )
                    tmp.write(pdf_bytes)
                    tmp.flush()
                    tmp.close()

                    tmp_paths.append(tmp.name)
                    # Goal 2: preserve the original filename for source metadata
                    pdf_infos.append({"path": tmp.name, "name": uploaded_file.name})

                if pdf_infos:
                    try:
                        num_chunks = ingest_documents(pdf_infos)
                        st.session_state.retriever = HybridRetriever()
                        st.success(
                            f"✅ Embedded {num_chunks} semantic chunks "
                            f"from {len(pdf_infos)} file(s)!"
                        )
                    finally:
                        # Clean up every temp file regardless of outcome
                        for path in tmp_paths:
                            if os.path.exists(path):
                                os.remove(path)
        else:
            st.error("Please upload at least one PDF first.")

# =============================================================================
# MAIN BODY — RETRIEVAL & CHAT
# =============================================================================
st.header("2. Query the Knowledge Base")

if st.session_state.retriever is None:
    st.info("👈 Please upload and process a document in the sidebar to begin.")
else:
    query = st.text_input("Ask a question about your documents:")

    if st.button("Generate Answer"):
        if query:
            with st.spinner("Retrieving (BM25 + Vector + RRF) and synthesizing answer…"):
                fused_docs = st.session_state.retriever.retrieve(query)
                answer = st.session_state.retriever.generate_answer(query, fused_docs)

            st.markdown("### Answer:")
            st.write(answer)

            # Goal 3: show source + page in the expander
            with st.expander("View Retrieved Sources (RRF Fused)"):
                for i, doc in enumerate(fused_docs):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(f"**Source {i+1} — {source}, Page {page}:**")
                    st.markdown(doc.page_content)
                    st.divider()
        else:
            st.warning("Please enter a question.")