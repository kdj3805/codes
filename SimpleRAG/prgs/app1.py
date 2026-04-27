"""
app.py — Clean Production UI
"""

import json
import os
import tempfile
from typing import List, Optional

import streamlit as st

from ingest1 import (
    ingest_pdf,
    list_ingested_files,
    delete_file_chunks,
    get_collection_stats,
)
from retrieve1 import (
    hybrid_retrieve,
    generate_answer,
    get_all_headings,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise Multi-Doc RAG",
    page_icon="🏢",
    layout="wide"
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "ingest_log" not in st.session_state:
    st.session_state.ingest_log: List[dict] = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved: list = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer: str = ""
if "last_query" not in st.session_state:
    st.session_state.last_query: str = ""

# ─────────────────────────────────────────────
# BADGES
# ─────────────────────────────────────────────

_TYPE_ICONS = {"section": "📄", "table_row": "🗂️"}

_SEVERITY_COLORS = {
    "critical": "#dc3545",
    "high": "#fd7e14",
    "medium": "#ffc107",
    "low": "#28a745",
}

def severity_badge(severity: str):
    if not severity:
        return ""
    color = _SEVERITY_COLORS.get(severity.lower(), "#6c757d")
    return f"""
    <span style="
        background:{color};
        color:white;
        padding:4px 10px;
        border-radius:4px;
        font-weight:bold;
        font-size:0.85em;">
        {severity.upper()}
    </span>
    """

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🏢 Enterprise Multi-Document RAG")
st.caption("Upload PDFs → Retrieve → Multi-chunk answer synthesis")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("📚 Knowledge Base")

    stats = get_collection_stats()
    st.metric("Total Chunks", stats["total_chunks"])

    st.divider()
    st.subheader("Ingested Files")

    ingested = list_ingested_files()
    if not ingested:
        st.info("No documents ingested yet.")
    else:
        for fname in ingested:
            n_chunks = stats["file_counts"].get(fname, 0)
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 **{fname}**  \n`{n_chunks} chunks`")
            if col2.button("🗑️", key=f"del_{fname}"):
                deleted = delete_file_chunks(fname)
                st.success(f"Deleted {deleted} chunks")
                st.rerun()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab_ingest, tab_query = st.tabs(["📥 Ingest", "🔍 Query"])

# ─────────────────────────────────────────────
# INGEST TAB
# ─────────────────────────────────────────────

with tab_ingest:
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    force_reingest = st.checkbox("Force re-ingest", value=False)

    if st.button("Start Ingestion", disabled=not uploaded_files):

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            result = ingest_pdf(
                pdf_path=tmp_path,
                source_name=uploaded_file.name,
                force_reingest=force_reingest
            )

            os.unlink(tmp_path)

            if result["status"] == "ingested":
                st.success(f"{uploaded_file.name} → {result['chunks_added']} chunks")
            else:
                st.warning(result["status"])

        st.rerun()

# ─────────────────────────────────────────────
# QUERY TAB
# ─────────────────────────────────────────────

with tab_query:

    ingested_files = list_ingested_files()

    if not ingested_files:
        st.warning("Ingest documents first.")
        st.stop()

    query = st.text_input("Your question")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=8)

    search_btn = st.button("Search", disabled=not query)

    if search_btn:
        st.session_state.last_query = query
        st.session_state.last_retrieved = []
        st.session_state.last_answer = ""

        retrieved = hybrid_retrieve(
            query=query,
            top_k=int(top_k),
            source_filter=None
        )

        st.session_state.last_retrieved = retrieved

        if retrieved:
            answer = generate_answer(query, retrieved, [])
            st.session_state.last_answer = answer

    # ─────────────────────────────────────────
    # DISPLAY RETRIEVED
    # ─────────────────────────────────────────

    if st.session_state.last_retrieved:

        st.subheader("Retrieved Chunks")

        for i, r in enumerate(st.session_state.last_retrieved, 1):

            chunk_type = r.get("chunk_type", "section")
            severity = r.get("severity", "")
            source = r.get("source_file", "")
            page = r.get("page", "")

            label = f"#{i} {_TYPE_ICONS.get(chunk_type,'📄')} {source} — Page {page}"

            if severity:
                label += f" — {severity.upper()}"

            with st.expander(label, expanded=(i == 1)):

                if severity:
                    st.markdown(
                        severity_badge(severity),
                        unsafe_allow_html=True
                    )

                # SAFE JSON PARSING
                raw_row_data = r.get("raw_row_data", "")

                parsed_json = {}
                if raw_row_data:
                    try:
                        parsed_json = json.loads(raw_row_data)
                    except:
                        parsed_json = {}

                if parsed_json:
                    st.markdown("### Structured Data")

                    for key, value in parsed_json.items():
                        if isinstance(value, list):
                            st.markdown(f"**{key}:**")
                            for item in value:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(f"**{key}:** {value}")

                    with st.expander("Raw JSON"):
                        st.code(json.dumps(parsed_json, indent=2))

                st.markdown("### Content")
                st.text_area(
                    "",
                    r.get("content", ""),
                    height=200,
                    key=f"content_{i}"
                )

    # ─────────────────────────────────────────
    # FINAL ANSWER
    # ─────────────────────────────────────────

    if st.session_state.last_answer:

        st.divider()
        st.subheader("Final Answer")

        # Show which severities contributed
        severities_used = sorted({
            r.get("severity", "").lower()
            for r in st.session_state.last_retrieved
            if r.get("severity")
        })

        if severities_used:
            st.markdown("### Severities Used")
            badges = " ".join(
                severity_badge(sev) for sev in severities_used
            )
            st.markdown(badges, unsafe_allow_html=True)

        st.success(st.session_state.last_answer)

        with st.expander("Copy"):
            st.code(st.session_state.last_answer)
