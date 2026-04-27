"""
app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enterprise Multi-Document RAG — Streamlit Interface
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two-tab layout:

  Tab 1 — 📥 Ingest Documents
    • Upload one or more PDFs
    • Shows whether each file is new or already ingested
    • Force re-ingest checkbox to replace existing chunks
    • Per-file status badge after ingestion

  Tab 2 — 🔍 Query Documents
    • Free-text question input
    • Optional: filter search to specific ingested files
    • Displays retrieved chunks in collapsible cards showing:
        - Source file name
        - Page number
        - Section heading
        - All retrieval scores (Dense, BM25, RRF, Boost, Final)
        - Full chunk content
    • Displays final LLM-generated answer with source attribution

Sidebar shows live Knowledge Base stats (total chunks, per-type, per-file).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import tempfile
from typing import List, Optional

import streamlit as st

from ingest import (
    ingest_pdf,
    list_ingested_files,
    delete_file_chunks,
    get_collection_stats,
)
from retrieve import (
    hybrid_retrieve,
    generate_answer,
    get_all_headings,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise Multi-Doc RAG",
    page_icon="🏢",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "ingest_log"      not in st.session_state:
    st.session_state.ingest_log: List[dict] = []
if "last_retrieved"  not in st.session_state:
    st.session_state.last_retrieved: list = []
if "last_answer"     not in st.session_state:
    st.session_state.last_answer: str = ""
if "last_query"      not in st.session_state:
    st.session_state.last_query: str = ""

# ─────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_ICONS = {"section": "📄", "table": "📊", "image": "🖼️"}

def type_icon(chunk_type: str) -> str:
    return _TYPE_ICONS.get(chunk_type, "📄")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏢 Enterprise Multi-Document RAG Pipeline")
st.caption(
    "Upload multiple PDFs → persistent ChromaDB vector store → "
    "Hybrid retrieval (Dense + BM25 + RRF) → Groq LLM answers"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — KNOWLEDGE BASE STATUS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📚 Knowledge Base")

    stats = get_collection_stats()
    st.metric("Total Chunks", stats["total_chunks"])

    # Per-type breakdown
    if stats["type_counts"]:
        type_cols = st.columns(len(stats["type_counts"]))
        for col, (ct, count) in zip(type_cols, stats["type_counts"].items()):
            col.metric(f"{type_icon(ct)} {ct}", count)

    st.divider()
    st.subheader("Ingested Files")

    ingested = list_ingested_files()
    if not ingested:
        st.info("No documents ingested yet.")
    else:
        for fname in ingested:
            n_chunks = stats["file_counts"].get(fname, 0)
            col_name, col_del = st.columns([4, 1])
            col_name.markdown(f"📄 **{fname}**  \n`{n_chunks} chunks`")
            if col_del.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                deleted = delete_file_chunks(fname)
                st.success(f"Deleted {deleted} chunks for `{fname}`")
                st.rerun()

    st.divider()
    st.caption(f"DB path: `./chroma_store`")
    if st.button("🔄 Refresh Stats"):
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

tab_ingest, tab_query = st.tabs(["📥 Ingest Documents", "🔍 Query Documents"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — INGESTION
# ═════════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.subheader("Upload & Ingest PDF Documents")
    st.markdown(
        "Upload one or more PDFs. Already-ingested files are **skipped automatically**. "
        "Enable **Force re-ingest** to replace existing chunks for a file."
    )

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDFs at once"
    )

    force_reingest = st.checkbox(
        "⚠️ Force re-ingest (replaces existing chunks for uploaded files)",
        value=False
    )

    # Preview what will happen before the user clicks
    if uploaded_files:
        already_in_db = list_ingested_files()
        st.markdown("**Files selected:**")
        for uf in uploaded_files:
            status = "✅ already ingested" if uf.name in already_in_db else "🆕 new"
            action = "will re-ingest" if (uf.name in already_in_db and force_reingest) else status
            st.write(f"  • `{uf.name}` — {action}")

    run_ingest = st.button(
        "▶️ Start Ingestion",
        type="primary",
        disabled=not uploaded_files
    )

    if run_ingest and uploaded_files:
        st.session_state.ingest_log = []
        outer_progress = st.progress(0.0, text="Starting…")
        log_area       = st.container()

        for i, uploaded_file in enumerate(uploaded_files):
            outer_progress.progress(
                i / len(uploaded_files),
                text=f"Processing `{uploaded_file.name}` ({i + 1}/{len(uploaded_files)})…"
            )

            # Write to a temp file (fitz needs a real path, not a BytesIO)
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                prefix="rag_tmp_"
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with log_area:
                with st.spinner(f"Ingesting `{uploaded_file.name}`…"):
                    try:
                        result = ingest_pdf(
                            pdf_path     = tmp_path,
                            source_name  = uploaded_file.name,   # ← use original name
                            force_reingest = force_reingest
                        )
                    except Exception as exc:
                        result = {
                            "source_file":  uploaded_file.name,
                            "status":       "error",
                            "chunks_added": 0,
                            "error":        str(exc)
                        }
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                st.session_state.ingest_log.append(result)

                if result["status"] == "ingested":
                    st.success(
                        f"✅ `{result['source_file']}` — "
                        f"**{result['chunks_added']} chunks** ingested successfully"
                    )
                elif result["status"] == "skipped":
                    st.info(
                        f"⏭️ `{result['source_file']}` — already in the knowledge base. "
                        f"Enable **Force re-ingest** to replace."
                    )
                elif result["status"] == "empty":
                    st.warning(
                        f"⚠️ `{result['source_file']}` — no extractable content found."
                    )
                else:
                    st.error(
                        f"❌ `{result['source_file']}` — "
                        f"error: {result.get('error', 'unknown error')}"
                    )

        outer_progress.progress(1.0, text="Done!")
        st.balloons()
        st.rerun()   # refresh sidebar stats

    # Session ingestion summary
    if st.session_state.ingest_log:
        st.divider()
        st.subheader("Session Ingestion Log")
        _status_icons = {
            "ingested": "✅", "skipped": "⏭️",
            "empty":    "⚠️", "error":   "❌"
        }
        for entry in st.session_state.ingest_log:
            icon = _status_icons.get(entry["status"], "❓")
            st.write(
                f"{icon} `{entry['source_file']}` — "
                f"**{entry['status']}** — {entry['chunks_added']} chunks"
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUERY
# ═════════════════════════════════════════════════════════════════════════════

with tab_query:
    st.subheader("Ask Questions Across Your Documents")

    # Refresh the file list for the query tab
    ingested_files = list_ingested_files()

    if not ingested_files:
        st.warning(
            "No documents have been ingested yet. "
            "Go to the **📥 Ingest Documents** tab to add PDFs."
        )
        st.stop()

    # ── Query Controls ────────────────────────────────────────────────────

    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Your question:",
            placeholder=(
                "e.g. What passcode types are supported?  "
                "/ What are the roles and responsibilities?"
            ),
            key="query_input"
        )
    with col_k:
        top_k = st.number_input(
            "Top-K results", min_value=1, max_value=20, value=5, step=1
        )

    # Optional source-file filter
    with st.expander("🔎 Filter by source document (optional — default: search all)"):
        selected_files: List[str] = st.multiselect(
            "Restrict retrieval to these files:",
            options=ingested_files,
            default=[],
            help="Leave empty to search across all ingested documents"
        )

    source_filter: Optional[List[str]] = selected_files if selected_files else None

    search_btn = st.button("🔍 Search & Answer", type="primary", disabled=not query)

    # ── Run Retrieval ─────────────────────────────────────────────────────

    if search_btn and query:
        st.session_state.last_query     = query
        st.session_state.last_retrieved = []
        st.session_state.last_answer    = ""

        with st.spinner("Embedding query and retrieving chunks…"):
            retrieved = hybrid_retrieve(
                query         = query,
                top_k         = int(top_k),
                source_filter = source_filter
            )
            st.session_state.last_retrieved = retrieved

        if not retrieved:
            st.warning(
                "No relevant chunks found. "
                "Try a different query, or check that documents are ingested."
            )
        else:
            with st.spinner("Generating answer…"):
                all_headings = get_all_headings(source_filter=source_filter)
                answer       = generate_answer(query, retrieved, all_headings)
                st.session_state.last_answer = answer

    # ── Display Retrieved Chunks ──────────────────────────────────────────

    if st.session_state.last_retrieved:
        retrieved = st.session_state.last_retrieved
        st.divider()
        st.subheader("📊 Retrieved Chunks")
        st.caption(
            f"Query: *{st.session_state.last_query}*  •  "
            f"{len(retrieved)} result(s)"
        )

        for rank, r in enumerate(retrieved, start=1):
            icon        = type_icon(r.get("chunk_type", "section"))
            source      = r.get("source_file", "unknown")
            page        = r.get("page", "?")
            heading     = r.get("heading",  "N/A")
            chunk_type  = r.get("chunk_type", "section")
            final_score = r.get("final_score", 0)

            expander_label = (
                f"#{rank}  {icon} {heading}  "
                f"│  📄 {source}  │  Page {page}  │  Score: {final_score}"
            )

            # Expand the top result by default
            with st.expander(expander_label, expanded=(rank == 1)):

                # ── Source attribution ──────────────────────────────────
                a1, a2, a3 = st.columns(3)
                a1.markdown(f"**📄 Source File**\n\n`{source}`")
                a2.markdown(f"**📖 Page**\n\n{page}")
                a3.markdown(f"**🏷️ Section Heading**\n\n{heading}")

                st.divider()

                # ── Score breakdown ─────────────────────────────────────
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Dense",       r.get("dense_score",   0))
                s2.metric("BM25",        r.get("bm25_score",    0))
                s3.metric("RRF Hybrid",  r.get("hybrid_score",  0))
                s4.metric("Boost",       r.get("heading_boost", 0))
                s5.metric("Final",       r.get("final_score",   0))

                st.divider()

                # ── Chunk content ───────────────────────────────────────
                st.text_area(
                    "Chunk Content",
                    r.get("content", ""),
                    height=240,
                    key=f"chunk_content_{rank}_{r.get('chunk_id', rank)}"
                )

    # ── Final Answer ──────────────────────────────────────────────────────

    if st.session_state.last_answer:
        st.divider()
        st.subheader("🧠 Final Answer")

        if st.session_state.last_retrieved:
            sources_used = sorted({
                c.get("source_file", "unknown")
                for c in st.session_state.last_retrieved
            })
            st.caption(
                "Answer synthesised from: "
                + ", ".join(f"`{s}`" for s in sources_used)
            )

        st.success(st.session_state.last_answer)

        with st.expander("📋 Copy plain text"):
            st.code(st.session_state.last_answer, language=None)