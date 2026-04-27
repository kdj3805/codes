
import streamlit as st
from pathlib import Path
from cancer_retrieval import generate_answer

st.set_page_config(
    page_title = "Medical RAG Assistant",
    page_icon  = "🧬",
    layout     = "wide",
)

# ==============================================================================
# SIDEBAR — PATIENT REPORT
# ==============================================================================
with st.sidebar:
    st.header("📄 Patient Report")
    st.markdown(
        "Upload or paste a medical report. "
        "The AI will use it to personalise answers."
    )

    uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])
    pasted_report = st.text_area("Or paste report text here:", height=200)

    patient_context = ""
    if uploaded_file is not None:
        patient_context = uploaded_file.getvalue().decode("utf-8")
    elif pasted_report.strip():
        patient_context = pasted_report.strip()

    if patient_context:
        st.success("✅ Patient report loaded!")
        with st.expander("View Loaded Report"):
            st.write(patient_context)

    st.divider()
    st.caption("Hybrid + MMR retrieval  •  BAAI/bge-base-en-v1.5  •  Groq llama-3.3-70b")

# ==============================================================================
# MAIN CHAT INTERFACE
# ==============================================================================
st.title("🧬 Personalized Medical RAG Assistant")
st.markdown(
    "Ask anything about your report or oncology. "
    "Backed by clinical research with **Hybrid + MMR** retrieval and **web search fallback**."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":    "assistant",
            "content": "Hello! Upload your medical report on the left, "
                       "then ask me any questions you have.",
        }
    ]

# Render existing chat history (text only — figures are shown live, not stored)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==============================================================================
# USER INPUT + RESPONSE
# ==============================================================================
if user_query := st.chat_input(
    "E.g., What does 'BRAF V600E positive' mean in my report?"
):
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Searching medical literature..."):
            try:
                # generate_answer returns 3 values:
                #   answer  (str)        — text answer (may include web section)
                #   sources (list[dict]) — [{"label": name, "url": link_or_empty}]
                #   figures (list[dict]) — [{"caption":..., "image_path":..., "source":...}]
                answer, sources, figures = generate_answer(
                    query          = user_query,
                    patient_report = patient_context,
                )

                # ── Main answer ───────────────────────────────────
                st.markdown(answer)

                # ── Figures ───────────────────────────────────────
                # Rendered when query is about a figure OR when caption
                # chunks surfaced through the MMR selection.
                if figures:
                    st.markdown("---")
                    st.markdown(f"**🖼️ Relevant Figures ({len(figures)})**")
                    for i in range(0, len(figures), 2):
                        cols = st.columns(2)
                        for j, col in enumerate(cols):
                            if i + j < len(figures):
                                fig = figures[i + j]
                                img_path = fig.get("image_path", "")
                                caption  = fig.get("caption", "")
                                source   = fig.get("source", "")
                                if img_path and Path(img_path).exists():
                                    with col:
                                        st.image(
                                            img_path,
                                            caption = f"{caption[:120]}...\n[{source}]",
                                            use_container_width = True,
                                        )

                # ── Sources ───────────────────────────────────────
                # RAG sources  → show PDF filename (no URL)
                # Web sources  → show as clickable hyperlink
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for s in sources:
                            label = s.get("label", "")
                            url   = s.get("url", "")
                            if url:
                                # Web search result — render as clickable link
                                st.markdown(f"- 🌐 [{label}]({url})")
                            else:
                                # RAG result — PDF filename, no hyperlink
                                st.markdown(f"- 📄 `{label}`")

                # Store only the text answer in chat history
                # (figures and sources are rendered live, not persisted)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                err_msg = f"❌ Error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg}
                )