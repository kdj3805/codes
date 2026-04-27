import io
import re
import hashlib
from pathlib import Path

import streamlit as st
import fitz

# ===================== GRAPH RAG IMPORTS =====================
from graphrag_integration import generate_answer_graphrag_stream as generate_answer_stream

# ===================== CONFIG =====================

st.set_page_config(
    page_title="MedChat - Cancer RAG Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar starts collapsed for new sessions
)
# ===================== CUSTOM UI THEME =====================

st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */
.stApp {
    background-color: #eef2f7;
}

/* ===== MAIN TITLE ===== */
h1 {
    font-weight: 700;
}

/* ===== CUSTOM TITLE COLORS ===== */
.medchat-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0;
}

.medchat-dark {
    color: #1f2a44;   /* navy */
}

.medchat-green {
    color: #2fa36b;   /* soft green */
}

/* ===== SUBTEXT ===== */
.medchat-subtext {
    color: #5f6c7b;
    font-size: 16px;
    margin-top: 5px;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e6ed;
}

/* ===== CHAT INPUT CONTAINER (REMOVE DOUBLE BORDER) ===== */
div[data-testid="stChatInput"] {
    border: none !important;
    box-shadow: none !important;
}

/* ===== CHAT INPUT TEXTAREA ===== */
div[data-testid="stChatInput"] textarea {
    background-color: #f4fbf7 !important;
    border: 1px solid #d0d7de !important;
    border-radius: 12px !important;
    color: #1f2a44 !important;
    padding: 10px !important;
    box-shadow: none !important;
}

/* ===== CHAT INPUT FOCUS (ONLY RED BORDER) ===== */
div[data-testid="stChatInput"] textarea:focus {
    border: 1px solid #e53935 !important;   /* RED */
    box-shadow: none !important;            /* REMOVE OUTER GLOW */
    outline: none !important;
}

/* ===== BUTTONS ===== */
.stButton>button {
    border-radius: 10px;
    background-color: #2fa36b;
    color: white;
    border: none;
}

.stButton>button:hover {
    background-color: #248a59;
}

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
    font-weight: 600;
}

/* ===== CHAT MESSAGE BOX ===== */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 10px;
}

/* ===== TEXT AREA (UPLOAD TAB) ===== */
textarea {
    background-color: #f4fbf7 !important;
    border: 1px solid #cfe8dc !important;
    border-radius: 12px !important;
    color: #1f2a44 !important;
    padding: 10px !important;
    box-shadow: none !important;
}

/* ===== TEXTAREA FOCUS (ONLY RED BORDER) ===== */
textarea:focus {
    border: 1px solid #e53935 !important;   /* RED */
    box-shadow: none !important;            /* REMOVE DOUBLE BORDER */
    outline: none !important;
}

/* ===== PLACEHOLDER ===== */
textarea::placeholder {
    color: #7a8a9a !important;
}

/* ===== SCROLLBAR (optional premium feel) ===== */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #c5d1dc;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


BASE_DIR  = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "output" / "images"

CANCER_TYPE_OPTIONS = [
    "All",
    "breast",
    "lung",
    "melanoma",
    "leukemia",
    "osteosarcoma",
    "skin",
]

_ANALYSIS_PROMPT = (
    "A patient has just uploaded their clinical report. "
    "Please analyse it carefully and provide a structured summary covering:\n"
    "1. **Diagnosis** — cancer type, subtype, any biomarkers or receptor status mentioned.\n"
    "2. **Stage** — TNM classification or overall stage if stated.\n"
    "3. **Treatment Plan** — therapies listed (surgery, chemo, radiation, targeted, immunotherapy).\n"
    "4. **Key Clinical Findings** — any notable lab values, imaging results, or pathology notes.\n"
    "5. **Relevant Literature** — based on the diagnosis and stage, what does the peer-reviewed "
    "clinical context say about prognosis, standard of care, or survival outcomes for this patient? "
    "Include any relevant figures or tables using [IMAGE: filename.png] if available.\n\n"
    "Be empathetic, clear, and avoid unnecessary jargon. "
    "End with a reminder to consult a qualified oncologist for personalised decisions."
)

# ===================== HELPER FUNCTIONS =====================

def _build_analysis_search_query(report_text: str) -> str:
    """
    Extract key medical entities from a patient report and build a concise,
    semantically-rich query for vector DB retrieval.

    Instead of sending the massive _ANALYSIS_PROMPT to the embedding model
    (which drowns out medical terms like 'Osteosarcoma' with boilerplate),
    we extract cancer type, drugs, side effects, and stage to create a
    focused search query that BGE embeddings can match accurately.
    """
    from entity_extractor import extract_entities

    entities = extract_entities("", report_text)

    parts: list[str] = []

    # Cancer type(s) — highest priority for retrieval
    if entities.cancers:
        parts.extend(entities.cancers)

    # Extract stage from the report text (regex catches "Stage IIB", "Stage IV", etc.)
    stage_match = re.search(r'[Ss]tage\s+([IVX]+[A-C]?|\d[A-C]?)', report_text)
    if stage_match:
        parts.append(f"Stage {stage_match.group(1)}")

    # Chemotherapy / targeted therapy drugs
    if entities.drugs:
        parts.extend(entities.drugs)
    if entities.ncdrugs:
        parts.extend(entities.ncdrugs)

    # Side effects mentioned in the report (cap at 5 to avoid query explosion)
    if entities.side_effects:
        parts.extend(entities.side_effects[:5])

    # Add general clinical analysis terms for broader retrieval
    parts.extend(["treatment", "prognosis", "survival", "clinical outcomes"])

    query = " ".join(parts)
    print(f"\n\U0001f50d [ANALYSIS SEARCH QUERY] Built from report: {query}")
    return query


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text().strip()
            if page_text:
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        doc.close()
    except Exception as e:
        return f"[PDF extraction error: {e}]"
    return "\n\n".join(text_parts)


def load_report_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    raw_bytes = uploaded_file.getvalue()
    name      = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(raw_bytes)
    else:
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="replace")


def _report_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def render_message_with_images(text: str):
    clean = text
    for wrap in ["**", "*", "`"]:
        clean = clean.replace(f"{wrap}[", "[").replace(f"]{wrap}", "]")

    parts = re.split(r'\[IMAGE:\s*([^\]]+)\]', clean, flags=re.IGNORECASE)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            filename = part.strip().strip('`"\'')
            img_path = IMAGE_DIR / filename

            if img_path.exists():
                st.markdown(f"**Reference Visual:** `{filename}`")
                st.image(
                    str(img_path),
                    caption=f"{filename}",
                    use_container_width=True,
                )

def render_followup_buttons(followups: list[str], turn_key: str):
    if not followups:
        return

    st.markdown("**Suggested follow-ups:**")
    cols = st.columns(len(followups))
    for i, (col, question) in enumerate(zip(cols, followups)):
        with col:
            key = f"fup_{turn_key}_{i}_{abs(hash(question)) % 999983}"
            if st.button(f"Q.{question}", key=key, use_container_width=True):
                st.session_state["triggered_followup"] = question


def render_streaming_answer(
    query: str,
    patient_report: str,
    chat_history: list[dict],
    cancer_filter: str,
    is_analysis: bool = False,
) -> tuple[str, list, list]:
    """
    Runs the streaming generator, displays tokens in real-time, then
    replaces the raw stream output with the clean post-processed answer
    stored in stream_buffer (citations stripped, Sources block appended).

    FIX 2: Displays a web-fallback warning banner when the answer came
            entirely from web search (both internal contexts were empty).
    FIX 3: Always re-renders from stream_buffer so the final output is
            citation-free and has the Sources block at the bottom.
    """
    stream_container = st.empty()

    # ── Stream tokens in real-time ───────────────────────────────
    with stream_container.container():
        st.write_stream(
            generate_answer_stream(
                query=query,
                patient_report=patient_report,
                chat_history=chat_history,
                cancer_filter=cancer_filter,
                is_analysis=is_analysis,
            )
        )

    # ── Read post-processed state set by the generator ───────────
    full_answer = st.session_state.get("stream_buffer", "")
    sources     = st.session_state.get("stream_sources", [])
    followups   = st.session_state.get("stream_followups", [])
    is_fallback = st.session_state.get("is_web_fallback", False)

    # ── FIX 2: Show web fallback warning banner ──────────────────
    if is_fallback:
        st.warning(
            "⚠️ **Web Search Fallback:** This answer was generated using web search "
            "and is **NOT** from the internal clinical knowledge base. "
            "Please verify with your healthcare provider."
        )
        # Reset flag so it doesn't persist to the next query
        st.session_state["is_web_fallback"] = False

    # ── FIX 3: Always re-render clean final answer ───────────────
    # This replaces the raw streamed output (which may still have [1][2])
    # with the post-processed version (citations stripped, Sources appended).
    stream_container.empty()
    with stream_container.container():
        render_message_with_images(full_answer)

    return full_answer, sources, followups


def _run_auto_analysis(patient_context: str, cancer_filter: str):
    st.session_state["analysed_report"] = _report_hash(patient_context)

    st.session_state.messages.append({
        "role": "user",
        "content": "Patient report loaded - please analyse it.",
    })

    with st.chat_message("user"):
        st.markdown("**Patient report loaded - please analyse it.**")

    with st.chat_message("assistant"):
        try:
            # Build a clean retrieval query from the report's medical entities
            # instead of using the boilerplate _ANALYSIS_PROMPT for vector search
            search_query = _build_analysis_search_query(patient_context)

            # Embed analysis instructions into the patient report so the LLM
            # receives them as generation guidance, while the vector DB only
            # sees the clean medical-entity query for accurate similarity search
            enriched_report = (
                f"{_ANALYSIS_PROMPT}\n\n"
                f"--- PATIENT CLINICAL REPORT ---\n\n"
                f"{patient_context}"
            )

            full_answer, sources, followups = render_streaming_answer(
                query=search_query,
                patient_report=enriched_report,
                chat_history=[],
                cancer_filter=cancer_filter,
                is_analysis=True,
            )

            if sources:
                with st.expander("Sources used in analysis"):
                    for s in sources:
                        if isinstance(s, dict) and s.get("url"):
                            st.markdown(f"[{s['label']}]({s['url']})")
                        elif isinstance(s, dict):
                            st.markdown(f"`{s.get('label', str(s))}`")
                        else:
                            st.markdown(f"`{s}`")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
            })

            turn_idx = len(st.session_state.messages) - 1
            st.session_state["followups"][turn_idx] = followups

        except Exception as e:
            st.error(f"Analysis error: {e}")
            st.exception(e)

    turn_idx = len(st.session_state.messages) - 1
    saved_fups = st.session_state["followups"].get(turn_idx, [])
    render_followup_buttons(saved_fups, turn_key=f"analysis_{turn_idx}")


def _init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Hello! I can answer questions about cancer - treatment, "
                "staging, diagnosis, survival rates and pull figures, "
                "flowcharts and tables directly from peer-reviewed literature.\n\n"
                "**Tip:** Upload or paste a patient report in the Upload Report tab and I will "
                "automatically analyse it against the clinical literature."
            ),
        }]
    if "analysed_report" not in st.session_state:
        st.session_state["analysed_report"] = ""
    if "followups" not in st.session_state:
        st.session_state["followups"] = {}
    if "triggered_followup" not in st.session_state:
        st.session_state["triggered_followup"] = ""
    if "stream_buffer" not in st.session_state:
        st.session_state["stream_buffer"] = ""
    if "stream_sources" not in st.session_state:
        st.session_state["stream_sources"] = []
    if "stream_followups" not in st.session_state:
        st.session_state["stream_followups"] = []
    # FIX 2: Flag set by the stream generator when web fallback is triggered
    if "is_web_fallback" not in st.session_state:
        st.session_state["is_web_fallback"] = False


# ===================== INIT =====================
_init_session_state()

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("Settings")

    st.subheader("Cancer Type Filter")
    cancer_filter_raw = st.selectbox(
        "Filter by cancer type",
        options=CANCER_TYPE_OPTIONS,
        index=0,
        help="Narrow retrieval to one cancer type. 'All' searches across all types.",
    )
    cancer_filter = "" if cancer_filter_raw == "All" else cancer_filter_raw
    if cancer_filter:
        st.info(f"Active filter: **{cancer_filter}** cancer")

    st.divider()
    st.caption("MedChat v3 — Tier 3 UX + Clinical Workflow")


# ===================== TABS & HEADER =====================

st.markdown("""
<div class="medchat-title">
    <span class="medchat-dark">MedChat – </span>
    <span class="medchat-green">Healthcare RAG Assistant</span>
</div>
<div class="medchat-subtext">
AI-powered system to help cancer patients understand clinical reports, treatments, and guidance using medical literature.
</div>
""", unsafe_allow_html=True)
st.caption(
    "Upload a patient report for instant analysis, or ask clinical questions directly. "
    "Figures, flowcharts and tables from peer-reviewed literature appear inline."
)

tab_chat, tab_upload = st.tabs(["Chat", "Upload Report"])

# ===================== UPLOAD TAB =====================

with tab_upload:
    st.subheader("Upload Patient Report")

    uploaded_file = st.file_uploader(
        "Upload report (.txt or .pdf)",
        type=["txt", "pdf"]
    )

    pasted_report = st.text_area(
        "Or paste report text here:",
        height=200
    )

    patient_context = ""
    upload_source   = ""

    if uploaded_file is not None:
        patient_context = load_report_from_upload(uploaded_file)
        upload_source   = uploaded_file.name
    elif pasted_report.strip():
        patient_context = pasted_report.strip()
        upload_source   = "pasted text"

    if patient_context:
        st.success(f"Report loaded ({upload_source})")
        with st.expander("Preview loaded report"):
            preview = patient_context[:800]
            if len(patient_context) > 800:
                preview += f"\n\n... ({len(patient_context) - 800} more characters)"
            st.text(preview)


# ===================== CHAT TAB =====================

with tab_chat:

    # 1. Read and clear any triggered follow-ups before rendering inputs
    triggered_followup = st.session_state.get("triggered_followup", "")
    if triggered_followup:
        st.session_state["triggered_followup"] = ""

    # 2. Setup structural container
    chat_container = st.container()

    # 3. Setup chat input (automatically pinned to bottom)
    typed_query = st.chat_input("E.g., What topical drug delivery systems are used in breast cancer?")
    user_query = triggered_followup or typed_query

    # 4. Render content inside the chat container
    with chat_container:

        # --- RENDER HISTORY ---
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                render_message_with_images(message["content"])

        last_turn_idx  = len(st.session_state.messages) - 1
        last_followups = st.session_state["followups"].get(last_turn_idx, [])

        # --- AUTO ANALYSIS ---
        if patient_context:
            current_hash = _report_hash(patient_context)
            if current_hash != st.session_state.get("analysed_report", ""):
                _run_auto_analysis(patient_context, cancer_filter)

        # --- SHOW EXISTING FOLLOW-UPS (If idle) ---
        if not user_query and last_followups:
            render_followup_buttons(last_followups, turn_key=str(last_turn_idx))

        # --- HANDLE NEW QUERY ---
        if user_query:

            with st.chat_message("user"):
                st.markdown(user_query)

            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                try:
                    history_for_llm = st.session_state.messages[:-1]

                    # Permanently stream responses
                    full_answer, sources, followups = render_streaming_answer(
                        query=user_query,
                        patient_report=patient_context,
                        chat_history=history_for_llm,
                        cancer_filter=cancer_filter,
                    )

                    # Sources expander (shows clickable links)
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                if isinstance(s, dict) and s.get("url"):
                                    st.markdown(f"[{s['label']}]({s['url']})")
                                elif isinstance(s, dict):
                                    st.markdown(f"`{s.get('label', str(s))}`")
                                else:
                                    st.markdown(f"`{s}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                    })

                    new_turn_idx = len(st.session_state.messages) - 1
                    st.session_state["followups"][new_turn_idx] = followups

                    if len(st.session_state["followups"]) > 10:
                        oldest = min(st.session_state["followups"].keys())
                        del st.session_state["followups"][oldest]

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)
                    followups    = []
                    new_turn_idx = len(st.session_state.messages) - 1

            # Render follow-ups immediately for the new answer
            render_followup_buttons(
                st.session_state["followups"].get(new_turn_idx, []),
                turn_key=str(new_turn_idx),
            )

            # Rerun only if it was typed, so the input box clears and history aligns
            if typed_query:
                st.rerun()