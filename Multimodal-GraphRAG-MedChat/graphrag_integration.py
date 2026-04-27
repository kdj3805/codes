from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Force absolute path to .env to prevent Streamlit from losing it
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

log = logging.getLogger(__name__)


#GROQ_MODEL = "llama-3.3-70b-versatile"
#GROQ_MODEL = "openai/gpt-oss-120b"  
GROQ_MODEL = "llama-3.1-8b-instant"   # temp experiment with instruct-tuned version for better adherence to answer guidelines
GROQ_TEMP  = 0.0

WEB_FALLBACK_MARKER = "<!-- WEB_FALLBACK_EMPTY_CONTEXT -->"


def _is_meaningful_context(ctx: str) -> bool:
    if not ctx:
        return False
    stripped = ctx.strip()
    # Must have at least 50 characters
    if len(stripped) < 50:
        return False
    # Must have at least 2 non-empty lines (header + one data line minimum)
    non_empty_lines = [l for l in stripped.splitlines() if l.strip()]
    return len(non_empty_lines) >= 2


def _strip_inline_citations(text: str) -> str:
    # Remove patterns: [1], [12], [W1], [W12]
    cleaned = re.sub(r'\[W?\d+\]', '', text)
    # Collapse multiple spaces into one
    cleaned = re.sub(r'  +', ' ', cleaned)
    # Remove space immediately before common punctuation
    cleaned = re.sub(r' ([,\.;:])', r'\1', cleaned)
    return cleaned.strip()


def _format_sources_block(sources: list) -> str:
    if not sources:
        return ""

    lines = ["\n\n---\n\n**Sources:**"]
    seen: set[str] = set()

    for s in sources:
        if isinstance(s, dict):
            label = s.get("label", "").strip()
            url   = s.get("url", "").strip()
            # Build a human-readable display name from the label
            display = label.replace("-", " ").replace("_", " ").title() if label else ""
            entry = f"[{display}]({url})" if (url and display) else (
                url if url else display
            )
        elif isinstance(s, str) and s.strip():
            entry = s.strip()
        else:
            continue

        if entry and entry not in seen:
            seen.add(entry)
            lines.append(f"- {entry}")

    # Only return the block if we actually added at least one source line
    return "\n".join(lines) if len(lines) > 1 else ""


# ══════════════════════════════════════════════════════════════════
# Section 1: Context fusion
# ══════════════════════════════════════════════════════════════════

def fuse_contexts(
    vector_context: str,
    graph_context:  str,
    max_total_chars: int = 12000,
) -> str:
    """
    Merge vector-retrieved context with graph-retrieved context.
    Safeguards the Visual Assets Database from being truncated.
    Called ONLY when the relevant context(s) have already been validated
    as meaningful by _is_meaningful_context() in generate_answer_graphrag().
    """
    # 1. Safely extract the Visual Assets Database so it NEVER gets truncated
    visual_assets = ""
    if "## Extracted Visual Assets Database" in vector_context:
        parts = vector_context.split("## Extracted Visual Assets Database")
        vector_context = parts[0]
        visual_assets = "\n\n## Extracted Visual Assets Database" + parts[1]

    # 2. Build the blocks
    graph_block  = "GRAPH KNOWLEDGE BASE (structured oncology knowledge):\n" + graph_context  if graph_context  else ""
    vector_block = "VECTOR RETRIEVAL CONTEXT (peer-reviewed literature):\n"  + vector_context if vector_context else ""
    separator    = "\n\n" + "─" * 60 + "\n\n"

    # 3. Calculate lengths and truncate ONLY the text part of the vector context
    combined_len = len(graph_block) + len(separator) + len(vector_block) + len(visual_assets)

    if combined_len > max_total_chars and vector_block:
        allowed_vector_len = max_total_chars - len(graph_block) - len(separator) - len(visual_assets)
        if allowed_vector_len > 200:
            vector_block = vector_block[:allowed_vector_len] + "\n...[literature truncated]"
        else:
            vector_block = ""

    # 4. Reassemble — only include non-empty blocks
    parts_to_join = [block for block in [graph_block, vector_block, visual_assets] if block]
    return separator.join(parts_to_join)


# ══════════════════════════════════════════════════════════════════
# Section 2: Modified generate_answer with GraphRAG
# ══════════════════════════════════════════════════════════════════

def generate_answer_graphrag(
    query:          str,
    patient_report: str        = "",
    chat_history:   list[dict] = None,
    cancer_filter:  str        = "",
    is_analysis:    bool       = False,
) -> tuple[str, list, list]:
    """
    Enhanced generate_answer that fuses:
      - Qdrant hybrid MMR retrieval (vector)
      - Neo4j GraphRAG retrieval (graph)
    Then sends fused context to Groq LLaMA 3.3 70B.

    FIX 1: Uses each context source only when it contains meaningful data.
    FIX 2: Both-empty → directly triggers web fallback, LLM is NOT called.
    FIX 3: Strips inline [1][2] citations; appends Sources block at end.
    """
    # ── LOCAL IMPORTS (Prevents Circular Import Error) ──
    from Cancer_retrieval_v2_visual import (
        get_hybrid_mmr_retriever,
        build_context,
        generate_followups,
        _web_search_fallback,
        _rag_has_no_answer,
        _extract_sources,          # FIX 3: use proper dict-returning extractor
    )

    if chat_history is None:
        chat_history = []

    try:
        log.info("[GraphRAG] Query: %s", query[:70])

        # ── Step 1: Retrieval query refinement ──────────────────
        retrieval_query = _build_retrieval_query(query, chat_history)

        # ── Step 2: Vector retrieval (Qdrant) ───────────────────
        retriever  = get_hybrid_mmr_retriever(cancer_filter=cancer_filter)
        retrieved  = retriever.invoke(retrieval_query)
        vector_ctx = build_context(retrieved, retrieval_query) if retrieved else ""
        sources    = _extract_sources(retrieved) if retrieved else []

        # ── Step 3: Graph retrieval (Neo4j) ─────────────────────
        try:
            from graph_retrieval import retrieve_graph_context
            graph_ctx = retrieve_graph_context(query, patient_report)
        except Exception as ge:
            log.warning("[GraphRAG] Graph retrieval failed: %s", ge)
            print(f"🚨 GRAPH RETRIEVAL FAILED: {ge}")
            graph_ctx = ""

        # ── Step 4: FIX 1 + FIX 2 ─ Evaluate context quality ───
        has_vector = _is_meaningful_context(vector_ctx)
        has_graph  = _is_meaningful_context(graph_ctx)

        # ── FIX 2: Both empty → DIRECTLY trigger web fallback ───
        if not has_vector and not has_graph:
            log.info("[GraphRAG] Both contexts empty → direct web fallback (LLM skipped)")
            if not is_analysis:
                fallback_ans, fallback_src, _ = _web_search_fallback(
                    "", query, patient_report, chat_history
                )
                # FIX 3: Strip citations from fallback answer + append sources
                fallback_ans = _strip_inline_citations(fallback_ans)
                fb_sources_block = _format_sources_block(fallback_src)
                if fb_sources_block:
                    fallback_ans += fb_sources_block
                # Prepend marker so UI can display the fallback warning
                return WEB_FALLBACK_MARKER + "\n\n" + fallback_ans, fallback_src, []
            return (
                "According to the provided clinical context, "
                "no relevant information was found for this query.",
                [], [],
            )

        # ── Step 5: FIX 1 ─ Selective context fusion ────────────
        if has_vector and has_graph:
            # Case 1: Both contexts available → full fusion
            fused_context = fuse_contexts(vector_ctx, graph_ctx)
        elif has_graph:
            # Case 2: Only graph context is meaningful → use graph only
            log.info("[GraphRAG] Using graph context only (vector context empty/insufficient)")
            fused_context = fuse_contexts("", graph_ctx)
        else:
            # Case 3: Only vector context is meaningful → use vector only
            log.info("[GraphRAG] Using vector context only (graph context empty/insufficient)")
            fused_context = fuse_contexts(vector_ctx, "")

        # ── Step 6: Build prompt ─────────────────────────────────
        system_message = (
            "You are an empathetic medical AI assistant helping "
            "cancer patients and clinicians understand medical information.\n"
            "You have access to both peer-reviewed literature AND a structured oncology knowledge graph.\n"
            "CRITICAL SYSTEM DIRECTIVE: You are connected to a frontend UI capable of rendering images. "
            "NEVER state that you are a text-based AI or cannot display images. "
            "When the context contains a relevant image reference, you MUST output it using the format `[IMAGE: filename.ext]`."
        )

        # Build graph-specific instruction only when graph context is actually used
        graph_instruction = ""
        if has_graph:
            graph_instruction = (
                "\n- HOW TO USE THE CONTEXT SECTIONS:\n"
                "  • VECTOR RETRIEVAL CONTEXT: Use this for clinical explanations, medical criteria, "
                "staging, prognosis, and any topic covered in peer-reviewed papers.\n"
                "  • GRAPH KNOWLEDGE BASE: Use this ONLY when the question is specifically about "
                "drug side effects, food recommendations, drug interactions, or treatment protocols. "
                "Do NOT pull graph data into answers about clinical criteria, diagnosis, or survival unless directly relevant."
            )

        current_user_msg = f"""PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

CLINICAL CONTEXT:
{fused_context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer the user's question thoroughly and completely. Explain the topic in depth — cover what things mean,
  how they work, and why they matter. A detailed, informative answer on the asked topic is always welcome.
- However, do NOT drift into UNRELATED topics that the user did not ask about.
  For example, if the user asks about the ABCDE criteria, explain each letter in detail — but do NOT
  also list chemotherapy side effects, food recommendations, or drug interactions unless the user asked.
- EXCEPTION: Relevant images are NEVER considered off-topic. ALWAYS include relevant images (see IMAGE RULE below).
- Answer using ONLY the clinical context provided above (both vector and graph sections).
- Provide a clear, structured, and patient-friendly explanation — avoid heavy jargon.

- IMPORTANT FORMAT RULES:
  • Mention "Graph Knowledge Base" ONLY ONCE in the entire answer.
  • Do NOT repeat "[Graph Knowledge Base]" after every point.
  • Do NOT mention "Vector Retrieval Context" at all.
  • Do NOT explain where the information came from internally.

- When the user SPECIFICALLY asks about side effects, food recommendations, or drug interactions:
  • Group them cleanly under each drug or symptom.
  • Use bullet points or numbering.
  • Keep it readable and natural (like a doctor explaining).
  • But do NOT include these unless the user asked for them.

- IMAGE RULE (IMPORTANT — images are NEVER considered "padding"):
  • The context may contain a "Visual Assets Database" section.
  • You MUST include any image from the Visual Assets Database that is relevant to the user's question, cancer type, or treatment. This is mandatory — do NOT skip relevant images.
  • Use the exact format [IMAGE: filename.png] to display images.
  • If an image is relevant but doesn't perfectly answer the exact question, display it anyway and explain what it represents.
  • Ignore completely unrelated images (e.g., showing a diagram for a different cancer).
  • If the user asks for a specific visual (like a survival graph) and none is available, simply state that no such visual is available in the database.

- Do NOT use inline citation numbers like [1] or [2]. Sources are listed separately at the end.{graph_instruction}
- CRITICAL — when the context does not contain the answer: You MUST respond with the exact phrase
  "I do not have enough information in the provided context to answer this question."
  Do NOT pivot to listing general information about related topics. Do NOT compensate with
  treatment protocols, side effects, or dietary advice if those were not what was asked.
- End with a short medical disclaimer advising consultation with a qualified doctor (1-2 lines only).
"""

        # ── Step 7: Build message history ───────────────────────
        groq_messages = [{"role": "system", "content": system_message}]
        groq_messages.extend(_format_history_for_groq(chat_history))
        groq_messages.append({"role": "user", "content": current_user_msg})

        # ── Step 8: Call Groq ────────────────────────────────────
        client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model       = GROQ_MODEL,
            temperature = GROQ_TEMP,
            messages    = groq_messages,
        )
        answer = response.choices[0].message.content or ""

        # ── Step 9: Fallback if LLM admits insufficient data ─────
        if _rag_has_no_answer(answer) and not is_analysis:
            log.info("[GraphRAG] RAG insufficient → web search fallback")
            # Pass empty string — we do NOT want the bad RAG answer prepended
            fallback_ans, fallback_src, _ = _web_search_fallback(
                "", query, patient_report, chat_history
            )
            # FIX 3: Strip citations from fallback answer + append sources
            fallback_ans = _strip_inline_citations(fallback_ans)
            fb_sources_block = _format_sources_block(fallback_src)
            if fb_sources_block:
                fallback_ans += fb_sources_block
            # Prepend marker so UI can display the fallback warning
            return WEB_FALLBACK_MARKER + "\n\n" + fallback_ans, fallback_src, []

        # ── Step 10: FIX 3 ─ Strip citations + append sources ───
        answer = _strip_inline_citations(answer)
        sources_block = _format_sources_block(sources)
        if sources_block:
            answer += sources_block

        # ── Step 11: Follow-ups ──────────────────────────────────
        followups = generate_followups(answer, fused_context, cancer_filter)

        return answer, sources, followups

    except Exception as e:
        log.exception("[GraphRAG] Error in generate_answer_graphrag")
        return f"Error: {str(e)}", [], []


def generate_answer_graphrag_stream(
    query:          str,
    patient_report: str        = "",
    chat_history:   list[dict] = None,
    cancer_filter:  str        = "",
    is_analysis:    bool       = False,
):
    """
    Streaming variant of generate_answer_graphrag.

    FIX 1: Uses each context source only when it contains meaningful data.
    FIX 2: Both-empty → directly triggers web fallback, LLM is NOT called.
    FIX 3: Post-processes full_answer after stream to strip citations + append Sources.
    """
    import streamlit as st
    from groq import Groq
    from Cancer_retrieval_v2_visual import (
        _web_search_fallback,
        _rag_has_no_answer,
        generate_followups,
        get_hybrid_mmr_retriever,
        build_context,
        _extract_sources,          # FIX 3: use proper dict-returning extractor
    )

    if chat_history is None:
        chat_history = []

    st.session_state["stream_buffer"]    = ""
    st.session_state["stream_sources"]   = []
    st.session_state["stream_followups"] = []
    # FIX 2: reset the fallback flag at the start of every new stream
    st.session_state["is_web_fallback"]  = False

    try:
        # ── Step 1: Retrieve contexts ────────────────────────────
        retrieval_query = _build_retrieval_query(query, chat_history)

        retriever  = get_hybrid_mmr_retriever(cancer_filter=cancer_filter)
        retrieved  = retriever.invoke(retrieval_query)
        vector_ctx = build_context(retrieved, retrieval_query) if retrieved else ""
        st.session_state["stream_sources"] = _extract_sources(retrieved) if retrieved else []

        try:
            from graph_retrieval import retrieve_graph_context
            graph_ctx = retrieve_graph_context(query, patient_report)
        except Exception as ge:
            print(f"🚨 GRAPH RETRIEVAL FAILED: {ge}")
            log.warning("[GraphRAG] Stream Graph retrieval failed: %s", ge)
            graph_ctx = ""

        # ── Step 2: FIX 1 + FIX 2 ─ Evaluate context quality ───
        has_vector = _is_meaningful_context(vector_ctx)
        has_graph  = _is_meaningful_context(graph_ctx)

        # ── FIX 2: Both empty → handle based on is_analysis flag ───
        if not has_vector and not has_graph:
            if is_analysis:
                # Knowledge-base-only mode for analysis — NEVER web fallback
                log.info("[GraphRAG-stream] Both contexts empty + is_analysis → KB-only message")
                msg = (
                    "The clinical knowledge base does not contain specific literature "
                    "matching this patient's profile at this time. The report has been "
                    "reviewed but no matching peer-reviewed data was found in the "
                    "ingested documents.\n\n"
                    "**Recommendation:** Please consult your oncologist for personalised "
                    "guidance based on the latest clinical evidence."
                )
                st.session_state["stream_buffer"] = msg
                yield msg
                return

            # Non-analysis path: trigger web fallback
            log.info("[GraphRAG-stream] Both contexts empty → direct web fallback (LLM skipped)")
            # Mark as fallback so UI shows the warning banner
            st.session_state["is_web_fallback"] = True

            yield "Local knowledge base missing data. Searching the clinical web...\n\n"

            fallback_ans, fallback_sources, _ = _web_search_fallback(
                "", query, patient_report, chat_history
            )

            # FIX 3: Strip citations + append sources block
            fallback_ans = _strip_inline_citations(fallback_ans)
            fb_sources_block = _format_sources_block(fallback_sources)
            if fb_sources_block:
                fallback_ans += fb_sources_block

            st.session_state["stream_buffer"]  = fallback_ans
            st.session_state["stream_sources"] = fallback_sources
            yield fallback_ans
            return

        # ── Step 3: FIX 1 ─ Selective context fusion ────────────
        if has_vector and has_graph:
            # Case 1: Both contexts available → full fusion
            fused_context = fuse_contexts(vector_ctx, graph_ctx)
        elif has_graph:
            # Case 2: Only graph context is meaningful
            log.info("[GraphRAG-stream] Using graph context only")
            fused_context = fuse_contexts("", graph_ctx)
        else:
            # Case 3: Only vector context is meaningful
            log.info("[GraphRAG-stream] Using vector context only")
            fused_context = fuse_contexts(vector_ctx, "")

        # ── Step 4: Build prompts ────────────────────────────────
        system_message = (
            "You are an empathetic medical AI assistant helping "
            "cancer patients and clinicians understand medical information.\n"
            "You have access to both peer-reviewed literature AND a structured oncology knowledge graph.\n"
            "CRITICAL SYSTEM DIRECTIVE: You are connected to a frontend UI capable of rendering images. "
            "NEVER state that you are a text-based AI or cannot display images. "
            "When the context contains a relevant image reference, you MUST output it using the format `[IMAGE: filename.ext]`."
        )

        # Build graph-specific instruction only when graph context is actually used
        graph_instruction = ""
        if has_graph:
            graph_instruction = (
                "\n- HOW TO USE THE CONTEXT SECTIONS:\n"
                "  • VECTOR RETRIEVAL CONTEXT: Use this for clinical explanations, medical criteria, "
                "staging, prognosis, and any topic covered in peer-reviewed papers.\n"
                "  • GRAPH KNOWLEDGE BASE: Use this ONLY when the question is specifically about "
                "drug side effects, food recommendations, drug interactions, or treatment protocols. "
                "Do NOT pull graph data into answers about clinical criteria, diagnosis, or survival unless directly relevant."
            )

        current_user_msg = f"""PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

CLINICAL CONTEXT:
{fused_context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer the user's question thoroughly and completely. Explain the topic in depth — cover what things mean,
  how they work, and why they matter. A detailed, informative answer on the asked topic is always welcome.
- However, do NOT drift into UNRELATED topics that the user did not ask about.
  For example, if the user asks about the ABCDE criteria, explain each letter in detail — but do NOT
  also list chemotherapy side effects, food recommendations, or drug interactions unless the user asked.
- EXCEPTION: Relevant images are NEVER considered off-topic. ALWAYS include relevant images (see IMAGE RULE below).
- Answer using ONLY the clinical context provided above (both vector and graph sections).
- Provide a clear, structured, and patient-friendly explanation — avoid heavy jargon.

- IMPORTANT FORMAT RULES:
  • Mention "Graph Knowledge Base" ONLY ONCE in the entire answer.
  • Do NOT repeat "[Graph Knowledge Base]" after every point.
  • Do NOT mention "Vector Retrieval Context" at all.
  • Do NOT explain where the information came from internally.

- When the user SPECIFICALLY asks about side effects, food recommendations, or drug interactions:
  • Group them cleanly under each drug or symptom.
  • Use bullet points or numbering.
  • Keep it readable and natural (like a doctor explaining).
  • But do NOT include these unless the user asked for them.

- IMAGE RULE (IMPORTANT — images are NEVER considered "padding"):
  • The context may contain a "Visual Assets Database" section.
  • You MUST include any image from the Visual Assets Database that is relevant to the user's question, cancer type, or treatment. This is mandatory — do NOT skip relevant images.
  • Use the exact format [IMAGE: filename.png] to display images.
  • If an image is relevant but doesn't perfectly answer the exact question, display it anyway and explain what it represents.
  • Ignore completely unrelated images (e.g., showing a diagram for a different cancer).
  • If the user asks for a specific visual (like a survival graph) and none is available, simply state that no such visual is available in the database.

- Do NOT use inline citation numbers like [1] or [2]. Sources are listed separately at the end.{graph_instruction}
- CRITICAL — when the context does not contain the answer: You MUST respond with the exact phrase
  "I do not have enough information in the provided context to answer this question."
  Do NOT pivot to listing general information about related topics. Do NOT compensate with
  treatment protocols, side effects, or dietary advice if those were not what was asked.
- End with a short medical disclaimer advising consultation with a qualified doctor (1-2 lines only).
"""

        groq_messages = [{"role": "system", "content": system_message}]
        groq_messages.extend(_format_history_for_groq(chat_history))
        groq_messages.append({"role": "user", "content": current_user_msg})

        # ── Step 5: Stream from Groq ─────────────────────────────
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        stream = client.chat.completions.create(
            model       = GROQ_MODEL,
            temperature = GROQ_TEMP,
            messages    = groq_messages,
            stream      = True,
        )

        full_answer = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_answer += token
                yield token

        # ── Step 6: Mid-stream fallback if LLM is still ignorant ─
        if _rag_has_no_answer(full_answer) and not is_analysis:
            # Mark as web fallback so UI shows warning banner
            st.session_state["is_web_fallback"] = True
            yield "\n\n---\n\n*Local data insufficient. Searching the web...*\n\n"

            # Pass empty string — we do NOT want the bad RAG answer prepended
            fallback_ans, fallback_sources, _ = _web_search_fallback(
                "", query, patient_report, chat_history
            )

            # FIX 3: Strip citations from fallback chunk too
            fallback_ans = _strip_inline_citations(fallback_ans)
            yield fallback_ans

            # Replace full_answer with ONLY the web result (discard the bad RAG answer)
            full_answer = fallback_ans

            # Replace sources with only the fallback sources (prevents irrelevant
            # images or citations from the bad RAG context leaking into the output)
            st.session_state["stream_sources"] = fallback_sources

        # ── Step 7: FIX 3 ─ Post-process full_answer ────────────
        # Strip any remaining inline citations from the completed answer
        full_answer = _strip_inline_citations(full_answer)
        # Append a formatted Sources block at the end
        sources_block = _format_sources_block(st.session_state["stream_sources"])
        if sources_block:
            full_answer += sources_block

        st.session_state["stream_buffer"] = full_answer

        st.session_state["stream_followups"] = generate_followups(
            full_answer, fused_context, cancer_filter
        )

    except Exception as e:
        err = f"Stream error: {str(e)}"
        st.session_state["stream_buffer"] = err
        yield err


# ──────────────────────────────────────────────
# Module-level stubs
# NOTE: _extract_sources below is a fallback stub only.
#       Both generate_answer_graphrag() and generate_answer_graphrag_stream()
#       import the proper _extract_sources from Cancer_retrieval_v2_visual
#       in their local import blocks above, which shadows this stub.
# ──────────────────────────────────────────────

def _build_retrieval_query(query: str, history: list[dict]) -> str:
    if not history:
        return query
    last = history[-1].get("content", "") if history else ""
    return f"{last} {query}".strip()

def _extract_sources(retrieved: list) -> list:
    """
    Stub — the proper implementation (Cancer_retrieval_v2_visual._extract_sources)
    is imported locally inside both generate_answer functions.
    """
    sources = []
    for doc in retrieved:
        meta = getattr(doc, "metadata", {})
        url  = meta.get("source") or meta.get("url") or ""
        if url and url not in sources:
            sources.append(url)
    return sources

def _format_history_for_groq(history: list[dict]) -> list[dict]:
    return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in history[-6:]]


# ══════════════════════════════════════════════════════════════════
# Section 3: Synthetic patient reports
# ══════════════════════════════════════════════════════════════════

SYNTHETIC_PATIENT_REPORTS: list[dict] = [
    {
        "patient_id": "PT-001",
        "name":       "Arjun Sharma",
        "age":        14,
        "diagnosis":  "Osteosarcoma",
        "stage":      "IIB",
        "treatment":  "MAP protocol (Methotrexate, Doxorubicin, Cisplatin)",
        "report": """
Patient: Arjun Sharma, 14-year-old male.
Diagnosis: High-grade osteosarcoma of the distal femur, Stage IIB.
Treatment: MAP protocol — high-dose Methotrexate (12 g/m²), Doxorubicin (75 mg/m²), Cisplatin (100 mg/m²).
Current symptoms: Severe nausea and vomiting post-cisplatin infusion, grade 3 mucositis,
                  significant appetite loss, weight loss of 4 kg over last cycle.
Medications: Ondansetron, Dexamethasone (antiemetic regimen).
Concurrent meds: None.
Labs: Neutropenia (ANC 800), Hb 9.2 g/dL (anemia).
""",
    },
    {
        "patient_id": "PT-002",
        "name":       "Priya Mehta",
        "age":        42,
        "diagnosis":  "Breast Cancer",
        "stage":      "III",
        "treatment":  "AC-T (Doxorubicin, Cyclophosphamide, Paclitaxel)",
        "report": """
Patient: Priya Mehta, 42-year-old female.
Diagnosis: HER2-negative ER/PR-positive invasive ductal carcinoma, Stage IIIA.
Treatment: AC-T regimen — Doxorubicin (60 mg/m²) + Cyclophosphamide (600 mg/m²) ×4 cycles,
           followed by Paclitaxel (175 mg/m²) ×4 cycles.
Current symptoms: Peripheral neuropathy (tingling fingers), alopecia, fatigue, taste changes.
Concurrent medications: Warfarin (for DVT prophylaxis), Omeprazole (GERD).
Labs: WBC 3.1, platelets 145k, Hb 10.8 g/dL.
""",
    },
    {
        "patient_id": "PT-003",
        "name":       "Rajan Pillai",
        "age":        67,
        "diagnosis":  "Lung Cancer",
        "stage":      "IV",
        "treatment":  "Pembrolizumab + Carboplatin + Paclitaxel",
        "report": """
Patient: Rajan Pillai, 67-year-old male.
Diagnosis: Stage IV non-small cell lung cancer (NSCLC), adenocarcinoma, PD-L1 TPS 65%.
Treatment: Pembrolizumab (200 mg) + Carboplatin (AUC 5) + Paclitaxel (200 mg/m²), q3w.
Current symptoms: Fatigue grade 2, decreased appetite, weight loss 6 kg, constipation.
Concurrent medications: Metformin (T2DM), Aspirin 75 mg (cardiac), Omeprazole.
Labs: Creatinine 1.4 mg/dL, Hb 10.1 g/dL, blood glucose 140 mg/dL.
""",
    },
    {
        "patient_id": "PT-004",
        "name":       "Kavitha Nair",
        "age":        58,
        "diagnosis":  "Acute Leukemia",
        "stage":      "AML",
        "treatment":  "7+3 (Cytarabine + Daunorubicin)",
        "report": """
Patient: Kavitha Nair, 58-year-old female.
Diagnosis: Acute Myeloid Leukemia (AML), intermediate risk.
Treatment: Standard 7+3 induction — Cytarabine 100 mg/m² CI days 1–7,
           Daunorubicin 60 mg/m² days 1–3.
Current symptoms: Severe mucositis grade 4, diarrhea, neutropenic fever,
                  complete appetite loss, severe oral pain.
Concurrent medications: Fluconazole (antifungal prophylaxis), Ciprofloxacin (antibacterial).
Labs: ANC 0 (aplastic), Hb 7.8 g/dL, Platelets 12k.
""",
    },
    {
        "patient_id": "PT-005",
        "name":       "Dev Krishnan",
        "age":        35,
        "diagnosis":  "Melanoma",
        "stage":      "IV",
        "treatment":  "Ipilimumab + Nivolumab (IO combination)",
        "report": """
Patient: Dev Krishnan, 35-year-old male.
Diagnosis: Stage IV metastatic melanoma, BRAF wild-type.
Treatment: Ipilimumab (3 mg/kg) + Nivolumab (1 mg/kg) q3w ×4 cycles, then Nivolumab maintenance.
Current symptoms: Immune-related diarrhea grade 2, fatigue, decreased appetite.
Concurrent medications: Dexamethasone (irAE management), Ibuprofen (joint pain).
Labs: LFTs mildly elevated (ALT 68 U/L).
""",
    },
]


# ══════════════════════════════════════════════════════════════════
# Section 4: Test queries with expected graph paths
# ══════════════════════════════════════════════════════════════════

TEST_QUERIES: list[dict] = [
    # ── Chemo side effects ─────────────────────────────────
    {
        "category":       "chemo_side_effects",
        "patient_id":     "PT-001",
        "query":          "What side effects should I expect from cisplatin in my osteosarcoma treatment?",
        "expected_graph_path": [
            "Osteosarcoma → TREATED_WITH → Cisplatin",
            "Cisplatin → CAUSES → Nausea [severe/very_common]",
            "Cisplatin → CAUSES → Nephrotoxicity [severe/common]",
            "Cisplatin → CAUSES → Ototoxicity [severe/common]",
        ],
        "expected_entities": {"cancers": ["Osteosarcoma"], "drugs": ["Cisplatin"]},
    },
    {
        "category":       "chemo_side_effects",
        "patient_id":     "PT-002",
        "query":          "My paclitaxel is causing tingling in my hands and feet. Is this normal?",
        "expected_graph_path": [
            "Paclitaxel → CAUSES → Peripheral Neuropathy [moderate/very_common]",
            "Peripheral Neuropathy → LEADS_TO → Tingling Hands/Feet",
        ],
        "expected_entities": {"drugs": ["Paclitaxel"], "side_effects": ["Peripheral Neuropathy"]},
    },
    # ── Food recommendations ───────────────────────────────
    {
        "category":       "food_recommendation",
        "patient_id":     "PT-001",
        "query":          "What foods can help me manage nausea and mouth sores from chemotherapy?",
        "expected_graph_path": [
            "Banana → HELPS → Nausea [bland BRAT food]",
            "Ginger Tea → HELPS → Nausea [gingerol effect]",
            "Ice Chips → HELPS → Mucositis [cold numbing]",
            "Custard → HELPS → Mucositis [soft high-calorie]",
        ],
        "expected_entities": {"side_effects": ["Nausea", "Mucositis"], "intents": ["food_recommendation"]},
    },
    {
        "category":       "food_recommendation",
        "patient_id":     "PT-003",
        "query":          "I have constipation during lung cancer treatment. What should I eat?",
        "expected_graph_path": [
            "Prune Juice → HELPS → Constipation [sorbitol + fiber]",
            "Oatmeal → HELPS → Constipation [soluble fiber]",
            "Whole Grain Bread → HELPS → Constipation [insoluble fiber]",
        ],
        "expected_entities": {"cancers": ["Lung Cancer"], "side_effects": ["Constipation"]},
    },
    {
        "category":       "food_recommendation",
        "patient_id":     "PT-004",
        "query":          "What foods should I absolutely avoid when I have severe mouth sores?",
        "expected_graph_path": [
            "Spicy Foods → WORSENS → Mucositis [capsaicin irritates inflamed mucosa]",
            "Citrus Juice → WORSENS → Mucositis [acid erodes damaged oral lining]",
            "Alcohol → WORSENS → Mucositis [ethanol desiccates and inflames]",
        ],
        "expected_entities": {"side_effects": ["Mucositis"], "intents": ["food_recommendation"]},
    },
    # ── Drug interactions ──────────────────────────────────
    {
        "category":       "drug_interaction",
        "patient_id":     "PT-002",
        "query":          "I'm on warfarin for blood clots and starting doxorubicin. Is there a dangerous interaction?",
        "expected_graph_path": [
            "Doxorubicin → INTERACTS_WITH → Warfarin [severe: Enhanced anticoagulation; bleeding risk]",
        ],
        "expected_entities": {"drugs": ["Doxorubicin"], "ncdrugs": ["Warfarin"], "intents": ["drug_interaction"]},
    },
    {
        "category":       "drug_interaction",
        "patient_id":     "PT-001",
        "query":          "Can I take ibuprofen for pain while on methotrexate?",
        "expected_graph_path": [
            "Methotrexate → INTERACTS_WITH → Ibuprofen [severe: NSAID inhibits renal MTX excretion]",
        ],
        "expected_entities": {"drugs": ["Methotrexate"], "ncdrugs": ["Ibuprofen"], "intents": ["drug_interaction"]},
    },
    {
        "category":       "drug_interaction",
        "patient_id":     "PT-005",
        "query":          "Will taking ibuprofen for joint pain affect my immunotherapy with ipilimumab?",
        "expected_graph_path": [
            "Ipilimumab → INTERACTS_WITH → Dexamethasone [moderate: steroids reduce checkpoint inhibitor efficacy]",
            "Ibuprofen NSAID class consideration → inflammation modulation",
        ],
        "expected_entities": {"drugs": ["Ipilimumab"], "ncdrugs": ["Ibuprofen"]},
    },
    # ── Multi-hop reasoning ────────────────────────────────
    {
        "category":       "multihop",
        "patient_id":     "PT-001",
        "query":          "What are the best foods for a child with osteosarcoma on MAP protocol?",
        "expected_graph_path": [
            "Osteosarcoma → TREATED_WITH → Cisplatin/Doxorubicin/Methotrexate",
            "Cisplatin → CAUSES → Nausea → Banana/Ginger Tea HELPS",
            "Doxorubicin → CAUSES → Mucositis → Ice Chips/Custard HELPS",
            "Methotrexate → CAUSES → Mucositis → Scrambled Eggs HELPS",
        ],
        "expected_entities": {"cancers": ["Osteosarcoma"], "intents": ["food_recommendation"]},
    },
    {
        "category":       "multihop",
        "patient_id":     "PT-004",
        "query":          "My leukemia treatment causes diarrhea. What foods help and what should I avoid?",
        "expected_graph_path": [
            "Acute Leukemia → TREATED_WITH → multiple drugs → CAUSES → Diarrhea",
            "White Rice → HELPS → Diarrhea",
            "Sports Drink → HELPS → Diarrhea [electrolytes]",
            "Spicy Foods → WORSENS → Diarrhea",
            "Dairy (full-fat) → WORSENS → Diarrhea [lactase deficiency]",
        ],
        "expected_entities": {"cancers": ["Acute Leukemia"], "side_effects": ["Diarrhea"]},
    },
]


# ══════════════════════════════════════════════════════════════════
# Section 5: Test runner (offline simulation)
# ══════════════════════════════════════════════════════════════════

def run_graph_retrieval_test(query_dict: dict) -> dict:
    """
    Runs graph retrieval for a test query and returns the context.
    Does NOT call Groq — tests only the graph retrieval layer.
    """
    from entity_extractor import extract_entities
    from graph_retrieval import query_graph, build_graph_context

    query  = query_dict["query"]
    report = ""
    for p in SYNTHETIC_PATIENT_REPORTS:
        if p["patient_id"] == query_dict.get("patient_id"):
            report = p["report"]
            break

    entities = extract_entities(query, report)
    results  = query_graph(entities)
    context  = build_graph_context(results)

    return {
        "query":             query,
        "patient_id":        query_dict.get("patient_id"),
        "category":          query_dict["category"],
        "extracted":         entities.to_dict(),
        "graph_results":     len(results),
        "graph_context_len": len(context),
        "graph_context":     context,
    }


def run_all_tests(verbose: bool = True) -> None:
    """Execute all test queries against the graph."""
    print("=" * 70)
    print("GRAPHRAG TEST SUITE")
    print("=" * 70)

    for i, tq in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Category: {tq['category']}")
        print(f"  Query: {tq['query'][:80]}...")
        try:
            result = run_graph_retrieval_test(tq)
            print(f"  Entities: {result['extracted']}")
            print(f"  Graph result sets: {result['graph_results']}")
            if verbose and result["graph_context"]:
                snippet = result["graph_context"][:400].replace("\n", "\n    ")
                print(f"  Context snippet:\n    {snippet}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("Tests complete.")


# ══════════════════════════════════════════════════════════════════
# Section 6: Expected output examples (documentation)
# ══════════════════════════════════════════════════════════════════

EXPECTED_OUTPUTS = {
    "PT-001_nausea_food": {
        "query":   "What foods help with nausea from cisplatin?",
        "graph_context_example": """
=== KNOWLEDGE GRAPH CONTEXT ===

[Recommended Foods]
  • Banana (fruit) helps with Nausea — bland easy-to-digest BRAT food
  • White Rice (grain) helps with Nausea — low-fiber easily digestible
  • Toast (grain) helps with Nausea — bland starchy food
  • Crackers (grain) helps with Nausea — bland carbohydrate buffer
  • Ginger Tea (beverage) helps with Nausea — gingerol anti-nausea effect
  • Applesauce (fruit) helps with Nausea — soft, low-fiber, easy to digest

[Foods to Avoid]
  • AVOID Fried Foods (general) — worsens Nausea: high fat delays gastric emptying
  • AVOID Alcohol (beverage) — worsens Nausea: direct gastric mucosal irritant
""",
        "expected_llm_answer_keywords": [
            "banana", "ginger", "bland", "crackers",
            "small meals", "avoid fried", "avoid alcohol",
        ],
    },
    "PT-002_warfarin_interaction": {
        "query":   "Is doxorubicin safe with warfarin?",
        "graph_context_example": """
=== KNOWLEDGE GRAPH CONTEXT ===

[Serious Drug Interactions]
  • Doxorubicin + Warfarin [severe]: Enhanced anticoagulation; bleeding risk
    (confidence: 0.94)
""",
        "expected_llm_answer_keywords": [
            "severe interaction", "bleeding risk", "anticoagulation",
            "monitor INR", "consult doctor",
        ],
    },
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Uncomment to run tests against live Neo4j:
    # run_all_tests(verbose=True)

    # Print synthetic reports for review
    print("\n=== SYNTHETIC PATIENT REPORTS ===")
    for p in SYNTHETIC_PATIENT_REPORTS:
        print(f"\n{p['patient_id']} | {p['name']} | {p['diagnosis']} | Stage {p['stage']}")
        print(f"  Treatment: {p['treatment']}")

    print("\n=== TEST QUERIES ===")
    for i, tq in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}] [{tq['category']}] {tq['query']}")
        print(f"  Expected graph path: {tq['expected_graph_path'][0]}")