from __future__ import annotations

import os
import re
import json
import math
from pathlib import Path
from typing import Any, Generator, List, Optional
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever  # BM25 sparse retrieval
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from groq import Groq

try:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    _QDRANT_FILTER_AVAILABLE = True
except ImportError:
    _QDRANT_FILTER_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    _DDG_AVAILABLE = False

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # kept from original


BASE_DIR   = Path(__file__).parent
QDRANT_DIR = BASE_DIR / "vector_db" / "qdrantt_store"   # double-t — intentional
CHUNK_DIR  = BASE_DIR / "output" / "chunks"
IMAGE_DIR  = BASE_DIR / "output" / "images"

COLLECTION_REVIEWS = "medical_reviews"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

K_DENSE     = 30   
K_SPARSE    = 30   
K_RRF_FINAL = 30   
K_MMR_FINAL = 15   
MMR_LAMBDA  = 0.6
RRF_K       = 60

GROQ_MODEL = "llama-3.3-70b-versatile"   # kept from original — can experiment with smaller models if latency is an issue
GROQ_TEMP  = 0.1

MEMORY_TURNS = 4

FOLLOWUP_COUNT = 3

_NO_ANSWER_PHRASES = [
    "do not have enough information",
    "not mentioned in",
    "not provided in",
    "not found in the context",
    "cannot find",
    "no information",
    "not in the context",
    "not recognized",
    "not described",
    "no mention of",
]

_embed_model: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embed_model
    if _embed_model is None:
        print("   Loading embedding model (once)...")
        _embed_model = HuggingFaceEmbeddings(
            model_name    = EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
    return _embed_model

def get_dense_retriever(cancer_filter: str = "") -> BaseRetriever:
    """
    Original: QdrantVectorStore, k=K_DENSE, no parameters.

    [T3-3] NEW param: cancer_filter
      When set (e.g. "breast"), adds a Qdrant must-filter on the
      cancer_type metadata field so only matching chunks are returned.
      When "" or "all" (default), behaves identically to original.
    """
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding       = get_embeddings(),
        path            = str(QDRANT_DIR),
        collection_name = COLLECTION_REVIEWS,
    )

    # Start with original search kwargs
    search_kwargs: dict = {"k": K_DENSE}

    #  Append Qdrant filter only when a specific type is selected
    if cancer_filter and cancer_filter.lower() not in ("", "all"):
        if _QDRANT_FILTER_AVAILABLE:
            search_kwargs["filter"] = Filter(
                must=[
                    FieldCondition(
                        key   = "cancer_type",
                        match = MatchValue(value=cancer_filter.lower()),
                    )
                ]
            )
            print(f"Dense filter active: cancer_type='{cancer_filter}'")
        else:
            # Older qdrant-client — skip filter, search all types
            print("Qdrant filter models unavailable — cancer filter ignored")

    return vector_store.as_retriever(search_kwargs=search_kwargs)


_bm25_retriever: Optional[BM25Retriever] = None

def get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever

    print("   Building BM25 index...")
    documents = []
    for json_path in sorted(CHUNK_DIR.glob("*_chunks.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                documents.append(Document(
                    page_content = chunk.get("content", ""),
                    metadata     = chunk,
                ))

    if not documents:
        raise FileNotFoundError(
            f"No chunk files in {CHUNK_DIR}. "
            "Run cancer_ingestion.py first."
        )

    _bm25_retriever   = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = K_SPARSE
    print(f"   BM25 ready: {len(documents)} chunks indexed")
    return _bm25_retriever


def reciprocal_rank_fusion(
    dense_docs:    List[Document],
    sparse_docs:   List[Document],
    k:             int = RRF_K,
    top_n:         int = K_RRF_FINAL,
    cancer_filter: str = "",          # [T3-3] NEW — BM25 post-filter
) -> List[Document]:
    
    scores:  dict[str, float]    = {}
    doc_map: dict[str, Document] = {}

    # Score dense results (already Qdrant-filtered if filter was set)
    for rank, doc in enumerate(dense_docs):
        did = doc.metadata.get("chunk_id", str(id(doc)))
        scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        doc_map[did] = doc

    
    filtered_sparse = sparse_docs
    if cancer_filter and cancer_filter.lower() not in ("", "all"):
        filtered_sparse = [
            doc for doc in sparse_docs
            if doc.metadata.get("cancer_type", "").lower() == cancer_filter.lower()
        ]

    for rank, doc in enumerate(filtered_sparse):
        did = doc.metadata.get("chunk_id", str(id(doc)))
        scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        doc_map[did] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[i] for i in sorted_ids[:top_n]]


def _cosine(v1: List[float], v2: List[float]) -> float:
    dot   = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return 0.0 if (norm1 == 0 or norm2 == 0) else dot / (norm1 * norm2)


def mmr_rerank(
    query:       str,
    candidates:  List[Document],
    embed_model: HuggingFaceEmbeddings,
    k:           int   = K_MMR_FINAL,
    lambda_mult: float = MMR_LAMBDA,
) -> List[Document]:
    
    if not candidates or len(candidates) <= k:
        return candidates

    query_vec = embed_model.embed_query(query)
    doc_vecs  = embed_model.embed_documents([d.page_content for d in candidates])
    relevance = [_cosine(v, query_vec) for v in doc_vecs]

    selected:  List[int] = []
    remaining: List[int] = list(range(len(candidates)))

    for _ in range(min(k, len(candidates))):
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best, best_score = -1, float("-inf")
            for idx in remaining:
                max_sim = max(_cosine(doc_vecs[idx], doc_vecs[s]) for s in selected)
                score   = lambda_mult * relevance[idx] - (1 - lambda_mult) * max_sim
                if score > best_score:
                    best_score, best = score, idx
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]

class HybridMMRRetriever(BaseRetriever):
    dense_ret:     Any = None
    sparse_ret:    Any = None
    embed_model:   Any = None
    cancer_filter: str = ""    

    def _get_relevant_documents(self, query: str) -> List[Document]:
        dense_docs  = self.dense_ret.invoke(query)
        sparse_docs = self.sparse_ret.invoke(query)
        rrf_results = reciprocal_rank_fusion(
            dense_docs,
            sparse_docs,
            cancer_filter = self.cancer_filter,
        )
        return mmr_rerank(query, rrf_results, self.embed_model)


def get_hybrid_mmr_retriever(cancer_filter: str = "") -> HybridMMRRetriever:
    return HybridMMRRetriever(
        dense_ret     = get_dense_retriever(cancer_filter),
        sparse_ret    = get_bm25_retriever(),
        embed_model   = get_embeddings(),
        cancer_filter = cancer_filter,    # [T3-3]
    )

def build_context(docs: List[Document]) -> str:

    parts = []
    for i, doc in enumerate(docs, 1):
        sf = (
            doc.metadata.get("source_file")
            or doc.metadata.get("source")
            or doc.metadata.get("chunk_id", "").rsplit("_", 1)[0]
            or "unknown"
        )
        parts.append(
            f"[{i}] Source: {sf} | "
            f"Cancer: {doc.metadata.get('cancer_type', 'general')} | "
            f"Section: {doc.metadata.get('section_hierarchy', 'Body')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _rag_has_no_answer(answer: str) -> bool:
    lower = answer.lower()
    return any(p in lower for p in _NO_ANSWER_PHRASES)


def _extract_sources(retrieved: List[Document]) -> list[dict]:

    sources: list[dict] = []
    seen:    set        = set()
    for doc in retrieved:
        sf = doc.metadata.get("source_file", "").strip()

        if not sf:
            cid = doc.metadata.get("chunk_id", "")
            if cid:
                sf = re.sub(r'_cap_\d+$', '', cid)
                sf = re.sub(r'_\d{4}$',   '', sf)

        if not sf:
            continue

        url = doc.metadata.get("source_url", "")
        if sf not in seen:
            seen.add(sf)
            sources.append({"label": sf, "url": url})

    return sources

def _build_retrieval_query(query: str, chat_history: list[dict]) -> str:

    if not chat_history:
        return query

    # Walk backwards to find the most recent user message
    last_user_msg = ""
    for turn in reversed(chat_history):
        if turn.get("role") == "user":
            # Truncate to 120 chars — embedding models handle short queries better
            last_user_msg = turn.get("content", "")[:120].strip()
            break

    # Only prepend if it adds new context (not identical to current query)
    if last_user_msg and last_user_msg.lower() != query.lower():
        return f"{last_user_msg} — {query}"

    return query

def _format_history_for_groq(
    chat_history: list[dict],
    max_turns:    int = MEMORY_TURNS,
) -> list[dict]:
    
    if not chat_history:
        return []

    # Keep last max_turns * 2 messages (user + assistant per turn)
    recent = chat_history[-(max_turns * 2):]

    groq_messages = []
    for turn in recent:
        role    = turn.get("role", "")
        content = turn.get("content", "").strip()

        if not content:
            continue

        # Skip the auto-analysis trigger message — not useful as history context
        if "patient report loaded" in content.lower():
            continue

        if role == "assistant":
            # Strip [IMAGE: ...] tags — saves tokens, LLM doesn't need them in history
            content = re.sub(r'\[IMAGE:\s*[^\]]+\]', '[figure]', content)
            # Truncate very long assistant messages to save token budget
            if len(content) > 400:
                content = content[:400] + "…"

        if role in ("user", "assistant"):
            groq_messages.append({"role": role, "content": content})

    return groq_messages

def generate_followups(
    answer:      str,
    context:     str,
    cancer_type: str = "",
) -> list[str]:
    
    try:
        cancer_hint = f" about {cancer_type} cancer" if cancer_type else ""
        prompt = (
            f"Based on this medical answer{cancer_hint}, suggest exactly "
            f"{FOLLOWUP_COUNT} short follow-up questions a patient might ask next.\n\n"
            f"ANSWER:\n{answer[:600]}\n\n"
            f"CONTEXT SUMMARY:\n{context[:400]}\n\n"
            "Rules:\n"
            "- Each question must be under 12 words\n"
            "- Make questions specific and clinically useful\n"
            "- Output ONLY the questions, one per line, no numbering, no bullets\n"
            "- Do not repeat the original question\n"
        )

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp   = client.chat.completions.create(
            model       = GROQ_MODEL,
            temperature = 0.4,   # slightly higher than main answer for variety
            max_tokens  = 120,   # tight budget — just 3 short questions
            messages    = [{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or ""
        questions = []
        for line in raw.strip().splitlines():
            q = line.strip().lstrip("0123456789.-) ").strip()
            if q and len(q) > 5:
                questions.append(q)

        return questions[:FOLLOWUP_COUNT]

    except Exception as e:
        # Non-critical — log quietly and return empty list
        print(f"Follow-up generation skipped (non-critical): {e}")
        return []

def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """Unchanged from original."""
    if not _DDG_AVAILABLE:
        return []
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                f"{query} medical oncology",
                max_results = max_results,
            ):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as e:
        print(f" DuckDuckGo search error: {e}")
        return []


def _web_search_fallback(
    rag_answer:     str,
    query:          str,
    patient_report: str,
    chat_history:   list[dict] = None,   # [T3-1] NEW
) -> tuple[str, list, list]:
  
    if chat_history is None:
        chat_history = []

    client      = Groq(api_key=os.getenv("GROQ_API_KEY"))
    web_results = _duckduckgo_search(query)
    web_sources = [{"label": r["url"], "url": r["url"]}
                    for r in web_results if r.get("url")]

    if web_results:
        web_context = "\n\n".join([
            f"[W{i+1}] {r['title']}\nURL: {r['url']}\n{r['snippet']}"
            for i, r in enumerate(web_results)
        ])
        # Prompt unchanged from original
        web_prompt = f"""You are a medical AI assistant.
The user asked a medical question not covered by local clinical literature.
Use the web search results below to provide an accurate answer.

PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

WEB SEARCH RESULTS:
{web_context}

QUESTION: {query}

INSTRUCTIONS:
- Answer using the web search results above.
- Cite results using [W1], [W2] etc.
- Explain clearly — avoid heavy jargon.
- End with a disclaimer advising consultation with a qualified doctor.
"""
    else:
        # Prompt unchanged from original
        web_prompt = f"""You are a medical AI assistant.
Answer this medical question using your training knowledge.

PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, accurate answer.
- End with a disclaimer advising consultation with a qualified doctor.
"""
        web_sources = [
            {"label": "https://www.cancer.gov",                "url": "https://www.cancer.gov"},
            {"label": "https://www.ncbi.nlm.nih.gov/pubmed/", "url": "https://www.ncbi.nlm.nih.gov/pubmed/"},
        ]

    web_answer = ""
    try:
        # [T3-1] Inject history so web-fallback answer has memory context
        groq_messages = _format_history_for_groq(chat_history)
        groq_messages.append({"role": "user", "content": web_prompt})

        resp = client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = groq_messages,
            temperature = GROQ_TEMP,
        )
        web_answer = resp.choices[0].message.content or ""
    except Exception as e:
        web_answer = f"Could not generate web answer: {e}"

    divider  = "\n\n---\n\n**Web Search Result:**\n\n"
    combined = rag_answer.strip() + divider + web_answer.strip()
    # 3-tuple — no follow-ups for web fallback
    return combined, web_sources, []

def generate_answer(
    query:          str,
    patient_report: str        = "",
    chat_history:   list[dict] = None,
    cancer_filter:  str        = "",
    is_analysis:    bool       = False,   
) -> tuple[str, list, list]:
    
    if chat_history is None:
        chat_history = []

    try:
        print(f"\n [v4] Query: {query[:70]}...")

        retrieval_query = _build_retrieval_query(query, chat_history)

        retriever = get_hybrid_mmr_retriever(cancer_filter=cancer_filter)
        retrieved = retriever.invoke(retrieval_query)

    
        if not retrieved:
            empty_rag = (
                "According to the provided clinical context, "
                "no relevant information was found for this query."
            )
            if not is_analysis:
                return _web_search_fallback(
                    empty_rag, query, patient_report, chat_history
                )
            else:
                return empty_rag, [], []

        context_text = build_context(retrieved)
        sources      = _extract_sources(retrieved)

        system_message = (
            "You are an empathetic medical AI assistant helping "
            "cancer patients and clinicians understand medical information.\n"
            "CRITICAL SYSTEM DIRECTIVE: You are connected to a frontend UI capable of rendering images. "
            "NEVER state that you are a text-based AI or cannot display images. "
            "If the context provides an image reference relevant to the answer, you MUST use the exact format `[IMAGE: filename.ext]` to display it."
        )

        current_user_msg = f"""PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

CLINICAL CONTEXT (from peer-reviewed medical literature):
{context_text}

QUESTION:
{query}

INSTRUCTIONS:
- Answer using ONLY the clinical context provided above.
- Explain clearly — avoid heavy jargon where possible.
- Cite source numbers [1], [2] etc. when referencing specific facts.
- IMPORTANT: If the clinical context contains a "Visual Assets Database"
  section with image references relevant to the question, you MUST
  include them in your answer using this EXACT format:
    [IMAGE: filename.png]
- If the answer is not in the context, clearly state you do not have enough information.
- End with a disclaimer advising consultation with a qualified doctor.
"""

        groq_messages = [{"role": "system", "content": system_message}]
        groq_messages.extend(_format_history_for_groq(chat_history))
        groq_messages.append({"role": "user", "content": current_user_msg})

        client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model       = GROQ_MODEL,
            temperature = GROQ_TEMP,
            messages    = groq_messages,
        )

        answer = response.choices[0].message.content or ""

        if _rag_has_no_answer(answer) and not is_analysis:
            print("   RAG insufficient → web search fallback...")
            return _web_search_fallback(answer, query, patient_report, chat_history)

        followups = generate_followups(answer, context_text, cancer_filter)

        return answer, sources, followups

    except Exception as e:
        return f"Error: {str(e)}", [], []

def generate_answer_stream(
    query:          str,
    patient_report: str        = "",
    chat_history:   list[dict] = None,
    cancer_filter:  str        = "",
) -> Generator[str, None, None]:

    import streamlit as st

    if chat_history is None:
        chat_history = []

    st.session_state["stream_buffer"]    = ""
    st.session_state["stream_sources"]   = []
    st.session_state["stream_followups"] = []

    try:
        print(f"\n [v4-stream] Query: {query[:70]}...")

        retrieval_query = _build_retrieval_query(query, chat_history)

        retriever = get_hybrid_mmr_retriever(cancer_filter=cancer_filter)
        retrieved = retriever.invoke(retrieval_query)

        if not retrieved:
            msg = "No relevant information found in the clinical literature."
            st.session_state["stream_buffer"] = msg
            yield msg
            return

        context_text = build_context(retrieved)
        st.session_state["stream_sources"] = _extract_sources(retrieved)

        system_message = (
            "You are an empathetic medical AI assistant helping "
            "cancer patients and clinicians understand medical information.\n"
            "CRITICAL SYSTEM DIRECTIVE: You are connected to a frontend UI capable of rendering images. "
            "NEVER state that you are a text-based AI or cannot display images. "
            "If the context provides an image reference relevant to the answer, you MUST use the exact format `[IMAGE: filename.ext]` to display it."
        )

        current_user_msg = f"""PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

CLINICAL CONTEXT:
{context_text}

QUESTION:
{query}

INSTRUCTIONS:
- Answer using ONLY the clinical context provided above.
- Explain clearly — avoid heavy jargon where possible.
- Cite source numbers [1], [2] etc. when referencing specific facts.
- IMPORTANT: If the clinical context contains a "Visual Assets Database"
  section with image references relevant to the question, you MUST
  include them in your answer using this EXACT format:
    [IMAGE: filename.png]
- If the answer is not in the context, clearly state you do not have enough information.
- End with a disclaimer advising consultation with a qualified doctor.
"""

        groq_messages = [{"role": "system", "content": system_message}]
        groq_messages.extend(_format_history_for_groq(chat_history))
        groq_messages.append({"role": "user", "content": current_user_msg})

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

        st.session_state["stream_buffer"] = full_answer

        st.session_state["stream_followups"] = generate_followups(
            full_answer, context_text, cancer_filter
        )

    except Exception as e:
        err = f"Stream error: {str(e)}"
        st.session_state["stream_buffer"] = err
        yield err



if __name__ == "__main__":

    # Test 1: standard queries (no history, no filter)
    tests = [
        ("What topical drug delivery systems are used in breast cancer?", ""),
        ("What is the 5-year survival rate for osteosarcoma?",            "osteosarcoma"),
        ("What is Stage V melanoma survival rate?",                       ""),
    ]
    for q, cf in tests:
        print(f"\n{'─'*62}\n {q}  [filter={cf or 'none'}]")
        answer, sources, followups = generate_answer(q, cancer_filter=cf)
        print(f"{answer[:300]}...")
        print(f"Image tags : {'YES' if '[IMAGE:' in answer else 'NO ❌'}")
        print(f"Sources    : {[s['label'] for s in sources]}")
        print(f"Follow-ups : {followups}")

    # Test 2: conversation memory
    print(f"\n{'─'*62}")
    print("Memory test — follow-up without restating cancer type")
    history = [
        {"role": "user",      "content": "Tell me about breast cancer treatment."},
        {"role": "assistant", "content": "Breast cancer treatment options include surgery..."},
    ]
    answer, sources, followups = generate_answer(
        query        = "what about for HER2 positive patients specifically?",
        chat_history = history,
    )
    print(f"{answer[:300]}...")
    print(f"Follow-ups: {followups}")