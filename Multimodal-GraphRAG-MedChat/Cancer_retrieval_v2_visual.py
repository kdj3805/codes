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
    from ddgs import DDGS
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

#GROQ_MODEL = "llama-3.3-70b-versatile"   # kept from original — can experiment with smaller models if latency is an issue
#GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL = "llama-3.1-8b-instant"   # temp experiment with instruct-tuned version for better adherence to answer guidelines
GROQ_TEMP  = 0.1

MEMORY_TURNS = 4
FOLLOWUP_COUNT = 3

# Phrases that reliably mean "I can't answer" wherever they appear in the response.
_NO_ANSWER_PHRASES_STRONG = [
    "do not have enough information",
    "not found in the context",
    "not in the context",
    "not provided in the context",
    "not available in the context",
    "not available in the provided context",
    "not covered in the provided",
    "outside the scope of the provided",
    "beyond the scope of the provided",
    "context does not contain",
    "context does not include",
    "context does not mention",
    "context does not provide",
    "no mention of",
]

# Broader phrases that might appear in passing in a valid answer.
# Only checked in the FIRST 250 characters of the answer to avoid false positives.
_NO_ANSWER_PHRASES_CONTEXTUAL = [
    "not mentioned in",
    "not provided in",
    "cannot find",
    "no information",
    "not recognized",
    "not described",
    "not explicitly stated",
    "not explicitly mentioned",
    "not directly mentioned",
    "not directly stated",
    "not directly addressed",
    "not able to provide",
    "unable to provide",
    "cannot provide specific",
    "no specific information",
    "no relevant information",
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
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding       = get_embeddings(),
        path            = str(QDRANT_DIR),
        collection_name = COLLECTION_REVIEWS,
    )

    search_kwargs: dict = {"k": K_DENSE}

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
            f"No chunk files in {CHUNK_DIR}. Run cancer_ingestion.py first."
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
    cancer_filter: str = "",
) -> List[Document]:
    
    scores:  dict[str, float]    = {}
    doc_map: dict[str, Document] = {}

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
        cancer_filter = cancer_filter,
    )


def select_best_images(query, docs, top_k=2):
    """
    Select most relevant images based on query + caption similarity.
    FIXED: Now filters out stop-words so 'the', 'of', 'are' do not trigger false positives.
    """
    query = query.lower()
    scored = []

    # Filter out common grammar words so only true keywords are scored
    stop_words = {"what", "are", "the", "of", "my", "in", "is", "for", "to", "a", "an", "and", "or", "on", "with", "from", "how", "does", "do", "can", "i", "you", "me", "about", "this", "that", "these", "those"}
    
    # Strip punctuation and keep only meaningful words
    query_words = [w for w in re.findall(r'\w+', query) if w not in stop_words and len(w) > 2]

    for doc in docs:
        meta = doc.metadata

        if not meta.get("image_filename"):
            continue

        caption = meta.get("caption", "").lower()
        score = 0

        # keyword overlap scoring using ONLY meaningful words
        for word in query_words:
            if word in caption:
                score += 2

        # strong boosts
        if "abcde" in query and "abcde" in caption:
            score += 15
        if "survival" in query and "survival" in caption:
            score += 15
        if "stage" in query and "stage" in caption:
            score += 10
        if "topical" in query and "topical" in caption:
            score += 10

        if score > 0:
            scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]


def build_visual_asset_section(query, docs):
    best_images = select_best_images(query, docs)

    if not best_images:
        return ""

    # Must EXACTLY match the string used in graphrag_integration.py to prevent truncation
    lines = ["\n\n## Extracted Visual Assets Database\n"]

    for doc in best_images:
        meta = doc.metadata
        
        # Must EXACTLY match the tag the LLM is instructed to output
        lines.append(
            f"[IMAGE: {meta.get('image_filename')}]\n"
            f"Description: {meta.get('caption')}\n"
        )

    return "\n".join(lines)


def build_context(docs: List[Document], query: str = "" ) -> str:
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

    visual_section = build_visual_asset_section(query or "", docs)

    return "\n\n".join(parts) + visual_section


def _rag_has_no_answer(answer: str) -> bool:
    if not answer:
        return True

    lower = answer.lower()
    prefix = lower[:250]

    if len(answer.strip()) < 120:
        all_phrases = _NO_ANSWER_PHRASES_STRONG + _NO_ANSWER_PHRASES_CONTEXTUAL
        if any(p in lower for p in all_phrases):
            return True

    if any(p in lower for p in _NO_ANSWER_PHRASES_STRONG):
        return True

    if any(p in prefix for p in _NO_ANSWER_PHRASES_CONTEXTUAL):
        return True

    return False


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

    last_user_msg = ""
    for turn in reversed(chat_history):
        if turn.get("role") == "user":
            last_user_msg = turn.get("content", "")[:120].strip()
            break

    if last_user_msg and last_user_msg.lower() != query.lower():
        return f"{last_user_msg} — {query}"

    return query


def _format_history_for_groq(
    chat_history: list[dict],
    max_turns:    int = MEMORY_TURNS,
) -> list[dict]:
    
    if not chat_history:
        return []

    recent = chat_history[-(max_turns * 2):]

    groq_messages = []
    for turn in recent:
        role    = turn.get("role", "")
        content = turn.get("content", "").strip()

        if not content:
            continue

        if "patient report loaded" in content.lower():
            continue

        if role == "assistant":
            content = re.sub(r'\[IMAGE:\s*[^\]]+\]', '[figure]', content)
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
            temperature = 0.4,
            max_tokens  = 120,
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
        print(f"Follow-up generation skipped (non-critical): {e}")
        return []


def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
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
    chat_history:   list[dict] = None,
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

    if rag_answer.strip():
        combined = rag_answer.strip() + "\n\n---\n\n**Web Search Result:**\n\n" + web_answer.strip()
    else:
        combined = web_answer.strip()
        
    return combined, web_sources, []