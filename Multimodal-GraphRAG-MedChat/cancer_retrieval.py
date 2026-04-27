# ================== cancer_retrieval.py ==================
# Multimodal-GraphRAG-MedChat — Hybrid + MMR Retrieval Pipeline
#
# RETRIEVAL STRATEGY (3 stages):
#   Stage 1 — HYBRID  : Dense (Qdrant) + Sparse (BM25) → RRF merge
#   Stage 2 — MMR     : Re-rank 20 candidates → 8 diverse final chunks
#   Stage 3 — LLM     : Groq llama-3.3-70b generates cited answer
#
# FALLBACK:
#   If the LLM answer admits "no information in context" →
#   DuckDuckGo search is triggered (free, no API key needed).
#   Top 5 snippets are passed as context to Groq LLM.
#   Answer format becomes:
#     [RAG section explaining gap]
#     ─────────────────────────────
#     🌐 Web Search Result:
#     [web answer citing W1, W2...]
#   Sources return real URLs from DuckDuckGo results.
#
# FIGURE DETECTION:
#   If query contains figure-related keywords, a secondary BM25 search
#   is run specifically for caption chunks of the same cancer type.
#   This prevents figure captions from being dropped by MMR.
#
# EMBEDDING MODEL (must match cancer_ingestion.py):
#   BAAI/bge-base-en-v1.5 — 768 dimensions — fully local
# ==========================================================

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Any, List, Optional
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from groq import Groq

try:
    from duckduckgo_search import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    _DDG_AVAILABLE = False

load_dotenv()

# ==========================================================
# ======================== CONFIG ==========================
# ==========================================================

BASE_DIR   = Path(__file__).parent
QDRANT_DIR = BASE_DIR / "vector_db" / "qdrant_store"
CHUNK_DIR  = BASE_DIR / "output" / "chunks"
IMAGE_DIR  = BASE_DIR / "output" / "images"

COLLECTION_REVIEWS = "medical_reviews"

# ── Embedding model — MUST match cancer_ingestion.py ──────
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM   = 768

# ── Retrieval parameters ───────────────────────────────────
K_DENSE     = 20    # dense candidate pool
K_SPARSE    = 20    # sparse candidate pool
K_RRF_FINAL = 20    # keep top N after RRF
K_MMR_FINAL = 8     # final docs sent to LLM
MMR_LAMBDA  = 0.6   # 1.0 = pure relevance, 0.0 = pure diversity
RRF_K       = 60    # standard RRF smoothing constant

# ── LLM ───────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMP  = 0.1

# ── Phrases that mean "RAG had no answer" ─────────────────
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

# ── Keywords that indicate the question is about a figure ─
_FIGURE_QUERY_KEYWORDS = [
    # Direct visual references
    "figure", "image", "diagram", "illustration", "photo",
    "graph", "chart", "panel", "shown", "depicted", "illustrat",
    # Medical paper visual types
    "flowchart", "flow chart", "flow diagram",
    "table", "prisma", "preferred reporting",
    "systematic review", "meta-analysis", "meta analysis",
    "kaplan-meier", "survival curve", "forest plot",
    "screening", "identification", "reporting items",
    # Histology / pathology visuals
    "growth phase", "staining", "microscop", "histolog",
    # Action words that imply a visual answer
    "describe the", "show the", "explain the", "what does the",
    "what are the", "illustrate",
]

# ==========================================================
# =================== EMBEDDINGS ===========================
# ==========================================================

_embed_model: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Cached embedding model — loaded once, reused forever."""
    global _embed_model
    if _embed_model is None:
        print("   Loading embedding model (once)...")
        _embed_model = HuggingFaceEmbeddings(
            model_name    = EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
    return _embed_model

# ==========================================================
# ================== STAGE 1A: DENSE ======================
# ==========================================================

def get_dense_retriever() -> BaseRetriever:
    """Qdrant HNSW cosine similarity — top K_DENSE results."""
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding       = get_embeddings(),
        path            = str(QDRANT_DIR),
        collection_name = COLLECTION_REVIEWS,
    )
    return vector_store.as_retriever(search_kwargs={"k": K_DENSE})

# ==========================================================
# ================ STAGE 1B: SPARSE (BM25) ================
# ==========================================================

_bm25_retriever: Optional[BM25Retriever] = None

def get_bm25_retriever() -> BM25Retriever:
    """
    BM25 keyword retriever — loaded once from output/chunks/*.json.
    Finds exact drug/gene names that dense search may miss.
    """
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
    print(f"   BM25 ready: {len(documents)} chunks")
    return _bm25_retriever

# ==========================================================
# ================== STAGE 1C: RRF MERGE ==================
# ==========================================================

def reciprocal_rank_fusion(
    dense_docs:  List[Document],
    sparse_docs: List[Document],
    k:    int = RRF_K,
    top_n: int = K_RRF_FINAL,
) -> List[Document]:
    """
    Merges dense + sparse by rank position (not raw scores).
    Score = 1/(k + rank). Docs in both lists get scores added.
    """
    scores:  dict[str, float]    = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        did = doc.metadata.get("chunk_id", str(id(doc)))
        scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        doc_map[did] = doc

    for rank, doc in enumerate(sparse_docs):
        did = doc.metadata.get("chunk_id", str(id(doc)))
        scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        doc_map[did] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[i] for i in sorted_ids[:top_n]]

# ==========================================================
# ================ STAGE 2: MMR RE-RANKING ================
# ==========================================================

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
    """
    Greedy MMR: picks k docs that are relevant AND diverse.
    MMR score = lambda * sim(doc, query)
                - (1 - lambda) * max(sim(doc, already_selected))
    """
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

# ==========================================================
# ================= HYBRID + MMR RETRIEVER ================
# ==========================================================

class HybridMMRRetriever(BaseRetriever):
    dense_ret:   Any = None
    sparse_ret:  Any = None
    embed_model: Any = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        dense_docs  = self.dense_ret.invoke(query)
        sparse_docs = self.sparse_ret.invoke(query)
        rrf_results = reciprocal_rank_fusion(dense_docs, sparse_docs)
        return mmr_rerank(query, rrf_results, self.embed_model)


def get_hybrid_mmr_retriever() -> HybridMMRRetriever:
    return HybridMMRRetriever(
        dense_ret   = get_dense_retriever(),
        sparse_ret  = get_bm25_retriever(),
        embed_model = get_embeddings(),
    )

# ==========================================================
# =================== HELPERS ==============================
# ==========================================================

def split_text_and_figures(
    docs: List[Document],
) -> tuple[List[Document], List[Document]]:
    """Separates text chunks from figure/table caption chunks."""
    text, figs = [], []
    for doc in docs:
        ct = doc.metadata.get("content_type", "clinical_text")
        (figs if ct in ("figure_caption", "table_caption") else text).append(doc)
    return text, figs


def build_context(text_docs: List[Document]) -> str:
    """Numbered context string for LLM prompt."""
    parts = []
    for i, doc in enumerate(text_docs, 1):
        sf = (
            doc.metadata.get("source_file")
            or doc.metadata.get("source")
            or doc.metadata.get("file")
            or doc.metadata.get("chunk_id", "").rsplit("_", 1)[0]
            or "unknown"
        )
        parts.append(
            f"[{i}] Source: {sf} | "
            f"Cancer: {doc.metadata.get('cancer_type','general')} | "
            f"Section: {doc.metadata.get('section_hierarchy','Body')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _rag_has_no_answer(answer: str) -> bool:
    """True if LLM admitted it could not find answer in RAG context."""
    lower = answer.lower()
    return any(p in lower for p in _NO_ANSWER_PHRASES)


def _collect_figures(
    figure_docs: List[Document],
    text_docs:   List[Document],
    query:       str,
) -> List[dict]:
    """
    Collects renderable figure dicts from two sources:

    Source 1 — caption chunks that survived MMR selection directly.

    Source 2 — dedicated BM25 caption search, triggered when:
      a) query contains figure-related keywords (flowchart, PRISMA,
         table, diagram, graph, etc.), OR
      b) query explicitly asks to describe/show something visual.

      Cancer_type filter behaviour:
        - If text_docs are all from one cancer type → filter to that type
        - If text_docs span multiple cancer types, or query mentions
          cross-paper visuals (PRISMA, systematic review, flowchart)
          → search ALL cancer types (no filter)
          This handles PRISMA flowcharts which appear in any paper.
    """
    figures    = []
    seen_paths: set = set()

    def _add_figure(doc: Document):
        img_path = doc.metadata.get("image_path")
        if not img_path:
            return
        full = BASE_DIR / img_path
        if full.exists() and str(full) not in seen_paths:
            seen_paths.add(str(full))
            sf = (
                doc.metadata.get("source_file")
                or doc.metadata.get("source")
                or doc.metadata.get("file")
                or ""
            )
            figures.append({
                "caption":    doc.page_content,
                "image_path": str(full),
                "source":     sf,
            })

    # ── Source 1: directly retrieved caption chunks ────────
    for doc in figure_docs:
        _add_figure(doc)

    # ── Source 2: dedicated BM25 caption search ────────────
    query_lower     = query.lower()
    is_figure_query = any(kw in query_lower for kw in _FIGURE_QUERY_KEYWORDS)

    # Keywords that indicate a cross-paper / methodology visual
    # → PRISMA, flowchart, systematic review appear in any cancer paper
    CROSS_PAPER_KEYWORDS = [
        "prisma", "flowchart", "flow chart", "flow diagram",
        "systematic review", "meta-analysis", "meta analysis",
        "preferred reporting", "identification", "screening",
        "forest plot", "reporting items",
    ]
    is_cross_paper = any(kw in query_lower for kw in CROSS_PAPER_KEYWORDS)

    if is_figure_query:
        # Determine cancer_type filter
        if is_cross_paper or not text_docs:
            # No cancer_type filter — search all papers
            cancer_types = None
        else:
            cancer_types = {doc.metadata.get("cancer_type", "") for doc in text_docs}
            # If mixed cancer types, also open up the filter
            if len(cancer_types) > 2:
                cancer_types = None

        try:
            cap_results = get_bm25_retriever().invoke(query)
            for doc in cap_results:
                if len(figures) >= 4:
                    break
                ct = doc.metadata.get("content_type", "")
                if ct not in ("figure_caption", "table_caption"):
                    continue
                # Apply cancer_type filter only when meaningful
                if cancer_types is not None:
                    if doc.metadata.get("cancer_type") not in cancer_types:
                        continue
                _add_figure(doc)
        except Exception:
            pass

    return figures

# ==========================================================
# ============= WEB SEARCH FALLBACK ========================
# ==========================================================

def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches DuckDuckGo and returns a list of:
      [{"title": ..., "url": ..., "snippet": ...}, ...]

    Requires: pip install duckduckgo-search
    Free — no API key needed.
    """
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
        print(f"   ⚠️  DuckDuckGo search error: {e}")
        return []


def _web_search_fallback(
    rag_answer:     str,
    query:          str,
    patient_report: str,
) -> tuple[str, list, list]:
    """
    Called when RAG answer contains an "I don't know" phrase.

    Flow:
      1. Search DuckDuckGo for top 5 results
      2. Pass result snippets as context to Groq LLM
      3. LLM generates an answer grounded in web snippets
      4. Return combined answer:
           [RAG section — explains what context DID say]
           ─────────────────────────────────────────────
           🌐 Web Search Result:
           [web answer]
      5. Sources = list of real URLs from DuckDuckGo results
    """
    client      = Groq(api_key=os.getenv("GROQ_API_KEY"))
    web_results = _duckduckgo_search(query)
    web_sources = [r["url"] for r in web_results if r.get("url")]

    if web_results:
        # Build web context from snippets
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

QUESTION:
{query}

INSTRUCTIONS:
- Answer using the web search results above.
- Cite results using [W1], [W2] etc.
- Explain clearly — avoid heavy jargon.
- End with a disclaimer advising consultation with a qualified doctor.
"""
    else:
        # DuckDuckGo unavailable or returned nothing — plain LLM
        web_prompt = f"""You are a medical AI assistant.
Answer this medical question using your training knowledge.

PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

QUESTION:
{query}

INSTRUCTIONS:
- Provide a clear, accurate answer based on current medical knowledge.
- Avoid heavy jargon where possible.
- End with a disclaimer advising consultation with a qualified doctor.
"""
        web_sources = [
            "https://www.cancer.gov",
            "https://www.ncbi.nlm.nih.gov/pubmed/",
            "https://www.cancer.org",
        ]

    web_answer = ""
    try:
        response   = client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = [{"role": "user", "content": web_prompt}],
            temperature = GROQ_TEMP,
        )
        web_answer = response.choices[0].message.content or ""
    except Exception as e:
        web_answer = f"Could not generate web answer: {e}"

    # ── Combine RAG answer + web answer ───────────────────
    divider = "\n\n---\n\n🌐 **Web Search Result:**\n\n"
    combined = rag_answer.strip() + divider + web_answer.strip()

    # Sources are URLs (label = URL itself for clickable rendering)
    source_dicts = [{"label": u, "url": u} for u in web_sources]

    return combined, source_dicts, []   # no local figures for web results

# ==========================================================
# ================== MAIN: generate_answer =================
# ==========================================================

def generate_answer(
    query:          str,
    patient_report: str = "",
) -> tuple[str, list, list]:
    """
    Full pipeline → answer + sources + figures.

    Returns:
      answer  (str)        — LLM answer (may include web section)
      sources (list[dict]) — [{"label": display_name, "url": link_or_empty}]
                             RAG: {"label": "melanoma-review", "url": ""}
                             Web: {"label": "https://...",     "url": "https://..."}
      figures (list[dict]) — [{"caption":..., "image_path":..., "source":...}]
    """
    try:
        print(f"\n🔍 Query: {query[:70]}...")

        # ── Stage 1 + 2: Hybrid + MMR ─────────────────────
        retriever = get_hybrid_mmr_retriever()
        retrieved = retriever.invoke(query)

        if not retrieved:
            # No chunks at all — go straight to web search
            empty_rag = (
                "According to the provided clinical context, "
                "no relevant information was found for this query."
            )
            return _web_search_fallback(empty_rag, query, patient_report)

        # ── Separate text from figure captions ─────────────
        text_docs, figure_docs = split_text_and_figures(retrieved)

        # ── Build LLM context + RAG sources ───────────────
        context_text = build_context(text_docs)

        rag_sources: list[dict] = []
        seen: set = set()
        for doc in text_docs:
            # Try all key names ingestion may have used —
            # older ingestion runs may have written "source" or "file"
            # instead of "source_file". Fall through all options.
            sf = (
                doc.metadata.get("source_file")
                or doc.metadata.get("source")
                or doc.metadata.get("file")
                or doc.metadata.get("chunk_id", "").rsplit("_", 1)[0]  # e.g. "osteosarcoma-review_0042" → "osteosarcoma-review"
                or "unknown"
            )
            # source_url is stored in metadata by ingestion (SOURCE_URLS dict)
            url = doc.metadata.get("source_url", "")
            if sf not in seen:
                seen.add(sf)
                rag_sources.append({"label": sf, "url": url})

        # ── Figures (direct + figure-query detection) ──────
        figures = _collect_figures(figure_docs, text_docs, query)

        # ── LLM prompt ─────────────────────────────────────
        prompt = f"""You are an empathetic medical AI assistant helping \
cancer patients and clinicians understand medical information.

PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

CLINICAL CONTEXT (from peer-reviewed medical literature):
{context_text}

QUESTION:
{query}

INSTRUCTIONS:
- Answer using ONLY the clinical context provided above.
- Explain clearly — avoid heavy jargon where possible.
- Cite source numbers [1], [2] etc. when referencing specific facts.
- If the answer is not in the context, clearly state you do not have \
enough information in the clinical context.
- End with a disclaimer advising consultation with a qualified doctor.
"""

        # ── Groq LLM ───────────────────────────────────────
        client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model       = GROQ_MODEL,
            temperature = GROQ_TEMP,
            messages    = [{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content or ""

        # ── Web search fallback if RAG was insufficient ────
        if _rag_has_no_answer(answer):
            print("   ⚠️  RAG insufficient → web search fallback...")
            return _web_search_fallback(answer, query, patient_report)

        return answer, rag_sources, figures

    except Exception as e:
        return f"Error: {str(e)}", [], []

# ==========================================================
# ========================= DEMO ===========================
# ==========================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  cancer_retrieval.py — Hybrid + MMR + Web Fallback")
    print("=" * 62)

    tests = [
        "What is the standard treatment for osteosarcoma?",
        "What is Stage V melanoma survival rate?",
        "In the figure demonstrating melanoma growth phases, what are the two stages illustrated?",
    ]

    for q in tests:
        print(f"\n❓ {q}")
        answer, sources, figures = generate_answer(q)
        print(f"📝 {answer[:300]}...")
        print(f"📚 Sources: {[s['label'] for s in sources]}")
        print(f"🖼️  Figures: {len(figures)}")