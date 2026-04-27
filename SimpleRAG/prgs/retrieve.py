"""
retrieve.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enterprise Hybrid Retrieval + Answer Generation Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles:
  - Loading the persistent ChromaDB collection
  - Embedding the user query via Ollama
  - Dense retrieval:  cosine similarity via ChromaDB HNSW index
  - Sparse retrieval: BM25 re-ranking over dense candidates
  - Fusion:           Reciprocal Rank Fusion (RRF, k=60)
  - Heading-aware boost (preserved from original logic)
  - Multi-source context assembly
  - LLM answer generation via Groq
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import re
import math
from typing import List, Dict, Any, Tuple, Optional

import requests
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi

load_dotenv()

# Disable ChromaDB telemetry to avoid KeyError race condition in Streamlit reruns
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL:   str = "openai/gpt-oss-120b"

OLLAMA_URL:   str = "http://localhost:11434/api/embeddings"
EMBED_MODEL:  str = "nomic-embed-text"

CHROMA_PERSIST_DIR: str = "./chroma_store"
CHROMA_COLLECTION:  str = "enterprise_docs"

# Pull this many candidates from the dense index before BM25 re-ranking.
# Higher = better recall at cost of BM25 speed (negligible at <10k chunks).
DENSE_FETCH_MULTIPLIER: int = 5

# Standard RRF constant from the original paper
RRF_K: int = 60


# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB
# ─────────────────────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    """Open the existing persistent collection (created by ingest.py)."""
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUERY EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def embed_query(query: str) -> List[float]:
    """Embed the query string via Ollama. Returns [] on failure."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": query},
            timeout=60
        )
        return resp.json()["embedding"]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS  (preserved from original)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a, b) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tokenize_text(text: str) -> List[str]:
    """
    BM25 tokenisation using re.findall(r'\\b\\w+\\b').
    Handles table cell content, hyphens, and numeric tokens that .split() misses.
    """
    return re.findall(r'\b\w+\b', text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# RRF  (preserved from original)
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_ranks: List[Tuple[int, float]],
    bm25_ranks:  List[Tuple[int, float]],
    k: int = RRF_K
) -> List[Tuple[int, float]]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.
    score(d) = Σ  1 / (k + rank(d) + 1)
    """
    rrf: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_ranks):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranks):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADING BOOST  (preserved from original)
# ─────────────────────────────────────────────────────────────────────────────

def heading_match_boost(query: str, chunk_heading: str) -> float:
    """
    Add deterministic score boost based on heading relevance:
      +10  exact numbered Best Practice match
      +5   keyword overlap (introduction, recommendations, maas360 …)
      +3   >50 % token overlap between query and heading
    """
    q_lower = query.lower()
    h_lower = chunk_heading.lower()

    # Numbered best-practice exact match
    match = re.search(r'best\s+practice\s*#?(\d+)', q_lower, re.IGNORECASE)
    if match:
        query_num = match.group(1)
        if query_num in h_lower and "best practice" in h_lower:
            return 10.0

    # Named-section keywords
    for kw in ["introduction", "recommendations", "maas360", "conclusion", "overview"]:
        if kw in q_lower and kw in h_lower:
            return 5.0

    # General token overlap
    q_tokens = set(tokenize_text(query))
    h_tokens = set(tokenize_text(chunk_heading))
    if q_tokens and len(q_tokens & h_tokens) / len(q_tokens) > 0.5:
        return 3.0

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_retrieve(
    query:         str,
    top_k:         int                  = 5,
    source_filter: Optional[List[str]]  = None
) -> List[Dict[str, Any]]:
    """
    Full hybrid retrieval pipeline:

      1. Dense  — pull DENSE_FETCH_MULTIPLIER×top_k candidates from ChromaDB
                  using the stored HNSW cosine index.
      2. Sparse — BM25 re-ranking over those candidates (fresh index each call;
                  fast at this scale, avoids stale index across sessions).
      3. Fuse   — Reciprocal Rank Fusion over dense + BM25 rankings.
      4. Boost  — Heading-aware deterministic boost (preserved from original).

    Args:
        query:         Natural language question.
        top_k:         Number of final results to return.
        source_filter: Optional list of source_file names to restrict search.

    Returns list of result dicts with content, metadata, and all scores.
    """
    collection     = get_collection()
    query_embedding = embed_query(query)
    if not query_embedding:
        return []

    n_candidates = min(
        top_k * DENSE_FETCH_MULTIPLIER,
        max(collection.count(), 1)
    )

    # ── Build optional ChromaDB where-filter ──────────────────────────────
    where_filter: Optional[Dict[str, Any]] = None
    if source_filter and len(source_filter) == 1:
        where_filter = {"source_file": source_filter[0]}
    elif source_filter and len(source_filter) > 1:
        where_filter = {"source_file": {"$in": source_filter}}

    # ── 1. Dense retrieval from ChromaDB ──────────────────────────────────
    query_kwargs: Dict[str, Any] = dict(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
        include=["documents", "metadatas", "embeddings", "distances"]
    )
    if where_filter:
        query_kwargs["where"] = where_filter

    chroma_res = collection.query(**query_kwargs)

    cand_docs:  List[str]             = chroma_res["documents"][0]
    cand_meta:  List[Dict]            = chroma_res["metadatas"][0]
    cand_embs:  List[List[float]]     = chroma_res["embeddings"][0]
    cand_ids:   List[str]             = chroma_res["ids"][0]

    if not cand_docs:
        return []

    # Recompute exact cosine scores (more reliable than ChromaDB distance vals)
    dense_scores: List[Tuple[int, float]] = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(cand_embs)
    ]
    dense_ranked = sorted(dense_scores, key=lambda x: x[1], reverse=True)

    # ── 2. BM25 re-ranking ────────────────────────────────────────────────
    tokenized_corpus = [tokenize_text(doc) for doc in cand_docs]
    bm25             = BM25Okapi(tokenized_corpus)
    raw_bm25         = bm25.get_scores(tokenize_text(query))

    bm25_ranked = sorted(
        [(i, float(s)) for i, s in enumerate(raw_bm25)],
        key=lambda x: x[1], reverse=True
    )

    # ── 3. RRF fusion ─────────────────────────────────────────────────────
    hybrid_scores = reciprocal_rank_fusion(dense_ranked, bm25_ranked, k=RRF_K)

    # ── 4. Heading boost ──────────────────────────────────────────────────
    boosted: List[Tuple[int, float]] = []
    for local_idx, rrf_score in hybrid_scores:
        boost      = heading_match_boost(query, cand_meta[local_idx].get("heading", ""))
        final      = rrf_score + boost
        boosted.append((local_idx, final))
    boosted.sort(key=lambda x: x[1], reverse=True)

    # ── 5. Assemble result dicts ──────────────────────────────────────────
    rrf_map: Dict[int, float] = dict(hybrid_scores)
    results: List[Dict[str, Any]] = []

    for local_idx, final_score in boosted[:top_k]:
        meta        = cand_meta[local_idx]
        dense_score = next((s for i, s in dense_ranked if i == local_idx), 0.0)
        bm25_score  = float(raw_bm25[local_idx])
        rrf_score   = rrf_map.get(local_idx, 0.0)
        boost       = heading_match_boost(query, meta.get("heading", ""))

        results.append({
            "chunk_id":      cand_ids[local_idx],
            "chunk_type":    meta.get("chunk_type", "section"),
            "heading":       meta.get("heading", ""),
            "content":       cand_docs[local_idx],
            "source_file":   meta.get("source_file", "unknown"),
            "page":          meta.get("page"),
            "dense_score":   round(dense_score, 4),
            "bm25_score":    round(bm25_score,  4),
            "hybrid_score":  round(rrf_score,   6),
            "heading_boost": round(boost,        4),
            "final_score":   round(final_score,  6),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def assemble_context(retrieved: List[Dict[str, Any]]) -> str:
    """
    Build an LLM-ready context string.
    Each block is labelled with type, heading, source file and page so the
    model can attribute answers accurately across multiple documents.
    """
    parts = []
    for c in retrieved:
        label = (
            f"[{c.get('chunk_type', 'section').upper()} "
            f"| {c.get('heading', 'N/A')} "
            f"| File: {c.get('source_file', 'unknown')} "
            f"| Page {c.get('page', '?')}]"
        )
        parts.append(f"{label}\n{c['content']}")
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    query:     str,
    retrieved: List[Dict[str, Any]],
    all_headings: Optional[List[str]] = None
) -> str:
    """
    Generate a grounded LLM answer using only the retrieved context.

    Preserves original logic:
      - TOC / headings queries → synthesised list (no LLM call needed)
      - All other queries      → Groq with strict context-only prompt
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY is not set."

    q_lower = query.lower()

    # Shortcut: "main sections / headings / topics" — no LLM needed
    if (
        all_headings
        and re.search(
            r'\b(main\s+sections?|all\s+sections?|headings?|topics?|table\s+of\s+contents)\b',
            q_lower
        )
    ):
        heading_list = "\n".join(f"- {h}" for h in all_headings)
        return f"The document(s) contain the following sections:\n\n{heading_list}"

    context = assemble_context(retrieved)

    # Tell the model which files are in scope
    sources = sorted({c.get("source_file", "unknown") for c in retrieved})

    prompt = f"""You are an enterprise document assistant.

The context below may come from multiple documents: {', '.join(sources)}.

Rules:
- Use ONLY the provided context to answer.
- If the answer spans multiple documents, reference the source file name when relevant.
- Do not truncate mid-sentence. If the answer is long, summarise the tail without losing detail.
- Do NOT say "Information not found" if any part of the answer appears in the context.
- If the answer is truly absent, say exactly: "Information not found in document."

Context:
{context}

Question:
{query}

Answer:"""

    groq_client = Groq(api_key=GROQ_API_KEY)
    response    = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_all_headings(source_filter: Optional[List[str]] = None) -> List[str]:
    """Return sorted unique section headings, optionally filtered by source files."""
    collection = get_collection()

    get_kwargs: Dict[str, Any] = {"include": ["metadatas"]}
    if source_filter and len(source_filter) == 1:
        get_kwargs["where"] = {"source_file": source_filter[0]}
    elif source_filter and len(source_filter) > 1:
        get_kwargs["where"] = {"source_file": {"$in": source_filter}}

    results  = collection.get(**get_kwargs)
    headings: set = set()
    for meta in (results.get("metadatas") or []):
        if meta and meta.get("chunk_type") == "section" and meta.get("heading"):
            headings.add(meta["heading"])
    return sorted(headings)