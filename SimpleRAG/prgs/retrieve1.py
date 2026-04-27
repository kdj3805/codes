"""
retrieve.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enterprise Hybrid Retrieval + Answer Generation Pipeline — PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  - Hybrid retrieval (Dense + BM25 + RRF)
  - Severity-aware boost for escalation tables
  - Time-threshold-aware boost
  - Relevance filtering before LLM call
  - Structured JSON context assembly for table_row chunks
  - Multi-chunk synthesis with conflict detection
  - Never says severity not found when it exists in retrieved chunks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import re
import json
import math
from typing import List, Dict, Any, Tuple, Optional

import requests
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL:   str = "openai/gpt-oss-120b"

OLLAMA_URL:   str = "http://localhost:11434/api/embeddings"
EMBED_MODEL:  str = "nomic-embed-text"

CHROMA_PERSIST_DIR: str = "./chroma_store"
CHROMA_COLLECTION:  str = "enterprise_docs"

DENSE_FETCH_MULTIPLIER: int = 5
RRF_K: int = 60

BOOST_WEIGHT: float = 0.3

_BOOST_EXACT_BP    = 10.0
_BOOST_KEYWORD     = 5.0
_BOOST_TOKEN_OVL   = 3.0
_BOOST_SEVERITY    = 7.0
_BOOST_TIME_MATCH  = 5.0
_BOOST_MAX         = _BOOST_EXACT_BP


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def embed_query(query: str) -> List[float]:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": query},
            timeout=60
        )
        return resp.json()["embedding"]
    except Exception:
        return []


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
    return re.findall(r'\b\w+\b', text.lower())


def reciprocal_rank_fusion(
    dense_ranks: List[Tuple[int, float]],
    bm25_ranks:  List[Tuple[int, float]],
    k: int = RRF_K
) -> List[Tuple[int, float]]:
    rrf: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_ranks):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranks):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


def _extract_severity_from_query(query: str) -> Optional[str]:
    """Extract severity level from query."""
    query_lower = query.lower()
    severity_keywords = ["critical", "high", "medium", "low", "urgent", "severe"]
    for severity in severity_keywords:
        pattern = r'\b' + re.escape(severity) + r'\b'
        if re.search(pattern, query_lower):
            return severity.capitalize()
    return None


def _extract_time_from_query(query: str) -> List[str]:
    """Extract time thresholds from query (e.g., '4h', '24 hours')."""
    time_pattern = re.compile(
        r'(\d+)\s*(?:h|hr|hrs|hour|hours|minutes?|mins?|days?)',
        re.IGNORECASE
    )
    matches = time_pattern.findall(query)
    return [m for m in matches if m]


def heading_match_boost(query: str, chunk_heading: str, chunk_metadata: Dict[str, Any]) -> float:
    """
    Enhanced boost that includes severity and time matching.
    Returns raw boost value for display.
    """
    q_lower = query.lower()
    h_lower = chunk_heading.lower()
    
    total_boost = 0.0
    
    match = re.search(r'best\s+practice\s*#?(\d+)', q_lower, re.IGNORECASE)
    if match:
        query_num = match.group(1)
        if query_num in h_lower and "best practice" in h_lower:
            total_boost += _BOOST_EXACT_BP
    
    for kw in ["introduction", "recommendations", "maas360", "conclusion", "overview", "escalation"]:
        if kw in q_lower and kw in h_lower:
            total_boost += _BOOST_KEYWORD
            break
    
    q_tokens = set(tokenize_text(query))
    h_tokens = set(tokenize_text(chunk_heading))
    if q_tokens and len(q_tokens & h_tokens) / len(q_tokens) > 0.5:
        total_boost += _BOOST_TOKEN_OVL
    
    chunk_severity = chunk_metadata.get("severity", "")
    if chunk_severity:
        query_severity = _extract_severity_from_query(query)
        if query_severity and query_severity.lower() == chunk_severity.lower():
            total_boost += _BOOST_SEVERITY
    
    query_times = _extract_time_from_query(query)
    if query_times:
        raw_row_data_str = chunk_metadata.get("raw_row_data", "")
        if raw_row_data_str:
            try:
                row_data = json.loads(raw_row_data_str)
                escalation_stages = row_data.get("escalation_stages", [])
                for stage in escalation_stages:
                    stage_time = stage.get("time_threshold", "")
                    for query_time in query_times:
                        if query_time in stage_time:
                            total_boost += _BOOST_TIME_MATCH
                            break
            except Exception:
                pass
    
    return total_boost


def _apply_boost(rrf_score: float, raw_boost: float) -> float:
    """Multiplicative boost to prevent domination."""
    normalised = min(raw_boost / _BOOST_MAX, 1.0)
    return rrf_score * (1.0 + BOOST_WEIGHT * normalised)


def filter_relevant_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    max_chunks: int = 10
) -> List[Dict[str, Any]]:
    """
    Pre-LLM filter that removes chunks unlikely to contribute to the answer.
    Enhanced to check severity and time matches.
    """
    if not chunks:
        return []

    q_tokens = set(tokenize_text(query))
    query_severity = _extract_severity_from_query(query)
    query_times = _extract_time_from_query(query)
    
    kept: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            kept.append(chunk)
            continue

        ctype = chunk.get("chunk_type", "section")

        if ctype == "table_row":
            chunk_severity = chunk.get("severity", "")
            
            if query_severity and chunk_severity:
                if query_severity.lower() == chunk_severity.lower():
                    kept.append(chunk)
                    continue
            
            try:
                row_data: Dict[str, Any] = chunk.get("raw_row_data", {})
                if isinstance(row_data, str):
                    row_data = json.loads(row_data) if row_data else {}
                
                row_text_parts = [chunk_severity]
                escalation_stages = row_data.get("escalation_stages", [])
                for stage in escalation_stages:
                    row_text_parts.append(stage.get("time_threshold", ""))
                    row_text_parts.extend(stage.get("actions", []))
                
                row_text = " ".join(str(p) for p in row_text_parts).lower()
                row_tokens = set(tokenize_text(row_text))
                
                if q_tokens & row_tokens:
                    kept.append(chunk)
            except Exception:
                kept.append(chunk)

        else:
            if chunk.get("dense_score", 0) > 0.3 or chunk.get("bm25_score", 0) > 0:
                kept.append(chunk)

        if len(kept) >= max_chunks:
            break

    return kept


def hybrid_retrieve(
    query:         str,
    top_k:         int                 = 5,
    source_filter: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Full hybrid retrieval pipeline with severity and time-aware boost."""
    collection      = get_collection()
    query_embedding = embed_query(query)
    if not query_embedding:
        return []

    n_candidates = min(
        top_k * DENSE_FETCH_MULTIPLIER,
        max(collection.count(), 1)
    )

    where_filter: Optional[Dict[str, Any]] = None
    if source_filter and len(source_filter) == 1:
        where_filter = {"source_file": source_filter[0]}
    elif source_filter and len(source_filter) > 1:
        where_filter = {"source_file": {"$in": source_filter}}

    query_kwargs: Dict[str, Any] = dict(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
        include=["documents", "metadatas", "embeddings", "distances"]
    )
    if where_filter:
        query_kwargs["where"] = where_filter

    chroma_res = collection.query(**query_kwargs)

    cand_docs: List[str]          = chroma_res["documents"][0]
    cand_meta: List[Dict]         = chroma_res["metadatas"][0]
    cand_embs: List[List[float]]  = chroma_res["embeddings"][0]
    cand_ids:  List[str]          = chroma_res["ids"][0]

    if not cand_docs:
        return []

    dense_scores: List[Tuple[int, float]] = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(cand_embs)
    ]
    dense_ranked = sorted(dense_scores, key=lambda x: x[1], reverse=True)

    tokenized_corpus = [tokenize_text(doc) for doc in cand_docs]
    bm25             = BM25Okapi(tokenized_corpus)
    raw_bm25         = bm25.get_scores(tokenize_text(query))

    bm25_ranked = sorted(
        [(i, float(s)) for i, s in enumerate(raw_bm25)],
        key=lambda x: x[1], reverse=True
    )

    hybrid_scores = reciprocal_rank_fusion(dense_ranked, bm25_ranked, k=RRF_K)

    boosted: List[Tuple[int, float]] = []
    for local_idx, rrf_score in hybrid_scores:
        meta = cand_meta[local_idx]
        raw_boost  = heading_match_boost(query, meta.get("heading", ""), meta)
        final      = _apply_boost(rrf_score, raw_boost)
        boosted.append((local_idx, final))
    boosted.sort(key=lambda x: x[1], reverse=True)

    rrf_map: Dict[int, float] = dict(hybrid_scores)
    results: List[Dict[str, Any]] = []

    for local_idx, final_score in boosted[:top_k]:
        meta        = cand_meta[local_idx]
        dense_score = next((s for i, s in dense_ranked if i == local_idx), 0.0)
        bm25_score  = float(raw_bm25[local_idx])
        rrf_score   = rrf_map.get(local_idx, 0.0)
        raw_boost   = heading_match_boost(query, meta.get("heading", ""), meta)

        raw_row_data_str = meta.get("raw_row_data", "") or ""
        try:
            parsed_row_data = json.loads(raw_row_data_str) if raw_row_data_str else {}
        except Exception:
            parsed_row_data = {}

        table_headers_str = meta.get("table_headers", "") or ""
        try:
            parsed_table_headers = json.loads(table_headers_str) if table_headers_str else []
        except Exception:
            parsed_table_headers = []

        results.append({
            "chunk_id":        cand_ids[local_idx],
            "chunk_type":      meta.get("chunk_type", "section"),
            "heading":         meta.get("heading", ""),
            "content":         cand_docs[local_idx],
            "source_file":     meta.get("source_file", "unknown"),
            "page":            meta.get("page"),
            "dense_score":     round(dense_score, 4),
            "bm25_score":      round(bm25_score,  4),
            "hybrid_score":    round(rrf_score,    6),
            "heading_boost":   round(raw_boost,    4),
            "final_score":     round(final_score,  6),
            "raw_row_data":    parsed_row_data,
            "table_headers":   parsed_table_headers,
            "row_index":       meta.get("row_index", -1),
            "table_index":     meta.get("table_index", -1),
            "severity":        meta.get("severity", ""),
        })

    return results


def assemble_context(retrieved: List[Dict[str, Any]]) -> str:
    """Build LLM-ready context with structured JSON for table_row chunks."""
    parts = []
    for c in retrieved:
        ctype   = c.get("chunk_type", "section")
        heading = c.get("heading", "N/A")
        source  = c.get("source_file", "unknown")
        page    = c.get("page", "?")
        content = c["content"]

        if ctype == "table_row":
            row_data = c.get("raw_row_data", {})
            severity = c.get("severity", "")
            label = (
                f"[TABLE_ROW | Severity: {severity} | {heading} "
                f"| File: {source} | Page {page} "
                f"| Row {c.get('row_index', '?')}]"
            )
            structured_json = json.dumps(row_data, indent=2, ensure_ascii=False)
            parts.append(
                f"{label}\n"
                f"Structured Data:\n{structured_json}\n\n"
                f"Natural Language:\n{content}"
            )
        else:
            label = (
                f"[{ctype.upper()} | {heading} "
                f"| File: {source} | Page {page}]"
            )
            parts.append(f"{label}\n{content}")

    return "\n\n---\n\n".join(parts)


def generate_answer(
    query:        str,
    retrieved:    List[Dict[str, Any]],
    all_headings: Optional[List[str]] = None
) -> str:
    """
    Generate grounded LLM answer with enhanced instructions for escalation tables.
    CRITICAL: Never says severity not found when it exists in retrieved chunks.
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY is not set."

    q_lower = query.lower()

    if (
        all_headings
        and re.search(
            r'\b(main\s+sections?|all\s+sections?|headings?|topics?|table\s+of\s+contents)\b',
            q_lower
        )
    ):
        heading_list = "\n".join(f"- {h}" for h in all_headings)
        return f"The document(s) contain the following sections:\n\n{heading_list}"

    relevant = filter_relevant_chunks(query, retrieved, max_chunks=10)

    if not relevant:
        return "No relevant information found for this query."

    context = assemble_context(relevant)

    sources = sorted({c.get("source_file", "unknown") for c in relevant})

    has_table_rows = any(c.get("chunk_type") == "table_row" for c in relevant)
    multi_source   = len(sources) > 1
    multi_chunk    = len(relevant) > 1

    severities_in_context = {
        c.get("severity", "").lower()
        for c in relevant
        if c.get("chunk_type") == "table_row" and c.get("severity")
    }

    instructions: List[str] = [
        "You are an enterprise document assistant.",
        "",
        f"Context is drawn from: {', '.join(sources)}.",
        "",
        "CRITICAL RULES:",
        "- Use ONLY the provided context to answer.",
        "- Do NOT fabricate or infer values not present in the context.",
        "- Do NOT truncate mid-sentence.",
    ]

    if has_table_rows:
        instructions += [
            "",
            "ESCALATION TABLE RULES (apply whenever Structured Data JSON blocks are present):",
            "- Match the EXACT severity level from the question.",
            "- Match the EXACT time threshold from the question.",
            "- Do NOT mix data from different severity rows.",
            "- List ALL actions for the matched stage — do not omit any.",
            "- Reference the Row JSON field names verbatim when quoting values.",
        ]
        
        if severities_in_context:
            sev_list = ", ".join(sorted(severities_in_context))
            instructions.append(
                f"- The following severity levels are present in the context: {sev_list}. "
                f"Do NOT say a severity is 'not found' if it appears in this list."
            )

    if multi_chunk:
        instructions += [
            "",
            "MULTI-CHUNK RULES:",
            "- Combine information from all relevant chunks into a single coherent answer.",
            "- When information comes from different source files, cite the file name.",
            "- If two chunks contain conflicting values for the same field, explicitly "
            '  state: "Discrepancy found: [File A] says X, [File B] says Y."',
        ]

    if multi_source:
        instructions += [
            "",
            "MULTI-DOCUMENT RULES:",
            "- Attribute each piece of information to its source file.",
            "- If documents agree, summarise once and note agreement.",
            "- If documents disagree on the same topic, flag the conflict.",
        ]

    instructions += [
        "",
        '- If the answer is truly absent from the context, say exactly: "Information not found in document."',
    ]

    system_block = "\n".join(instructions)

    prompt = f"""{system_block}

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


def get_all_headings(source_filter: Optional[List[str]] = None) -> List[str]:
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