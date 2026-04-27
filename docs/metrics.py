"""
evaluation/metrics.py
---------------------
Core metric primitives for the Medical GraphRAG Evaluation Framework.

Covers:
  - Text similarity (TF-IDF cosine, token overlap, BLEU-like)
  - Precision / Recall / F1 over token sets
  - Normalized Discounted Cumulative Gain (nDCG)
  - Faithfulness score (claim-by-claim grounding)
  - Hallucination signal words
  - Redundancy (Jaccard, centroid distance)
  - Context completeness heuristics
  - Utility helpers (tokenise, normalise, chunk sentences)

All functions are pure-Python with numpy; no torch / transformers required
at import time so the module loads fast even in CI.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Text normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_STOP_WORDS: frozenset[str] = frozenset({
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","need","dare","ought","used","to","of",
    "in","on","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","from","up","down",
    "out","off","over","under","again","further","then","once","and","but",
    "or","nor","so","yet","both","either","neither","not","no","only",
    "own","same","than","too","very","just","because","as","until","while",
    "that","this","these","those","it","its","itself","they","them","their",
})

_PUNCT_TRANS = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return text.lower().translate(_PUNCT_TRANS).split().__str__()


def tokenise(text: str, remove_stops: bool = True) -> List[str]:
    """Tokenise *text* into a list of lowercase words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if remove_stops:
        tokens = [t for t in tokens if t not in _STOP_WORDS]
    return tokens


def sentence_split(text: str) -> List[str]:
    """Split *text* into sentences (simple heuristic)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Token-level overlap metrics
# ─────────────────────────────────────────────────────────────────────────────

def token_overlap_f1(prediction: str, reference: str) -> float:
    """
    Token-overlap F1 (SQuAD-style).
    Returns a score in [0, 1].
    """
    pred_tokens = Counter(tokenise(prediction))
    ref_tokens  = Counter(tokenise(reference))

    common = sum((pred_tokens & ref_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall    = common / sum(ref_tokens.values())
    f1        = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def token_precision(prediction: str, reference: str) -> float:
    """Fraction of prediction tokens found in the reference."""
    pred_tokens = tokenise(prediction)
    if not pred_tokens:
        return 0.0
    ref_set = set(tokenise(reference))
    return round(sum(1 for t in pred_tokens if t in ref_set) / len(pred_tokens), 4)


def token_recall(prediction: str, reference: str) -> float:
    """Fraction of reference tokens found in the prediction."""
    ref_tokens = tokenise(reference)
    if not ref_tokens:
        return 0.0
    pred_set = set(tokenise(prediction))
    return round(sum(1 for t in ref_tokens if t in pred_set) / len(ref_tokens), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TF-IDF cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def _tfidf_vector(doc_tokens: List[str], vocab: List[str], idf: np.ndarray) -> np.ndarray:
    tf = Counter(doc_tokens)
    total = max(len(doc_tokens), 1)
    vec   = np.array([tf.get(w, 0) / total for w in vocab], dtype=float)
    return vec * idf


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    TF-IDF weighted cosine similarity between two texts.
    Does NOT require sklearn; fully in-house.
    Returns score in [0, 1].
    """
    tokens_a = tokenise(text_a)
    tokens_b = tokenise(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    vocab = list(set(tokens_a + tokens_b))
    N     = 2   # two "documents"

    idf = np.array([
        math.log((N + 1) / (1 + sum(1 for tok_list in [tokens_a, tokens_b] if w in tok_list)))
        for w in vocab
    ])

    vec_a = _tfidf_vector(tokens_a, vocab, idf)
    vec_b = _tfidf_vector(tokens_b, vocab, idf)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return round(float(np.dot(vec_a, vec_b) / (norm_a * norm_b)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BLEU-like n-gram precision (simplified)
# ─────────────────────────────────────────────────────────────────────────────

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """
    Corpus-free sentence BLEU (no brevity penalty).
    Returns geometric mean of n-gram precisions for n=1..max_n.
    """
    hyp = tokenise(hypothesis, remove_stops=False)
    ref = tokenise(reference,  remove_stops=False)
    if not hyp or not ref:
        return 0.0

    log_sum = 0.0
    count   = 0
    for n in range(1, min(max_n, len(hyp), len(ref)) + 1):
        hyp_ng  = _ngrams(hyp, n)
        ref_ng  = _ngrams(ref, n)
        clipped = sum(min(c, ref_ng[ng]) for ng, c in hyp_ng.items())
        total   = sum(hyp_ng.values())
        if total == 0:
            continue
        prec = clipped / total
        if prec == 0:
            return 0.0
        log_sum += math.log(prec)
        count   += 1

    if count == 0:
        return 0.0
    return round(math.exp(log_sum / count), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Ranking quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    relevances = relevances[:k]
    return sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevances)
    )


def ndcg_at_k(relevances: List[float], k: int) -> float:
    """
    Normalised DCG at rank k.
    *relevances* is the actual ranked list; ideal is sorted descending.
    Returns a score in [0, 1].
    """
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return round(dcg_at_k(relevances, k) / ideal_dcg, 4)


def mean_reciprocal_rank(relevant_positions: List[int]) -> float:
    """
    MRR over a list of 1-based positions of relevant documents.
    E.g., [1, 3] means the first and third retrieved docs are relevant.
    """
    if not relevant_positions:
        return 0.0
    return round(1.0 / min(relevant_positions), 4)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """P@k: fraction of top-k retrieved docs that are relevant."""
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(hits / k, 4) if k > 0 else 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@k: fraction of relevant docs found in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(hits / len(relevant_ids), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Faithfulness / grounding metrics
# ─────────────────────────────────────────────────────────────────────────────

def claim_grounding_score(answer: str, context: str, threshold: float = 0.35) -> float:
    """
    RAGAS-inspired claim grounding.

    Split the answer into sentences (claims).
    For each claim, compute TF-IDF cosine similarity to the full context.
    A claim is "grounded" if similarity >= threshold.
    Return the fraction of grounded claims.

    This is a lightweight proxy — use LLM-based faithfulness for production.
    """
    claims = sentence_split(answer)
    if not claims:
        return 0.0

    grounded = sum(
        1 for claim in claims
        if tfidf_cosine_similarity(claim, context) >= threshold
    )
    return round(grounded / len(claims), 4)


_HALLUCINATION_SIGNAL_PATTERNS: List[str] = [
    r"\b(definitely|certainly|always|never|guaranteed|proven|100%)\b",
    r"\b(cure[sd]?|miraculous|breakthrough)\b",
    r"\b(i don'?t know|i'?m not sure|i cannot|i can'?t)\b",   # model confusion
    r"\b(as of \d{4})\b",                                       # stale date claims
]


def hallucination_signal_score(answer: str) -> float:
    """
    Heuristic hallucination signal in [0, 1].
    Higher → more hallucination signals found.
    0 = clean; 1 = heavily flagged.
    """
    flags = sum(
        1 for pat in _HALLUCINATION_SIGNAL_PATTERNS
        if re.search(pat, answer, re.IGNORECASE)
    )
    return round(min(flags / len(_HALLUCINATION_SIGNAL_PATTERNS), 1.0), 4)


def answer_relevance_score(query: str, answer: str) -> float:
    """
    Lightweight answer relevance: TF-IDF cosine between query and answer.
    Measures how much the answer talks about the same topic as the query.
    """
    return tfidf_cosine_similarity(query, answer)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Context precision / recall (RAGAS-style, text-based)
# ─────────────────────────────────────────────────────────────────────────────

def context_precision(
    retrieved_contexts: List[str],
    expected_answer:    str,
    threshold:          float = 0.25,
) -> float:
    """
    Fraction of retrieved contexts that are relevant to the expected answer.
    Relevance is determined by TF-IDF cosine >= threshold.
    """
    if not retrieved_contexts:
        return 0.0
    relevant = sum(
        1 for ctx in retrieved_contexts
        if tfidf_cosine_similarity(ctx, expected_answer) >= threshold
    )
    return round(relevant / len(retrieved_contexts), 4)


def context_recall(
    retrieved_contexts: List[str],
    expected_answer:    str,
    threshold:          float = 0.20,
) -> float:
    """
    Fraction of expected-answer sentences that are covered by any retrieved context.
    Proxy for: "does the retrieved set contain enough to answer the question?"
    """
    claims = sentence_split(expected_answer)
    if not claims:
        return 0.0

    combined_ctx = " ".join(retrieved_contexts)
    covered = sum(
        1 for claim in claims
        if tfidf_cosine_similarity(claim, combined_ctx) >= threshold
    )
    return round(covered / len(claims), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Redundancy metrics
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity on token sets."""
    set_a = set(tokenise(text_a))
    set_b = set(tokenise(text_b))
    if not set_a and not set_b:
        return 0.0
    return round(len(set_a & set_b) / len(set_a | set_b), 4)


def redundancy_score(contexts: List[str]) -> float:
    """
    Mean pairwise Jaccard similarity across all context pairs.
    High score → high redundancy.
    Returns 0.0 for fewer than 2 contexts.
    """
    if len(contexts) < 2:
        return 0.0

    total, count = 0.0, 0
    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            total += pairwise_jaccard(contexts[i], contexts[j])
            count += 1
    return round(total / count, 4) if count > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Graph-specific metrics
# ─────────────────────────────────────────────────────────────────────────────

def entity_alignment_score(
    extracted_entities: List[str],
    graph_entities:     List[str],
) -> float:
    """
    Fraction of query-extracted entities found in graph node results.
    Both lists should contain normalised canonical names.
    """
    if not extracted_entities:
        return 0.0
    graph_set = {e.lower() for e in graph_entities}
    hits = sum(1 for e in extracted_entities if e.lower() in graph_set)
    return round(hits / len(extracted_entities), 4)


def multihop_path_validity(paths: List[List[str]]) -> float:
    """
    Fraction of multi-hop paths (each path is a list of node labels) that
    are non-empty and contain at least 2 nodes (i.e. at least one hop).
    """
    if not paths:
        return 0.0
    valid = sum(1 for path in paths if isinstance(path, list) and len(path) >= 2)
    return round(valid / len(paths), 4)


def graph_context_density(graph_context: str) -> float:
    """
    Simple heuristic: proportion of unique medical tokens in graph context.
    Higher → richer graph context.
    """
    tokens = tokenise(graph_context)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    return round(unique_ratio, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Fusion contribution metrics
# ─────────────────────────────────────────────────────────────────────────────

def fusion_contribution(
    answer:         str,
    vector_context: str,
    graph_context:  str,
) -> dict[str, float]:
    """
    Estimate relative contribution of vector vs graph context to the answer.
    Returns dict with keys: vector_score, graph_score, dominant_source.

    "dominant_source" is 'vector', 'graph', or 'balanced'.
    """
    vec_score   = tfidf_cosine_similarity(answer, vector_context)
    graph_score = tfidf_cosine_similarity(answer, graph_context)

    if vec_score == 0 and graph_score == 0:
        dominant = "none"
    elif abs(vec_score - graph_score) < 0.05:
        dominant = "balanced"
    elif vec_score > graph_score:
        dominant = "vector"
    else:
        dominant = "graph"

    return {
        "vector_score":    round(vec_score, 4),
        "graph_score":     round(graph_score, 4),
        "dominant_source": dominant,
    }


def context_completeness(
    query:          str,
    vector_context: str,
    graph_context:  str,
) -> float:
    """
    Estimate how completely the fused context covers the query concepts.
    Score in [0, 1]: 1.0 = all query concepts covered.
    """
    query_tokens  = set(tokenise(query))
    if not query_tokens:
        return 0.0
    fused_tokens  = set(tokenise(vector_context + " " + graph_context))
    covered       = query_tokens & fused_tokens
    return round(len(covered) / len(query_tokens), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Multimodal metrics
# ─────────────────────────────────────────────────────────────────────────────

def image_relevance_score(
    query:   str,
    caption: str,
) -> float:
    """Relevance of an image (by its caption) to the query."""
    return tfidf_cosine_similarity(query, caption)


def image_selection_accuracy(
    selected_captions: List[str],
    relevant_captions: List[str],
    threshold:         float = 0.30,
) -> float:
    """
    Fraction of selected image captions that are relevant to at least one
    entry in the ground-truth relevant captions list.
    """
    if not selected_captions:
        return 0.0
    hits = sum(
        1 for sel in selected_captions
        if any(
            tfidf_cosine_similarity(sel, rel) >= threshold
            for rel in relevant_captions
        )
    )
    return round(hits / len(selected_captions), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Fallback evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def fallback_trigger_correct(
    had_fallback:      bool,
    vector_ctx_empty:  bool,
    graph_ctx_empty:   bool,
) -> bool:
    """
    Validate fallback trigger logic.
    Fallback should be triggered when BOTH contexts are empty, and NOT
    when at least one context has content.
    """
    expected = vector_ctx_empty and graph_ctx_empty
    return had_fallback == expected


def fallback_response_quality(
    query:             str,
    fallback_response: str,
) -> float:
    """
    Basic quality check for a fallback (web search) response.
    Uses answer relevance + a length heuristic (100-500 words is ideal).
    """
    relevance = answer_relevance_score(query, fallback_response)
    word_count = len(fallback_response.split())
    length_score = min(word_count / 200, 1.0) if word_count < 200 else max(1 - (word_count - 500) / 1000, 0.5)
    return round(0.6 * relevance + 0.4 * length_score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Aggregate helpers
# ─────────────────────────────────────────────────────────────────────────────

def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    """
    Compute a weighted average.
    Missing weights default to 1.0.
    """
    total_w, total_s = 0.0, 0.0
    for key, score in scores.items():
        w = weights.get(key, 1.0)
        total_s += score * w
        total_w += w
    return round(total_s / total_w, 4) if total_w > 0 else 0.0


def letter_grade(score: float) -> str:
    """Convert a [0,1] score to a letter grade."""
    if score >= 0.90: return "A"
    if score >= 0.75: return "B"
    if score >= 0.60: return "C"
    if score >= 0.45: return "D"
    return "F"
