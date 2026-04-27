"""
evaluation/feedbacks.py
-----------------------
Custom feedback (metric) functions for the Medical GraphRAG system.

Four domains:
  1. graph_correctness    — did the Neo4j traversal return relevant, well-structured results?
  2. fusion_quality       — does the fused context balance vector + graph contributions?
  3. fallback_correctness — did the DuckDuckGo fallback trigger at the right time?
  4. multimodal_relevance — are the selected images relevant to the query / answer?

All functions:
  • Accept plain Python types (str, list, dict, bool) — no ML framework deps
  • Return float in [0.0, 1.0]  (1.0 = best)
  • Are individually importable and unit-testable

Dependencies: only stdlib + numpy (already in most ML envs)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared text utilities  (self-contained — no metrics.py import needed)
# ─────────────────────────────────────────────────────────────────────────────

_STOP = frozenset({
    "a","an","the","is","are","was","were","in","on","at","to","of",
    "for","with","by","from","and","or","but","not","it","its",
})
_PUNCT = str.maketrans("", "", string.punctuation)


def _tokens(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop words."""
    return [w for w in text.lower().translate(_PUNCT).split() if w not in _STOP]


def _cosine(text_a: str, text_b: str) -> float:
    """TF-weighted cosine similarity — pure Python, no sklearn."""
    ta, tb = _tokens(text_a), _tokens(text_b)
    if not ta or not tb:
        return 0.0
    vocab   = list(set(ta + tb))
    ca, cb  = Counter(ta), Counter(tb)
    la, lb  = len(ta), len(tb)
    vec_a   = [ca[w] / la for w in vocab]
    vec_b   = [cb[w] / lb for w in vocab]
    dot     = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a  = math.sqrt(sum(x * x for x in vec_a))
    norm_b  = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return round(dot / (norm_a * norm_b), 4)


def _token_f1(pred: str, ref: str) -> float:
    """Token-overlap F1 (SQuAD-style)."""
    p_tok = Counter(_tokens(pred))
    r_tok = Counter(_tokens(ref))
    common = sum((p_tok & r_tok).values())
    if common == 0:
        return 0.0
    prec = common / sum(p_tok.values())
    rec  = common / sum(r_tok.values())
    return round(2 * prec * rec / (prec + rec), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Graph Correctness
# ─────────────────────────────────────────────────────────────────────────────

def graph_correctness(
    query:              str,
    graph_context:      str,
    graph_results:      List[Dict[str, Any]],
    extracted_entities: List[str],
) -> Dict[str, float]:
    """
    Measure how correct and useful the Neo4j graph retrieval was.

    Sub-scores:
      entity_hit_rate   — fraction of extracted entities found in graph results
      context_richness  — unique-token density of the graph context string
      query_alignment   — cosine similarity between query and graph context
      result_coverage   — fraction of query_types that returned non-empty records

    Parameters
    ----------
    query              : original user query
    graph_context      : formatted string returned by build_graph_context()
    graph_results      : list of raw GraphResult dicts  {query_type, entities, records}
    extracted_entities : canonical names from entity_extractor.py
                         e.g. ["Osteosarcoma", "Cisplatin"]

    Returns
    -------
    dict with individual sub-scores + composite "graph_correctness_score"
    """
    # ── Entity hit rate ──────────────────────────────────────────────────────
    if extracted_entities:
        all_graph_text = graph_context.lower()
        hits = sum(1 for e in extracted_entities if e.lower() in all_graph_text)
        entity_hit_rate = round(hits / len(extracted_entities), 4)
    else:
        entity_hit_rate = 0.0

    # ── Context richness (unique token ratio) ────────────────────────────────
    tokens = _tokens(graph_context)
    if tokens:
        context_richness = round(len(set(tokens)) / len(tokens), 4)
    else:
        context_richness = 0.0

    # ── Query alignment ──────────────────────────────────────────────────────
    query_alignment = _cosine(query, graph_context) if graph_context else 0.0

    # ── Result coverage (how many query types returned data) ─────────────────
    if graph_results:
        non_empty = sum(
            1 for r in graph_results
            if (isinstance(r, dict) and r.get("records"))
            or (hasattr(r, "records") and r.records)
        )
        result_coverage = round(non_empty / len(graph_results), 4)
    else:
        result_coverage = 0.0

    # ── Composite ────────────────────────────────────────────────────────────
    composite = round(
        0.35 * entity_hit_rate
        + 0.25 * context_richness
        + 0.25 * query_alignment
        + 0.15 * result_coverage,
        4,
    )

    return {
        "entity_hit_rate":           entity_hit_rate,
        "context_richness":          context_richness,
        "query_alignment":           query_alignment,
        "result_coverage":           result_coverage,
        "graph_correctness_score":   composite,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fusion Quality
# ─────────────────────────────────────────────────────────────────────────────

def fusion_quality(
    query:          str,
    final_answer:   str,
    vector_context: str,
    graph_context:  str,
    fused_context:  str,
) -> Dict[str, float]:
    """
    Evaluate how well the vector + graph contexts were fused.

    Sub-scores:
      vector_contribution   — cosine(answer, vector_context)
      graph_contribution    — cosine(answer, graph_context)
      source_balance        — penalises one source dominating completely
      query_completeness    — fraction of query tokens present in fused context
      fused_answer_overlap  — cosine(fused_context, answer)

    Returns
    -------
    dict with sub-scores + "fusion_quality_score"
    """
    vec_score  = _cosine(final_answer, vector_context) if vector_context else 0.0
    grp_score  = _cosine(final_answer, graph_context)  if graph_context  else 0.0

    # Source balance: both contributing is better than one dominating
    if vector_context and graph_context:
        # Perfect balance → 1.0; total imbalance → ~0.5
        balance = 1.0 - abs(vec_score - grp_score)
    elif vector_context or graph_context:
        balance = 0.7  # only one source available — partial credit
    else:
        balance = 0.0

    # Query completeness
    q_tokens   = set(_tokens(query))
    f_tokens   = set(_tokens(fused_context))
    query_completeness = round(len(q_tokens & f_tokens) / len(q_tokens), 4) if q_tokens else 0.0

    # Fused context → answer overlap
    fused_answer_overlap = _cosine(fused_context, final_answer) if fused_context else 0.0

    composite = round(
        0.20 * vec_score
        + 0.20 * grp_score
        + 0.25 * balance
        + 0.20 * query_completeness
        + 0.15 * fused_answer_overlap,
        4,
    )

    return {
        "vector_contribution":  round(vec_score, 4),
        "graph_contribution":   round(grp_score, 4),
        "source_balance":       round(balance, 4),
        "query_completeness":   query_completeness,
        "fused_answer_overlap": round(fused_answer_overlap, 4),
        "dominant_source":      "vector" if vec_score > grp_score + 0.05
                                else "graph" if grp_score > vec_score + 0.05
                                else "balanced",
        "fusion_quality_score": composite,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fallback Correctness
# ─────────────────────────────────────────────────────────────────────────────

# Marker injected by generate_answer_graphrag() when web fallback is used
_WEB_FALLBACK_MARKER = "<!-- WEB_FALLBACK_EMPTY_CONTEXT -->"


def fallback_correctness(
    query:          str,
    final_answer:   str,
    vector_context: str,
    graph_context:  str,
) -> Dict[str, float]:
    """
    Verify whether the web-search fallback was triggered correctly.

    Logic (mirrors graphrag_integration.py):
      • Fallback SHOULD trigger  → both contexts empty/insufficient
      • Fallback should NOT trigger → at least one context has content

    Sub-scores:
      trigger_accuracy    — 1.0 if trigger logic matches context availability
      response_relevance  — cosine(query, fallback_answer) when fallback fired
      marker_present      — 1.0 if WEB_FALLBACK_MARKER is correctly present/absent
      length_adequacy     — penalises very short fallback responses

    Returns
    -------
    dict with sub-scores + "fallback_correctness_score"
    """
    had_fallback   = _WEB_FALLBACK_MARKER in final_answer
    answer_text    = final_answer.replace(_WEB_FALLBACK_MARKER, "").strip()

    # Meaningful context check (mirrors _is_meaningful_context from pipeline)
    def _is_meaningful(ctx: str) -> bool:
        if not ctx or len(ctx.strip()) < 50:
            return False
        return len([l for l in ctx.strip().splitlines() if l.strip()]) >= 2

    has_vector  = _is_meaningful(vector_context)
    has_graph   = _is_meaningful(graph_context)
    both_empty  = not has_vector and not has_graph

    # Should fallback have fired?
    should_fallback = both_empty
    trigger_correct = had_fallback == should_fallback
    trigger_accuracy = 1.0 if trigger_correct else 0.0

    # Marker logic: present ↔ both empty
    marker_present = 1.0 if (both_empty and had_fallback) or (not both_empty and not had_fallback) \
                     else 0.0

    # Response relevance (only meaningful if fallback actually fired)
    if had_fallback and answer_text:
        response_relevance = _cosine(query, answer_text)
    elif not had_fallback:
        response_relevance = 1.0  # fallback didn't fire, not applicable — full credit
    else:
        response_relevance = 0.0

    # Length adequacy (fallback should return a non-trivial response)
    word_count = len(answer_text.split())
    if not had_fallback:
        length_adequacy = 1.0
    elif word_count >= 50:
        length_adequacy = 1.0
    else:
        length_adequacy = round(word_count / 50, 4)

    composite = round(
        0.40 * trigger_accuracy
        + 0.25 * response_relevance
        + 0.20 * marker_present
        + 0.15 * length_adequacy,
        4,
    )

    return {
        "had_fallback":              had_fallback,
        "should_have_fired":         should_fallback,
        "trigger_accuracy":          trigger_accuracy,
        "response_relevance":        round(response_relevance, 4),
        "marker_present":            marker_present,
        "length_adequacy":           length_adequacy,
        "fallback_correctness_score": composite,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multimodal Relevance
# ─────────────────────────────────────────────────────────────────────────────

def multimodal_relevance(
    query:                  str,
    final_answer:           str,
    selected_captions:      List[str],
    all_candidate_captions: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate how relevant the selected images are to the query and answer.

    Sub-scores:
      caption_query_relevance   — mean cosine(caption, query)    across selected images
      caption_answer_relevance  — mean cosine(caption, answer)   across selected images
      selection_precision       — fraction of selected captions with relevance ≥ threshold
      caption_diversity         — 1 - mean pairwise Jaccard  (avoids selecting duplicates)
      coverage_ratio            — selected / total candidates  (sanity check)

    Parameters
    ----------
    query                  : user query string
    final_answer           : full LLM-generated answer
    selected_captions      : captions of images actually rendered in the answer
    all_candidate_captions : captions of ALL images available in retrieved docs

    Returns
    -------
    dict with sub-scores + "multimodal_relevance_score"
    """
    if not selected_captions:
        return {
            "caption_query_relevance":   0.0,
            "caption_answer_relevance":  0.0,
            "selection_precision":       0.0,
            "caption_diversity":         0.0,
            "coverage_ratio":            0.0,
            "multimodal_relevance_score": 0.0,
            "note": "No images were selected",
        }

    # ── Per-caption relevance ────────────────────────────────────────────────
    RELEVANCE_THRESHOLD = 0.12   # minimum cosine to call an image "relevant"

    query_sims  = [_cosine(query,        cap) for cap in selected_captions]
    answer_sims = [_cosine(final_answer, cap) for cap in selected_captions]

    caption_query_relevance  = round(sum(query_sims)  / len(query_sims),  4)
    caption_answer_relevance = round(sum(answer_sims) / len(answer_sims), 4)

    relevant_count     = sum(1 for s in query_sims if s >= RELEVANCE_THRESHOLD)
    selection_precision = round(relevant_count / len(selected_captions), 4)

    # ── Caption diversity (avoid redundant selections) ───────────────────────
    if len(selected_captions) >= 2:
        jaccard_pairs = []
        for i in range(len(selected_captions)):
            for j in range(i + 1, len(selected_captions)):
                set_a = set(_tokens(selected_captions[i]))
                set_b = set(_tokens(selected_captions[j]))
                union = set_a | set_b
                jaccard = len(set_a & set_b) / len(union) if union else 0.0
                jaccard_pairs.append(jaccard)
        caption_diversity = round(1.0 - (sum(jaccard_pairs) / len(jaccard_pairs)), 4)
    else:
        caption_diversity = 1.0   # single image is trivially diverse

    # ── Coverage ratio ───────────────────────────────────────────────────────
    total_candidates = len(all_candidate_captions) if all_candidate_captions else len(selected_captions)
    coverage_ratio   = round(len(selected_captions) / total_candidates, 4) \
                       if total_candidates > 0 else 0.0

    composite = round(
        0.30 * caption_query_relevance
        + 0.25 * caption_answer_relevance
        + 0.25 * selection_precision
        + 0.20 * caption_diversity,
        4,
    )

    return {
        "caption_query_relevance":    caption_query_relevance,
        "caption_answer_relevance":   caption_answer_relevance,
        "selection_precision":        selection_precision,
        "caption_diversity":          caption_diversity,
        "coverage_ratio":             coverage_ratio,
        "num_images_selected":        len(selected_captions),
        "multimodal_relevance_score": composite,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all four custom feedbacks in one call
# ─────────────────────────────────────────────────────────────────────────────

def run_all_feedbacks(
    query:                  str,
    final_answer:           str,
    vector_context:         str,
    graph_context:          str,
    fused_context:          str,
    graph_results:          List[Dict[str, Any]],
    extracted_entities:     List[str],
    selected_captions:      List[str],
    all_candidate_captions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run all four custom feedback functions and return a combined report dict.

    Returns
    -------
    {
        "graph":      { ... graph sub-scores ... },
        "fusion":     { ... fusion sub-scores ... },
        "fallback":   { ... fallback sub-scores ... },
        "multimodal": { ... multimodal sub-scores ... },
        "composite":  <mean of the four top-level scores>
    }
    """
    graph_scores    = graph_correctness(
        query, graph_context, graph_results, extracted_entities
    )
    fusion_scores   = fusion_quality(
        query, final_answer, vector_context, graph_context, fused_context
    )
    fallback_scores = fallback_correctness(
        query, final_answer, vector_context, graph_context
    )
    mm_scores       = multimodal_relevance(
        query, final_answer, selected_captions, all_candidate_captions
    )

    top_scores = [
        graph_scores["graph_correctness_score"],
        fusion_scores["fusion_quality_score"],
        fallback_scores["fallback_correctness_score"],
        mm_scores["multimodal_relevance_score"],
    ]
    composite = round(sum(top_scores) / len(top_scores), 4)

    return {
        "graph":      graph_scores,
        "fusion":     fusion_scores,
        "fallback":   fallback_scores,
        "multimodal": mm_scores,
        "composite":  composite,
    }
