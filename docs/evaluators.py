"""
evaluation/evaluators.py
------------------------
Evaluator classes for each component of the Medical GraphRAG pipeline.

Classes:
    RetrievalEvaluator    – BM25 + dense + RRF + MMR output quality
    GenerationEvaluator   – LLM answer faithfulness, relevance, hallucination
    GraphEvaluator        – Neo4j traversal correctness, multi-hop, entity align
    FusionEvaluator       – Graph vs vector contribution, redundancy, completeness
    FallbackEvaluator     – Web-search fallback trigger & quality
    MultimodalEvaluator   – Image selection accuracy & caption relevance

Each evaluator:
  - Accepts structured inputs matching the live pipeline outputs
  - Returns a typed dict of scores in [0, 1] plus metadata
  - Is completely standalone (can be unit-tested independently)

Design note:
  LLM-based evaluation (faithfulness judge) is OPTIONAL and gated behind
  a use_llm=True flag so the framework runs in CI without API keys.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from evaluation.metrics import (
    answer_relevance_score,
    bleu_score,
    claim_grounding_score,
    context_completeness,
    context_precision,
    context_recall,
    entity_alignment_score,
    fallback_response_quality,
    fallback_trigger_correct,
    fusion_contribution,
    graph_context_density,
    hallucination_signal_score,
    image_relevance_score,
    image_selection_accuracy,
    multihop_path_validity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    redundancy_score,
    tfidf_cosine_similarity,
    token_overlap_f1,
    weighted_average,
    letter_grade,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Standardised result returned by every evaluator."""
    component:   str
    scores:      Dict[str, float]
    metadata:    Dict[str, Any] = field(default_factory=dict)
    grade:       str = ""
    latency_ms:  float = 0.0
    warnings:    List[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Mean of all score values."""
        if not self.scores:
            return 0.0
        return round(sum(self.scores.values()) / len(self.scores), 4)

    def to_dict(self) -> dict:
        return {
            "component":  self.component,
            "scores":     self.scores,
            "overall":    self.overall,
            "grade":      self.grade or letter_grade(self.overall),
            "metadata":   self.metadata,
            "latency_ms": round(self.latency_ms, 2),
            "warnings":   self.warnings,
        }


def _timed(fn):
    """Decorator that records wall-clock time on evaluator methods."""
    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        result: EvalResult = fn(self, *args, **kwargs)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.grade = letter_grade(result.overall)
        return result
    wrapper.__name__ = fn.__name__
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# 1. RetrievalEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Evaluates the hybrid retrieval stack:
        BM25 + dense embeddings → RRF → MMR → final retrieved docs

    Parameters
    ----------
    k : int
        Cutoff rank for P@k and R@k. Default 5.
    relevance_threshold : float
        TF-IDF cosine threshold for calling a doc "relevant". Default 0.25.
    """

    def __init__(self, k: int = 5, relevance_threshold: float = 0.25):
        self.k   = k
        self.thr = relevance_threshold

    @_timed
    def evaluate(
        self,
        query:              str,
        retrieved_docs:     List[str],          # page_content strings
        expected_answer:    str = "",           # ground-truth or reference
        relevant_doc_ids:   Optional[List[str]] = None,  # for P@k / R@k
        retrieved_doc_ids:  Optional[List[str]] = None,
        rrf_scores:         Optional[List[float]] = None,  # raw RRF fusion scores
    ) -> EvalResult:
        """
        Compute retrieval metrics.

        Scores returned:
            context_precision     – fraction of retrieved docs relevant to answer
            context_recall        – fraction of answer claims covered by retrieved docs
            redundancy            – mean pairwise Jaccard among retrieved docs
            ndcg_at_k             – NDCG@k (requires rrf_scores)
            precision_at_k        – P@k  (requires relevant/retrieved_doc_ids)
            recall_at_k           – R@k  (requires relevant/retrieved_doc_ids)
            query_coverage        – fraction of query tokens found in retrieved text
        """
        warnings = []

        # --- Context precision & recall (text-based) ---
        c_prec = context_precision(retrieved_docs, expected_answer, self.thr) \
                 if expected_answer else 0.0
        c_rec  = context_recall(retrieved_docs, expected_answer, self.thr) \
                 if expected_answer else 0.0

        if not expected_answer:
            warnings.append("expected_answer not provided; context_precision/recall set to 0")

        # --- Redundancy ---
        redund = redundancy_score(retrieved_docs)

        # --- nDCG (if rrf_scores available) ---
        ndcg = 0.0
        if rrf_scores and len(rrf_scores) >= 1:
            ndcg = ndcg_at_k(rrf_scores, self.k)
        else:
            warnings.append("rrf_scores not provided; ndcg_at_k set to 0")

        # --- P@k / R@k (if ID lists available) ---
        p_at_k = 0.0
        r_at_k = 0.0
        if relevant_doc_ids and retrieved_doc_ids:
            p_at_k = precision_at_k(retrieved_doc_ids, relevant_doc_ids, self.k)
            r_at_k = recall_at_k(retrieved_doc_ids, relevant_doc_ids, self.k)
        else:
            warnings.append("doc ID lists not provided; P@k and R@k set to 0")

        # --- Query coverage ---
        fused_text     = " ".join(retrieved_docs)
        query_coverage = tfidf_cosine_similarity(query, fused_text)

        scores = {
            "context_precision":  c_prec,
            "context_recall":     c_rec,
            "redundancy":         1.0 - redund,          # invert: lower redundancy = better
            "ndcg_at_k":          ndcg,
            "precision_at_k":     p_at_k,
            "recall_at_k":        r_at_k,
            "query_coverage":     query_coverage,
        }

        metadata = {
            "num_retrieved":     len(retrieved_docs),
            "k":                 self.k,
            "raw_redundancy":    redund,
        }

        return EvalResult(
            component="retrieval",
            scores=scores,
            metadata=metadata,
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. GenerationEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class GenerationEvaluator:
    """
    Evaluates the LLM-generated answer.

    Optionally uses a Groq LLM judge for faithfulness & relevance when
    `use_llm=True`.  Falls back to heuristic metrics when False or when
    the API call fails.

    Parameters
    ----------
    use_llm : bool
        Enable Groq-based LLM judge. Requires GROQ_API_KEY env var.
    model : str
        Groq model to use for judging. Default: llama-3.3-70b-versatile.
    groq_client : optional
        Pre-built groq.Groq() client (avoids re-instantiation in batch runs).
    """

    _FAITHFULNESS_PROMPT = """You are a strict medical fact-checker.

CONTEXT:
{context}

ANSWER:
{answer}

Task: Rate how faithfully the ANSWER is supported by the CONTEXT.
Respond with JSON only:
{{"faithfulness_score": <float 0.0-1.0>, "unsupported_claims": [<list of strings>]}}
where 0.0 = completely unsupported, 1.0 = fully supported.
Do NOT include any text outside the JSON."""

    _RELEVANCE_PROMPT = """You are a relevance evaluator for a medical Q&A system.

QUERY: {query}
ANSWER: {answer}

Task: Rate how directly and completely the ANSWER addresses the QUERY.
Respond with JSON only:
{{"relevance_score": <float 0.0-1.0>, "reason": "<one sentence>"}}
where 0.0 = irrelevant, 1.0 = perfectly relevant.
Do NOT include any text outside the JSON."""

    def __init__(
        self,
        use_llm:      bool = False,
        model:        str  = "llama-3.3-70b-versatile",
        groq_client=None,
    ):
        self.use_llm = use_llm
        self.model   = model
        self._client = groq_client

        if use_llm and groq_client is None:
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set")
                self._client = Groq(api_key=api_key)
            except Exception as e:
                log.warning("[GenerationEvaluator] LLM judge disabled: %s", e)
                self.use_llm = False

    def _llm_json(self, prompt: str) -> dict:
        """Call Groq and parse JSON response. Returns empty dict on failure."""
        import json
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            text = resp.choices[0].message.content.strip()
            # Extract JSON even if model adds surrounding text
            match = __import__("re").search(r"\{.*\}", text, __import__("re").DOTALL)
            return json.loads(match.group()) if match else {}
        except Exception as e:
            log.warning("[GenerationEvaluator] LLM call failed: %s", e)
            return {}

    @_timed
    def evaluate(
        self,
        query:           str,
        answer:          str,
        context:         str,
        reference:       str = "",
        is_web_fallback: bool = False,
    ) -> EvalResult:
        """
        Scores returned:
            faithfulness          – how grounded the answer is in the context
            answer_relevance      – how well the answer addresses the query
            hallucination_signal  – heuristic hallucination indicators (inverted)
            bleu                  – BLEU vs reference (0 if no reference)
            token_f1              – Token F1 vs reference (0 if no reference)
        """
        warnings = []

        # ── Heuristic scores (always computed) ──────────────────────────────
        h_faithfulness  = claim_grounding_score(answer, context)
        h_relevance     = answer_relevance_score(query, answer)
        h_hallucination = 1.0 - hallucination_signal_score(answer)  # inverted: higher = cleaner

        bleu   = bleu_score(answer, reference)    if reference else 0.0
        tok_f1 = token_overlap_f1(answer, reference) if reference else 0.0

        if not reference:
            warnings.append("reference not provided; bleu and token_f1 set to 0")

        if is_web_fallback:
            warnings.append("answer came from web fallback (not RAG context)")

        # ── LLM judge scores (optional) ─────────────────────────────────────
        llm_faithfulness = None
        llm_relevance    = None
        unsupported      = []

        if self.use_llm and context and answer:
            faith_prompt = self._FAITHFULNESS_PROMPT.format(
                context=context[:3000],
                answer=answer[:1500],
            )
            faith_resp = self._llm_json(faith_prompt)
            if "faithfulness_score" in faith_resp:
                llm_faithfulness = float(faith_resp["faithfulness_score"])
                unsupported = faith_resp.get("unsupported_claims", [])

            rel_prompt = self._RELEVANCE_PROMPT.format(
                query=query,
                answer=answer[:1500],
            )
            rel_resp = self._llm_json(rel_prompt)
            if "relevance_score" in rel_resp:
                llm_relevance = float(rel_resp["relevance_score"])

        # ── Merge heuristic + LLM scores ────────────────────────────────────
        faithfulness   = llm_faithfulness if llm_faithfulness is not None else h_faithfulness
        ans_relevance  = llm_relevance    if llm_relevance    is not None else h_relevance

        scores = {
            "faithfulness":         faithfulness,
            "answer_relevance":     ans_relevance,
            "hallucination_clean":  h_hallucination,
            "bleu":                 bleu,
            "token_f1":             tok_f1,
        }

        metadata = {
            "used_llm_judge":       self.use_llm and llm_faithfulness is not None,
            "heuristic_faithfulness": h_faithfulness,
            "heuristic_relevance":  h_relevance,
            "unsupported_claims":   unsupported,
            "answer_word_count":    len(answer.split()),
            "is_web_fallback":      is_web_fallback,
        }

        return EvalResult(
            component="generation",
            scores=scores,
            metadata=metadata,
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. GraphEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class GraphEvaluator:
    """
    Evaluates the Neo4j graph retrieval component.

    Parameters
    ----------
    min_density : float
        Minimum acceptable graph context density (unique token ratio). Default 0.3.
    """

    def __init__(self, min_density: float = 0.3):
        self.min_density = min_density

    @_timed
    def evaluate(
        self,
        query:              str,
        graph_context:      str,
        graph_results:      List[dict],         # raw records from graph_retrieval.py
        extracted_entities: List[str],          # from entity_extractor.py
        multihop_paths:     Optional[List[List[str]]] = None,  # paths as node-label lists
    ) -> EvalResult:
        """
        Scores returned:
            entity_alignment      – fraction of query entities found in graph results
            context_density       – unique token ratio in graph context
            traversal_coverage    – fraction of graph result types that returned data
            multihop_validity     – fraction of valid multi-hop paths
            graph_answer_overlap  – similarity of graph context to the query
        """
        warnings = []

        # ── Entity alignment ─────────────────────────────────────────────────
        # Collect all node values mentioned in graph results
        graph_entities: List[str] = []
        for rec in graph_results:
            for val in rec.values() if isinstance(rec, dict) else []:
                if isinstance(val, str):
                    graph_entities.append(val)

        e_align = entity_alignment_score(extracted_entities, graph_entities)

        # ── Context density ──────────────────────────────────────────────────
        density = graph_context_density(graph_context) if graph_context else 0.0
        if density < self.min_density:
            warnings.append(f"Graph context density {density:.2f} < threshold {self.min_density}")

        # ── Traversal coverage ───────────────────────────────────────────────
        # Count how many distinct query_type categories returned non-empty results
        non_empty = sum(
            1 for rec in graph_results
            if rec and any(v for v in rec.values() if isinstance(v, dict) and v.get("records"))
        )
        total_types = len(graph_results)
        traversal   = round(non_empty / total_types, 4) if total_types > 0 else 0.0

        # ── Multi-hop validity ───────────────────────────────────────────────
        mh_valid = multihop_path_validity(multihop_paths) if multihop_paths else 0.0
        if multihop_paths is None:
            warnings.append("multihop_paths not provided; multihop_validity set to 0")

        # ── Graph–query overlap ──────────────────────────────────────────────
        g_overlap = tfidf_cosine_similarity(query, graph_context) if graph_context else 0.0

        scores = {
            "entity_alignment":    e_align,
            "context_density":     density,
            "traversal_coverage":  traversal,
            "multihop_validity":   mh_valid,
            "graph_query_overlap": g_overlap,
        }

        metadata = {
            "num_graph_results":   len(graph_results),
            "num_entities_found":  len(graph_entities),
            "num_entities_queried": len(extracted_entities),
        }

        return EvalResult(
            component="graph",
            scores=scores,
            metadata=metadata,
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. FusionEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class FusionEvaluator:
    """
    Evaluates the context fusion step (fuse_contexts()).

    Measures relative contribution, redundancy between the two sources,
    and overall completeness of the fused context w.r.t. the query.
    """

    @_timed
    def evaluate(
        self,
        query:          str,
        answer:         str,
        vector_context: str,
        graph_context:  str,
        fused_context:  str,
    ) -> EvalResult:
        """
        Scores returned:
            vector_contribution   – how much the answer aligns with vector context
            graph_contribution    – how much the answer aligns with graph context
            cross_source_redundancy – Jaccard overlap between the two sources
            context_completeness  – fraction of query concepts in fused context
            fused_relevance       – similarity of fused context to query
        """
        warnings = []

        contrib   = fusion_contribution(answer, vector_context, graph_context)
        vec_score = contrib["vector_score"]
        grp_score = contrib["graph_score"]

        cross_redundancy = __import__("evaluation.metrics", fromlist=["pairwise_jaccard"]) \
                               .pairwise_jaccard(vector_context, graph_context) \
                           if vector_context and graph_context else 0.0

        completeness = context_completeness(query, vector_context, graph_context)
        fused_rel    = tfidf_cosine_similarity(query, fused_context) if fused_context else 0.0

        if not vector_context:
            warnings.append("vector_context is empty")
        if not graph_context:
            warnings.append("graph_context is empty")

        scores = {
            "vector_contribution":      vec_score,
            "graph_contribution":       grp_score,
            "cross_source_uniqueness":  1.0 - cross_redundancy,   # invert
            "context_completeness":     completeness,
            "fused_relevance":          fused_rel,
        }

        metadata = {
            "dominant_source":          contrib["dominant_source"],
            "raw_cross_redundancy":     cross_redundancy,
            "vector_context_chars":     len(vector_context),
            "graph_context_chars":      len(graph_context),
            "fused_context_chars":      len(fused_context),
        }

        return EvalResult(
            component="fusion",
            scores=scores,
            metadata=metadata,
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. FallbackEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class FallbackEvaluator:
    """
    Evaluates the DuckDuckGo web-search fallback mechanism.

    Checks:
      - Whether fallback was triggered when it should have been (logic correctness)
      - Quality of the fallback response
      - Whether the fallback contains the WEB_FALLBACK_MARKER
    """

    WEB_FALLBACK_MARKER = "<!-- WEB_FALLBACK_EMPTY_CONTEXT -->"

    @_timed
    def evaluate(
        self,
        query:             str,
        final_answer:      str,
        vector_ctx_empty:  bool,
        graph_ctx_empty:   bool,
    ) -> EvalResult:
        """
        Scores returned:
            trigger_correctness   – 1.0 if fallback logic is correct, 0.0 otherwise
            response_quality      – relevance + length heuristic
            marker_present        – 1.0 if WEB_FALLBACK_MARKER found when expected
        """
        had_fallback = self.WEB_FALLBACK_MARKER in final_answer

        # Extract the actual answer text (strip marker if present)
        answer_text = final_answer.replace(self.WEB_FALLBACK_MARKER, "").strip()

        trigger_ok   = fallback_trigger_correct(had_fallback, vector_ctx_empty, graph_ctx_empty)
        resp_quality = fallback_response_quality(query, answer_text) if had_fallback else 1.0

        # marker_present: 1.0 if both empty and marker IS present, or not-both-empty and no marker
        both_empty   = vector_ctx_empty and graph_ctx_empty
        marker_ok    = (both_empty and had_fallback) or (not both_empty and not had_fallback)

        scores = {
            "trigger_correctness": 1.0 if trigger_ok  else 0.0,
            "response_quality":    resp_quality,
            "marker_correctness":  1.0 if marker_ok   else 0.0,
        }

        warnings = []
        if not trigger_ok:
            if had_fallback and not both_empty:
                warnings.append("Fallback triggered even though context was available")
            elif not had_fallback and both_empty:
                warnings.append("Fallback was NOT triggered even though both contexts are empty")

        metadata = {
            "had_fallback":       had_fallback,
            "vector_ctx_empty":   vector_ctx_empty,
            "graph_ctx_empty":    graph_ctx_empty,
            "answer_word_count":  len(answer_text.split()),
        }

        return EvalResult(
            component="fallback",
            scores=scores,
            metadata=metadata,
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. MultimodalEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalEvaluator:
    """
    Evaluates the image selection step (select_best_images / build_visual_asset_section).

    Parameters
    ----------
    relevance_threshold : float
        Min TF-IDF cosine for an image to be considered "relevant". Default 0.20.
    """

    def __init__(self, relevance_threshold: float = 0.20):
        self.thr = relevance_threshold

    @_timed
    def evaluate(
        self,
        query:               str,
        selected_captions:   List[str],         # captions of images actually selected
        all_candidate_captions: List[str],      # captions of ALL images in the docs
        ground_truth_captions:  Optional[List[str]] = None,  # ideal image captions
    ) -> EvalResult:
        """
        Scores returned:
            image_query_relevance    – mean relevance of selected images to query
            selection_precision      – fraction of selected images that are relevant
            selection_recall         – fraction of ground-truth images selected (if GT given)
            caption_diversity        – 1 - redundancy among selected captions
        """
        warnings = []

        # ── Mean relevance to query ──────────────────────────────────────────
        if selected_captions:
            img_rel = sum(
                image_relevance_score(query, cap) for cap in selected_captions
            ) / len(selected_captions)
        else:
            img_rel = 0.0
            warnings.append("No images were selected")

        # ── Precision: how many selected are relevant? ───────────────────────
        # Use all_candidate_captions as a proxy "any candidate is relevant if
        # its caption overlaps with query"
        sel_precision = sum(
            1 for cap in selected_captions
            if tfidf_cosine_similarity(query, cap) >= self.thr
        ) / len(selected_captions) if selected_captions else 0.0

        # ── Recall: ground-truth based ───────────────────────────────────────
        sel_recall = 0.0
        if ground_truth_captions:
            sel_recall = image_selection_accuracy(
                selected_captions, ground_truth_captions, self.thr
            )
        else:
            warnings.append("ground_truth_captions not provided; selection_recall set to 0")

        # ── Caption diversity ────────────────────────────────────────────────
        cap_diversity = 1.0 - redundancy_score(selected_captions) if selected_captions else 0.0

        scores = {
            "image_query_relevance": round(img_rel, 4),
            "selection_precision":   round(sel_precision, 4),
            "selection_recall":      sel_recall,
            "caption_diversity":     round(cap_diversity, 4),
        }

        metadata = {
            "num_selected":          len(selected_captions),
            "num_candidates":        len(all_candidate_captions),
            "selected_captions":     selected_captions,
        }

        return EvalResult(
            component="multimodal",
            scores=scores,
            metadata=metadata,
            warnings=warnings,
        )
