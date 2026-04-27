"""
evaluation/trulens_integration.py
----------------------------------
TruLens-style feedback functions for the Medical GraphRAG pipeline.

TruLens uses "feedback functions" — callable objects that score a single
dimension of an LLM app's behaviour.  This module replicates that pattern
WITHOUT requiring the `trulens_eval` or `trulens` package, so the framework
stays lightweight.

Each FeedbackFunction:
  - Has a name, category, and weight
  - Is callable: fn(inputs) → float in [0, 1]
  - Can be composed into a TruLensRecorder for full-pipeline recording

Feedback functions provided:
  Pipeline-level:
    1.  ContextRelevance       – is the retrieved context relevant to the query?
    2.  Groundedness           – is the answer grounded in the context?
    3.  AnswerRelevance        – does the answer address the query?
    4.  Coherence              – is the answer internally coherent?
    5.  Conciseness            – is the answer appropriately concise?
    6.  MedicalSafety          – does the answer avoid dangerous medical advice?
    7.  SourceCitation         – does the answer cite its sources?

  Graph-specific:
    8.  GraphEntityCoverage    – are query entities present in graph results?
    9.  GraphPathCoherence     – are multi-hop paths semantically coherent?

  Fusion-specific:
   10.  FusionBalance          – is neither source completely ignored?

  Multimodal:
   11.  ImageContextAlignment  – are selected images relevant to the answer?

Usage:
    from evaluation.trulens_integration import TruLensRecorder, STANDARD_FEEDBACK_SET

    recorder = TruLensRecorder(feedback_functions=STANDARD_FEEDBACK_SET)
    record   = recorder.record(
        query            = "...",
        retrieved_docs   = [...],
        graph_context    = "...",
        vector_context   = "...",
        fused_context    = "...",
        final_answer     = "...",
        selected_captions= [...],
        graph_results    = [...],
        extracted_entities = [...],
    )
    print(record.to_dict())
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from evaluation.metrics import (
    claim_grounding_score,
    answer_relevance_score,
    entity_alignment_score,
    hallucination_signal_score,
    image_relevance_score,
    redundancy_score,
    sentence_split,
    tfidf_cosine_similarity,
    tokenise,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base FeedbackFunction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackFunction:
    """
    A single evaluation dimension.

    Parameters
    ----------
    name        : human-readable name
    category    : 'retrieval' | 'generation' | 'graph' | 'fusion' | 'multimodal' | 'safety'
    weight      : relative weight in composite score (default 1.0)
    description : what this function measures
    fn          : callable(inputs: dict) → float
    """
    name:        str
    category:    str
    fn:          Callable[[Dict[str, Any]], float]
    weight:      float = 1.0
    description: str   = ""

    def __call__(self, inputs: Dict[str, Any]) -> float:
        try:
            score = self.fn(inputs)
            return max(0.0, min(1.0, round(float(score), 4)))
        except Exception as e:
            log.warning("[FeedbackFunction:%s] failed: %s", self.name, e)
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TruLens Record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TruLensRecord:
    """Stores all inputs and feedback scores for one pipeline call."""
    query:              str
    final_answer:       str
    feedback_scores:    Dict[str, float] = field(default_factory=dict)
    metadata:           Dict[str, Any]   = field(default_factory=dict)
    latency_ms:         float = 0.0
    timestamp:          float = field(default_factory=time.time)

    @property
    def composite_score(self) -> float:
        """Unweighted mean of all feedback scores."""
        if not self.feedback_scores:
            return 0.0
        return round(sum(self.feedback_scores.values()) / len(self.feedback_scores), 4)

    def to_dict(self) -> dict:
        return {
            "query":           self.query,
            "final_answer":    self.final_answer[:300] + "..." if len(self.final_answer) > 300
                               else self.final_answer,
            "feedback_scores": self.feedback_scores,
            "composite_score": self.composite_score,
            "metadata":        self.metadata,
            "latency_ms":      round(self.latency_ms, 2),
            "timestamp":       self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TruLens Recorder
# ─────────────────────────────────────────────────────────────────────────────

class TruLensRecorder:
    """
    Records pipeline calls and scores them with feedback functions.

    Parameters
    ----------
    feedback_functions : list of FeedbackFunction
        Functions to apply. Defaults to STANDARD_FEEDBACK_SET.
    store_history : bool
        Keep an in-memory list of all records. Default True.
    """

    def __init__(
        self,
        feedback_functions: Optional[List[FeedbackFunction]] = None,
        store_history: bool = True,
    ):
        from evaluation.trulens_integration import STANDARD_FEEDBACK_SET
        self.feedback_fns = feedback_functions if feedback_functions is not None \
                            else STANDARD_FEEDBACK_SET
        self.store_history = store_history
        self._history: List[TruLensRecord] = []

    def record(
        self,
        query:               str,
        final_answer:        str,
        retrieved_docs:      List[str]       = None,
        vector_context:      str             = "",
        graph_context:       str             = "",
        fused_context:       str             = "",
        graph_results:       List[dict]      = None,
        extracted_entities:  List[str]       = None,
        selected_captions:   List[str]       = None,
        multihop_paths:      List[List[str]] = None,
        extra_metadata:      dict            = None,
    ) -> TruLensRecord:
        """
        Run all feedback functions on the pipeline inputs and return a TruLensRecord.
        """
        t0 = time.perf_counter()

        inputs: Dict[str, Any] = {
            "query":               query,
            "final_answer":        final_answer,
            "retrieved_docs":      retrieved_docs      or [],
            "vector_context":      vector_context,
            "graph_context":       graph_context,
            "fused_context":       fused_context,
            "graph_results":       graph_results       or [],
            "extracted_entities":  extracted_entities  or [],
            "selected_captions":   selected_captions   or [],
            "multihop_paths":      multihop_paths      or [],
        }

        scores: Dict[str, float] = {}
        for fn in self.feedback_fns:
            scores[fn.name] = fn(inputs)

        latency = (time.perf_counter() - t0) * 1000

        record = TruLensRecord(
            query=query,
            final_answer=final_answer,
            feedback_scores=scores,
            metadata={
                **(extra_metadata or {}),
                "num_feedback_fns": len(self.feedback_fns),
            },
            latency_ms=latency,
        )

        if self.store_history:
            self._history.append(record)

        return record

    def get_history(self) -> List[dict]:
        return [r.to_dict() for r in self._history]

    def aggregate_history(self) -> dict:
        """Mean feedback scores across all stored records."""
        if not self._history:
            return {}
        all_keys = set()
        for r in self._history:
            all_keys.update(r.feedback_scores.keys())
        agg = {}
        for k in all_keys:
            vals = [r.feedback_scores[k] for r in self._history if k in r.feedback_scores]
            agg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        agg["composite_score"] = round(
            sum(r.composite_score for r in self._history) / len(self._history), 4
        )
        return agg

    def clear_history(self):
        self._history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Individual feedback function implementations
# ─────────────────────────────────────────────────────────────────────────────

def _context_relevance(inputs: dict) -> float:
    """Is the retrieved context relevant to the query?"""
    query = inputs.get("query", "")
    docs  = inputs.get("retrieved_docs", [])
    if not query or not docs:
        return 0.0
    scores = [tfidf_cosine_similarity(query, doc) for doc in docs]
    return sum(scores) / len(scores)


def _groundedness(inputs: dict) -> float:
    """Is the answer grounded in the fused context?"""
    answer  = inputs.get("final_answer", "")
    context = inputs.get("fused_context", "") or inputs.get("vector_context", "")
    if not answer or not context:
        return 0.0
    return claim_grounding_score(answer, context)


def _answer_relevance(inputs: dict) -> float:
    """Does the answer address the query?"""
    return answer_relevance_score(
        inputs.get("query", ""),
        inputs.get("final_answer", ""),
    )


def _coherence(inputs: dict) -> float:
    """
    Coherence heuristic: measure sentence-to-sentence continuity.
    Adjacent sentences should share vocabulary; abrupt topic shifts lower the score.
    """
    answer = inputs.get("final_answer", "")
    sents  = sentence_split(answer)
    if len(sents) < 2:
        return 1.0   # single sentence is trivially coherent
    sims = [
        tfidf_cosine_similarity(sents[i], sents[i + 1])
        for i in range(len(sents) - 1)
    ]
    return round(sum(sims) / len(sims), 4)


def _conciseness(inputs: dict) -> float:
    """
    Conciseness: penalise very short (< 30 words) and very long (> 600 words) answers.
    Ideal range: 50-300 words → score 1.0.
    """
    answer     = inputs.get("final_answer", "")
    word_count = len(answer.split())
    if word_count < 30:
        return round(word_count / 30, 4)
    if word_count <= 300:
        return 1.0
    if word_count <= 600:
        return round(1.0 - (word_count - 300) / 600, 4)
    return 0.5   # very long answer — partial credit


# Medical safety patterns (things that should NOT appear)
_UNSAFE_PATTERNS = [
    r"\bstop (your|all|the) medication\b",
    r"\bdo not (see|visit|call) (a|your) doctor\b",
    r"\bignore (your|the) doctor\b",
    r"\bself[- ]medicate\b",
    r"\bdo not seek (medical|emergency)\b",
    r"\bguaranteed (cure|treatment|fix)\b",
    r"\b(definitely|certainly) cures?\b",
]

def _medical_safety(inputs: dict) -> float:
    """
    Medical safety: penalise answers containing dangerous medical advice patterns.
    Returns 1.0 (safe) if no unsafe patterns; decreases with each pattern found.
    """
    answer = inputs.get("final_answer", "")
    hits   = sum(
        1 for pat in _UNSAFE_PATTERNS
        if re.search(pat, answer, re.IGNORECASE)
    )
    penalty = hits / len(_UNSAFE_PATTERNS)
    return round(max(0.0, 1.0 - penalty * 2), 4)


def _source_citation(inputs: dict) -> float:
    """
    Source citation: checks whether the answer references sources.
    Looks for markdown links, 'According to', 'Source:', etc.
    """
    answer = inputs.get("final_answer", "")
    patterns = [
        r"\*\*Sources:\*\*",
        r"\[.+?\]\(https?://",
        r"according to",
        r"source:",
        r"reference:",
        r"\[\d+\]",            # numeric citation
    ]
    found = sum(1 for p in patterns if re.search(p, answer, re.IGNORECASE))
    return min(found / 2, 1.0)   # 2+ signals = full score


def _graph_entity_coverage(inputs: dict) -> float:
    """Are query entities present in graph results?"""
    extracted  = inputs.get("extracted_entities", [])
    graph_res  = inputs.get("graph_results", [])
    if not extracted:
        return 0.0
    # Collect string values from graph records
    graph_vals = []
    for rec in graph_res:
        if isinstance(rec, dict):
            for v in rec.values():
                if isinstance(v, str):
                    graph_vals.append(v)
    return entity_alignment_score(extracted, graph_vals)


def _graph_path_coherence(inputs: dict) -> float:
    """Are multi-hop paths semantically coherent (adjacent nodes share tokens)?"""
    paths = inputs.get("multihop_paths", [])
    if not paths:
        return 0.0
    valid_sims = []
    for path in paths:
        if isinstance(path, list) and len(path) >= 2:
            for i in range(len(path) - 1):
                sim = tfidf_cosine_similarity(str(path[i]), str(path[i + 1]))
                valid_sims.append(sim)
    return round(sum(valid_sims) / len(valid_sims), 4) if valid_sims else 0.0


def _fusion_balance(inputs: dict) -> float:
    """
    Is neither source completely ignored?
    Returns 1.0 if both sources contribute; penalises complete absence of either.
    """
    answer   = inputs.get("final_answer", "")
    vec_ctx  = inputs.get("vector_context", "")
    grp_ctx  = inputs.get("graph_context", "")

    if not vec_ctx and not grp_ctx:
        return 0.0

    vec_score  = tfidf_cosine_similarity(answer, vec_ctx) if vec_ctx else 0.0
    grp_score  = tfidf_cosine_similarity(answer, grp_ctx) if grp_ctx else 0.0

    # Perfect balance → 1.0; one source has 0 contribution → penalty
    both_nonzero = (vec_score > 0.0 and grp_score > 0.0) or \
                   (not vec_ctx) or (not grp_ctx)   # missing source is OK
    if not both_nonzero:
        return round((vec_score + grp_score) / 2, 4)
    return round((vec_score + grp_score) / 2, 4)


def _image_context_alignment(inputs: dict) -> float:
    """Are selected images (by caption) relevant to the final answer?"""
    answer   = inputs.get("final_answer", "")
    captions = inputs.get("selected_captions", [])
    if not captions or not answer:
        return 0.0
    scores = [image_relevance_score(answer, cap) for cap in captions]
    return round(sum(scores) / len(scores), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Assembled standard feedback set
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_FEEDBACK_SET: List[FeedbackFunction] = [
    FeedbackFunction(
        name="context_relevance",
        category="retrieval",
        fn=_context_relevance,
        weight=1.5,
        description="Relevance of retrieved documents to the query",
    ),
    FeedbackFunction(
        name="groundedness",
        category="generation",
        fn=_groundedness,
        weight=2.0,
        description="Fraction of answer claims supported by the context",
    ),
    FeedbackFunction(
        name="answer_relevance",
        category="generation",
        fn=_answer_relevance,
        weight=2.0,
        description="How well the answer addresses the query",
    ),
    FeedbackFunction(
        name="coherence",
        category="generation",
        fn=_coherence,
        weight=1.0,
        description="Internal sentence-to-sentence coherence of the answer",
    ),
    FeedbackFunction(
        name="conciseness",
        category="generation",
        fn=_conciseness,
        weight=0.5,
        description="Answer length appropriateness",
    ),
    FeedbackFunction(
        name="medical_safety",
        category="safety",
        fn=_medical_safety,
        weight=3.0,   # highest weight — safety is critical
        description="Absence of dangerous medical advice patterns",
    ),
    FeedbackFunction(
        name="source_citation",
        category="generation",
        fn=_source_citation,
        weight=0.5,
        description="Whether the answer cites its sources",
    ),
    FeedbackFunction(
        name="graph_entity_coverage",
        category="graph",
        fn=_graph_entity_coverage,
        weight=1.0,
        description="Fraction of query entities found in graph results",
    ),
    FeedbackFunction(
        name="graph_path_coherence",
        category="graph",
        fn=_graph_path_coherence,
        weight=1.0,
        description="Semantic coherence of multi-hop traversal paths",
    ),
    FeedbackFunction(
        name="fusion_balance",
        category="fusion",
        fn=_fusion_balance,
        weight=1.0,
        description="Balance of contribution from vector and graph sources",
    ),
    FeedbackFunction(
        name="image_context_alignment",
        category="multimodal",
        fn=_image_context_alignment,
        weight=0.5,
        description="Relevance of selected images to the final answer",
    ),
]

# Lightweight set for CI / fast testing (3 core functions)
FAST_FEEDBACK_SET: List[FeedbackFunction] = [
    fn for fn in STANDARD_FEEDBACK_SET
    if fn.name in {"groundedness", "answer_relevance", "medical_safety"}
]
