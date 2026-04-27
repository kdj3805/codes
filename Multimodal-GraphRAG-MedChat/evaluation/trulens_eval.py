"""
evaluation/trulens_eval.py
--------------------------
TruLens-style pipeline wrapper for the Medical GraphRAG system.

What this does:
  • Wraps generate_answer_graphrag() and captures every intermediate state
  • Runs 9 feedback functions on the captured trace
  • Saves a structured evaluation record (JSON)
  • Maintains a session history for batch analysis

Captured states per call:
  ┌─ query              original user question
  ├─ vector_context     Qdrant hybrid-MMR retrieved text
  ├─ graph_context      Neo4j multi-hop retrieved text
  ├─ fused_context      combined context sent to LLM
  ├─ final_answer       LLM-generated response
  ├─ sources            document source list
  ├─ had_fallback       True if DuckDuckGo web search was triggered
  └─ selected_captions  image captions embedded in the answer

Feedback functions (all return float in [0, 1]):
  Retrieval:
    1. context_relevance        — retrieved docs relevant to query
    2. context_diversity        — low redundancy among retrieved docs
  Generation:
    3. groundedness             — answer grounded in context
    4. answer_relevance         — answer addresses query
    5. medical_safety           — no dangerous advice patterns
  Graph:
    6. graph_entity_coverage    — query entities in graph results
  Fusion:
    7. source_balance           — neither source dominates
  Multimodal:
    8. image_answer_alignment   — selected images match answer
  Fallback:
    9. fallback_trigger_logic   — fallback fires only when needed

Usage:
    from evaluation.trulens_eval import TruLensEvaluator

    evaluator = TruLensEvaluator()
    record    = evaluator.evaluate(
        query          = "What are cisplatin side effects?",
        patient_report = "Patient has osteosarcoma ...",
    )
    print(record)

    # Batch analysis
    evaluator.print_session_summary()
    evaluator.save_session("results/session.json")

Standalone CLI:
    python trulens_eval.py --query "What treats osteosarcoma?" --patient-report ""
    python trulens_eval.py --session-file results/session.json --summary
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Pipeline fallback marker (from graphrag_integration.py)
_WEB_FALLBACK_MARKER = "<!-- WEB_FALLBACK_EMPTY_CONTEXT -->"


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities  (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

_STOP = frozenset({
    "a","an","the","is","are","was","were","in","on","at","to","of",
    "for","with","by","from","and","or","but","not","it","its",
})
_PUNCT = str.maketrans("", "", string.punctuation)


def _tok(text: str) -> List[str]:
    return [w for w in text.lower().translate(_PUNCT).split() if w not in _STOP]


def _cos(a: str, b: str) -> float:
    ta, tb = _tok(a), _tok(b)
    if not ta or not tb:
        return 0.0
    vocab  = list(set(ta + tb))
    ca, cb = Counter(ta), Counter(tb)
    la, lb = len(ta), len(tb)
    dot    = sum((ca[w] / la) * (cb[w] / lb) for w in vocab)
    na     = math.sqrt(sum((ca[w] / la) ** 2 for w in vocab))
    nb     = math.sqrt(sum((cb[w] / lb) ** 2 for w in vocab))
    return round(dot / (na * nb), 4) if na and nb else 0.0


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline trace  (what we capture from one generate_answer_graphrag() call)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineTrace:
    """
    Holds all intermediate states from a single pipeline call.
    Populated by TruLensEvaluator._run_pipeline().
    """

    def __init__(self, query: str):
        self.query:              str             = query
        self.patient_report:     str             = ""
        self.vector_context:     str             = ""
        self.graph_context:      str             = ""
        self.fused_context:      str             = ""
        self.final_answer:       str             = ""
        self.sources:            List[dict]      = []
        self.had_fallback:       bool            = False
        self.selected_captions:  List[str]       = []
        self.extracted_entities: List[str]       = []
        self.graph_results:      List[Any]       = []
        self.latency_ms:         float           = 0.0
        self.error:              Optional[str]   = None
        self.timestamp:          float           = time.time()

    def to_dict(self) -> dict:
        return {
            "query":              self.query,
            "patient_report":     self.patient_report[:200] if self.patient_report else "",
            "vector_context_len": len(self.vector_context),
            "graph_context_len":  len(self.graph_context),
            "fused_context_len":  len(self.fused_context),
            "final_answer_len":   len(self.final_answer),
            "had_fallback":       self.had_fallback,
            "num_sources":        len(self.sources),
            "num_images":         len(self.selected_captions),
            "num_entities":       len(self.extracted_entities),
            "latency_ms":         self.latency_ms,
            "error":              self.error,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feedback Functions
# ─────────────────────────────────────────────────────────────────────────────

# Medical safety: patterns that should NOT appear in a medical answer
_UNSAFE_PATTERNS = [
    r"\bstop (your|all|the) (medication|treatment|chemo)\b",
    r"\bdo not (see|visit|consult|call) (a|your|the)? ?(doctor|physician|oncologist)\b",
    r"\bignore (your|the) (doctor|physician|specialist)\b",
    r"\bself[- ]medicate\b",
    r"\b(guaranteed|100%) (cure|treatment|fix|solution)\b",
    r"\bdefinitely cures?\b",
]


def fb_context_relevance(trace: PipelineTrace) -> float:
    """Mean cosine similarity between query and each retrieved document."""
    docs = [
        chunk for chunk in trace.fused_context.split("\n\n")
        if len(chunk.strip()) > 30
    ]
    if not docs:
        return 0.0
    sims = [_cos(trace.query, doc) for doc in docs]
    return round(sum(sims) / len(sims), 4)


def fb_context_diversity(trace: PipelineTrace) -> float:
    """
    Low redundancy among context chunks → high diversity.
    Returns 1 - mean pairwise Jaccard.
    """
    docs = [
        chunk for chunk in trace.fused_context.split("\n\n")
        if len(chunk.strip()) > 30
    ]
    if len(docs) < 2:
        return 1.0
    pairs, total = 0, 0.0
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sa = set(_tok(docs[i]))
            sb = set(_tok(docs[j]))
            u  = sa | sb
            total += len(sa & sb) / len(u) if u else 0.0
            pairs += 1
    return round(1.0 - (total / pairs), 4) if pairs else 1.0


def fb_groundedness(trace: PipelineTrace) -> float:
    """
    Fraction of answer sentences grounded in the fused context.
    Uses TF-IDF cosine ≥ 0.20 as the grounding threshold.
    """
    context = trace.fused_context or trace.vector_context
    if not context or not trace.final_answer:
        return 0.0
    claims  = _sentences(trace.final_answer)
    if not claims:
        return 0.0
    THRESHOLD = 0.20
    grounded  = sum(1 for c in claims if _cos(c, context) >= THRESHOLD)
    return round(grounded / len(claims), 4)


def fb_answer_relevance(trace: PipelineTrace) -> float:
    """Direct cosine similarity between query and answer."""
    return _cos(trace.query, trace.final_answer)


def fb_medical_safety(trace: PipelineTrace) -> float:
    """
    Penalise answers containing dangerous medical advice patterns.
    Returns 1.0 (safe) − 0.2 per pattern found.
    """
    hits = sum(
        1 for pat in _UNSAFE_PATTERNS
        if re.search(pat, trace.final_answer, re.IGNORECASE)
    )
    return round(max(0.0, 1.0 - hits * 0.2), 4)


def fb_graph_entity_coverage(trace: PipelineTrace) -> float:
    """Fraction of extracted entities found anywhere in the graph context."""
    if not trace.extracted_entities:
        return 0.0
    ctx_lower = trace.graph_context.lower()
    hits = sum(1 for e in trace.extracted_entities if e.lower() in ctx_lower)
    return round(hits / len(trace.extracted_entities), 4)


def fb_source_balance(trace: PipelineTrace) -> float:
    """
    Penalise scenarios where one context source is completely dominant.
    Both contexts contributing → 1.0.
    Only one context available → 0.7 (not penalised for missing source).
    One source present but contributing nothing → 0.4.
    """
    has_vec = bool(trace.vector_context and len(trace.vector_context.strip()) > 50)
    has_grp = bool(trace.graph_context  and len(trace.graph_context.strip())  > 50)

    if not has_vec and not has_grp:
        return 0.0

    if not has_vec or not has_grp:
        return 0.7   # only one source available — expected

    vec_score = _cos(trace.final_answer, trace.vector_context)
    grp_score = _cos(trace.final_answer, trace.graph_context)

    if vec_score == 0 and grp_score == 0:
        return 0.3   # both present but neither contributed
    if vec_score == 0 or grp_score == 0:
        return 0.5   # one source ignored despite being available
    return round(1.0 - abs(vec_score - grp_score), 4)


def fb_image_answer_alignment(trace: PipelineTrace) -> float:
    """Mean cosine similarity between selected image captions and the answer."""
    if not trace.selected_captions:
        return 0.5   # no images selected — neutral (some queries don't need images)
    sims = [_cos(cap, trace.final_answer) for cap in trace.selected_captions]
    return round(sum(sims) / len(sims), 4)


def fb_fallback_trigger_logic(trace: PipelineTrace) -> float:
    """
    1.0 if the fallback trigger decision matches context availability.
    Logic from graphrag_integration.py:
      both contexts empty/insufficient → fallback SHOULD fire
      at least one context meaningful  → fallback should NOT fire
    """
    def _meaningful(ctx: str) -> bool:
        if not ctx or len(ctx.strip()) < 50:
            return False
        return len([l for l in ctx.strip().splitlines() if l.strip()]) >= 2

    should_fire = not _meaningful(trace.vector_context) and not _meaningful(trace.graph_context)
    correct     = trace.had_fallback == should_fire
    return 1.0 if correct else 0.0


# Registry: name → (function, weight, category)
_FEEDBACK_REGISTRY: List[Tuple[str, Any, float, str]] = [
    ("context_relevance",     fb_context_relevance,     1.5, "retrieval"),
    ("context_diversity",     fb_context_diversity,     1.0, "retrieval"),
    ("groundedness",          fb_groundedness,          2.0, "generation"),
    ("answer_relevance",      fb_answer_relevance,      2.0, "generation"),
    ("medical_safety",        fb_medical_safety,        3.0, "safety"),
    ("graph_entity_coverage", fb_graph_entity_coverage, 1.0, "graph"),
    ("source_balance",        fb_source_balance,        1.0, "fusion"),
    ("image_answer_alignment",fb_image_answer_alignment,0.5, "multimodal"),
    ("fallback_trigger_logic",fb_fallback_trigger_logic,1.5, "fallback"),
]


def run_feedback_functions(trace: PipelineTrace) -> Dict[str, float]:
    """Execute all registered feedback functions on a pipeline trace."""
    scores = {}
    for name, fn, _, _ in _FEEDBACK_REGISTRY:
        try:
            scores[name] = fn(trace)
        except Exception as exc:
            log.warning("[Feedback:%s] failed: %s", name, exc)
            scores[name] = 0.0
    return scores


def weighted_composite(scores: Dict[str, float]) -> float:
    """Weighted average of feedback scores using registry weights."""
    weight_map = {name: w for name, _, w, _ in _FEEDBACK_REGISTRY}
    total_w, total_s = 0.0, 0.0
    for name, score in scores.items():
        w = weight_map.get(name, 1.0)
        total_s += score * w
        total_w += w
    return round(total_s / total_w, 4) if total_w > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TruLensEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class TruLensEvaluator:
    """
    Wraps the Medical GraphRAG pipeline and evaluates each call.

    Tries to import and call generate_answer_graphrag() from graphrag_integration.py.
    Falls back gracefully to offline mode if the pipeline is unavailable
    (e.g., during CI, no Qdrant/Neo4j connection) so the evaluation
    framework can still be tested independently.

    Parameters
    ----------
    pipeline_available : bool
        Set False to force offline mode (useful for unit tests).
    store_history : bool
        Keep all records in memory. Default True.
    """

    def __init__(
        self,
        pipeline_available: bool = True,
        store_history:      bool = True,
    ):
        self._pipeline_ok = pipeline_available
        self._store       = store_history
        self._history:  List[dict] = []

        if pipeline_available:
            self._pipeline_ok = self._check_pipeline()

    def _check_pipeline(self) -> bool:
        """Try importing the pipeline module; warn and continue if unavailable."""
        try:
            import importlib
            importlib.import_module("graphrag_integration")
            log.info("[TruLens] Pipeline (graphrag_integration) connected ✓")
            return True
        except ImportError:
            log.warning(
                "[TruLens] graphrag_integration not importable — running in offline mode. "
                "Pass a pre-built PipelineTrace to evaluate_trace() to score existing data."
            )
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Live pipeline call
    # ──────────────────────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        query:          str,
        patient_report: str = "",
        cancer_filter:  str = "",
    ) -> PipelineTrace:
        """
        Call generate_answer_graphrag() and capture all intermediate states.

        We extract intermediate context by also calling the retrieval and
        graph modules directly (they are idempotent reads, not writes).
        """
        trace                = PipelineTrace(query)
        trace.patient_report = patient_report
        t0                   = time.perf_counter()

        try:
            # ── Step 1: Vector retrieval (capture context) ───────────────────
            try:
                from Cancer_retrieval_v2_visual import (
                    get_hybrid_mmr_retriever,
                    build_context,
                    _extract_sources,
                )
                retriever    = get_hybrid_mmr_retriever(cancer_filter=cancer_filter)
                retrieved    = retriever.invoke(query)
                trace.vector_context = build_context(retrieved, query) if retrieved else ""
                trace.sources        = _extract_sources(retrieved)     if retrieved else []
            except Exception as exc:
                log.warning("[TruLens] Vector retrieval failed: %s", exc)

            # ── Step 2: Graph retrieval (capture context + entities) ──────────
            try:
                from entity_extractor import extract_entities
                from graph_retrieval  import query_graph, build_graph_context

                entities             = extract_entities(query, patient_report)
                trace.extracted_entities = (
                    entities.cancers + entities.drugs +
                    entities.side_effects + entities.foods
                )
                graph_results         = query_graph(entities)
                trace.graph_results   = [
                    {"query_type": r.query_type, "records": r.records}
                    for r in graph_results
                ]
                trace.graph_context   = build_graph_context(graph_results)
            except Exception as exc:
                log.warning("[TruLens] Graph retrieval failed: %s", exc)

            # ── Step 3: Generate answer (the full pipeline call) ─────────────
            from graphrag_integration import generate_answer_graphrag, fuse_contexts

            answer, sources, _ = generate_answer_graphrag(
                query          = query,
                patient_report = patient_report,
                cancer_filter  = cancer_filter,
            )

            trace.final_answer = answer
            trace.had_fallback = _WEB_FALLBACK_MARKER in answer

            # ── Step 4: Fused context ─────────────────────────────────────────
            trace.fused_context = fuse_contexts(trace.vector_context, trace.graph_context)

            # ── Step 5: Extract image captions from answer ────────────────────
            trace.selected_captions = re.findall(r"\[IMAGE:\s*([^\]]+)\]", answer)

        except Exception as exc:
            trace.error = str(exc)
            log.error("[TruLens] Pipeline call failed: %s", exc)

        trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return trace

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation entry points
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        query:          str,
        patient_report: str = "",
        cancer_filter:  str = "",
    ) -> dict:
        """
        Run the full pipeline and evaluate the result.

        Returns a structured record dict with:
          - trace metadata
          - feedback scores (one float per function)
          - composite score
          - grade
        """
        if not self._pipeline_ok:
            raise RuntimeError(
                "Pipeline not available. Use evaluate_trace(trace) to score "
                "a pre-built PipelineTrace, or set pipeline_available=True."
            )
        trace  = self._run_pipeline(query, patient_report, cancer_filter)
        return self._score_and_record(trace)

    def evaluate_trace(self, trace: PipelineTrace) -> dict:
        """
        Score a pre-built PipelineTrace (offline mode, unit-testing, replays).
        """
        return self._score_and_record(trace)

    def _score_and_record(self, trace: PipelineTrace) -> dict:
        """Run feedback functions and build the evaluation record."""
        scores    = run_feedback_functions(trace)
        composite = weighted_composite(scores)

        # Grade
        if composite >= 0.85: grade = "A"
        elif composite >= 0.70: grade = "B"
        elif composite >= 0.55: grade = "C"
        elif composite >= 0.40: grade = "D"
        else: grade = "F"

        record = {
            "timestamp":       trace.timestamp,
            "query":           trace.query,
            "latency_ms":      trace.latency_ms,
            "had_fallback":    trace.had_fallback,
            "error":           trace.error,
            "trace_metadata":  trace.to_dict(),
            "feedback_scores": scores,
            "composite_score": composite,
            "grade":           grade,
            # Category breakdowns
            "by_category": _scores_by_category(scores),
        }

        if self._store:
            self._history.append(record)

        return record

    # ──────────────────────────────────────────────────────────────────────
    # Session management
    # ──────────────────────────────────────────────────────────────────────

    def session_summary(self) -> dict:
        """Aggregate all feedback scores across stored records."""
        if not self._history:
            return {"n_records": 0, "message": "No records in history"}

        keys    = list(self._history[0]["feedback_scores"].keys())
        totals  = {k: [] for k in keys}
        for rec in self._history:
            for k in keys:
                v = rec["feedback_scores"].get(k)
                if v is not None:
                    totals[k].append(v)

        mean_scores = {k: round(sum(v) / len(v), 4) for k, v in totals.items() if v}
        composites  = [r["composite_score"] for r in self._history]

        return {
            "n_records":        len(self._history),
            "mean_composite":   round(sum(composites) / len(composites), 4),
            "mean_scores":      mean_scores,
            "fallback_rate":    round(
                sum(1 for r in self._history if r["had_fallback"]) / len(self._history), 4
            ),
            "error_rate":       round(
                sum(1 for r in self._history if r.get("error")) / len(self._history), 4
            ),
            "mean_latency_ms":  round(
                sum(r["latency_ms"] for r in self._history) / len(self._history), 2
            ),
        }

    def print_session_summary(self):
        summary = self.session_summary()
        print("\n── TruLens Session Summary ─────────────────────────")
        print(f"  Records:          {summary.get('n_records', 0)}")
        print(f"  Mean composite:   {summary.get('mean_composite', 0):.4f}")
        print(f"  Fallback rate:    {summary.get('fallback_rate', 0):.2%}")
        print(f"  Error rate:       {summary.get('error_rate', 0):.2%}")
        print(f"  Mean latency ms:  {summary.get('mean_latency_ms', 0):.1f}")
        print("  ── Feedback scores ──────────────────────────────")
        for k, v in (summary.get("mean_scores") or {}).items():
            bar = "█" * int(v * 20)
            print(f"    {k:<26} {v:.4f}  {bar}")
        print("────────────────────────────────────────────────────\n")

    def save_session(self, path: str):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.session_summary(),
            "records": self._history,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[TruLens] Session saved → {out}")

    def clear_history(self):
        self._history.clear()

    @property
    def history(self) -> List[dict]:
        return self._history


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _scores_by_category(scores: Dict[str, float]) -> Dict[str, float]:
    """Group feedback scores by category and return mean per category."""
    category_map = {name: cat for name, _, _, cat in _FEEDBACK_REGISTRY}
    cat_vals: Dict[str, List[float]] = {}
    for name, score in scores.items():
        cat = category_map.get(name, "other")
        cat_vals.setdefault(cat, []).append(score)
    return {cat: round(sum(v) / len(v), 4) for cat, v in cat_vals.items()}


def _print_record(record: dict):
    """Pretty-print a single evaluation record."""
    print(f"\n{'='*55}")
    print(f"  Query:      {record['query'][:65]}")
    print(f"  Composite:  {record['composite_score']:.4f}  [{record['grade']}]")
    print(f"  Latency:    {record['latency_ms']:.1f} ms")
    print(f"  Fallback:   {'Yes' if record['had_fallback'] else 'No'}")
    if record.get("error"):
        print(f"  ERROR:      {record['error']}")
    print("  ── Feedback ──────────────────────────────────")
    for name, score in record["feedback_scores"].items():
        bar = "█" * int(score * 20)
        print(f"    {name:<26} {score:.4f}  {bar}")
    print(f"  ── By Category ───────────────────────────────")
    for cat, score in record["by_category"].items():
        print(f"    {cat:<20} {score:.4f}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trulens_eval.py",
        description="TruLens-style pipeline evaluation for Medical GraphRAG",
    )
    sub = p.add_subparsers(dest="cmd")

    # Subcommand: evaluate a single query live
    run_p = sub.add_parser("run", help="Evaluate a single query against the live pipeline")
    run_p.add_argument("--query",          required=True,  help="User query")
    run_p.add_argument("--patient-report", default="",     help="Patient context text")
    run_p.add_argument("--cancer-filter",  default="",     help="Cancer type filter")
    run_p.add_argument("--output",         default="",     help="Save record to JSON file")

    # Subcommand: summarise a saved session file
    summ_p = sub.add_parser("summary", help="Print summary of a saved session file")
    summ_p.add_argument("--session-file", required=True, help="Path to session JSON file")

    # Subcommand: offline demo (no pipeline needed)
    demo_p = sub.add_parser("demo", help="Run offline demo with synthetic trace")

    return p


def _offline_demo():
    """Run the evaluator on a synthetic trace (no pipeline connection needed)."""
    print("\n[TruLens] Running offline demo with synthetic trace...\n")

    trace                    = PipelineTrace("What drugs treat osteosarcoma?")
    trace.patient_report     = "Patient diagnosed with osteosarcoma stage IIB"
    trace.vector_context     = (
        "Osteosarcoma treatment involves neoadjuvant chemotherapy, surgery, and adjuvant "
        "chemotherapy. The four agents include methotrexate with leucovorin rescue, "
        "doxorubicin, cisplatin, and ifosfamide. Patients with metastatic disease may "
        "also receive etoposide. Please consult your oncologist."
    )
    trace.graph_context      = (
        "[Treatment Protocols]\n"
        "  • Osteosarcoma → Cisplatin\n"
        "  • Osteosarcoma → Doxorubicin\n"
        "  • Osteosarcoma → Methotrexate\n"
        "  • Cisplatin → Nephrotoxicity (side effect)\n"
    )
    trace.fused_context      = trace.graph_context + "\n\n" + trace.vector_context
    trace.final_answer       = (
        "Osteosarcoma is typically treated with a combination of chemotherapy drugs "
        "including cisplatin, doxorubicin, methotrexate (with leucovorin rescue), and "
        "ifosfamide. These are used in both neoadjuvant and adjuvant settings. "
        "Please consult your oncologist for personalised treatment advice."
    )
    trace.had_fallback       = False
    trace.extracted_entities = ["Osteosarcoma", "Cisplatin", "Doxorubicin"]
    trace.latency_ms         = 1234.5

    evaluator = TruLensEvaluator(pipeline_available=False)
    record    = evaluator.evaluate_trace(trace)
    _print_record(record)


def main_trulens(args: Optional[argparse.Namespace] = None):
    """Entry point — called by evaluate.py --mode trulens, or directly."""
    if args is None:
        parser = _build_parser()
        args   = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    if not hasattr(args, "cmd") or args.cmd is None or args.cmd == "demo":
        _offline_demo()
        return

    if args.cmd == "run":
        evaluator = TruLensEvaluator(pipeline_available=True)
        record    = evaluator.evaluate(
            query          = args.query,
            patient_report = getattr(args, "patient_report", ""),
            cancer_filter  = getattr(args, "cancer_filter", ""),
        )
        _print_record(record)
        if getattr(args, "output", ""):
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(record, f, indent=2, default=str)
            print(f"\n[TruLens] Record saved → {out}")

    elif args.cmd == "summary":
        path = Path(args.session_file)
        if not path.exists():
            print(f"[ERROR] Session file not found: {path}")
            return
        with open(path) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        print(f"\n── Session: {path.name} ─────────────────────────────")
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"      {sk:<28} {sv}")
            else:
                print(f"  {k:<32} {v}")
        print("─" * 55)


if __name__ == "__main__":
    main_trulens()
