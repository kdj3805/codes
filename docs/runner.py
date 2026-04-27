"""
evaluation/runner.py
--------------------
Main evaluation runner for the Medical GraphRAG pipeline.

Orchestrates all evaluators into a single call and produces a
structured JSON-serialisable evaluation report.

Usage (minimal):
    from evaluation.runner import EvaluationRunner

    runner = EvaluationRunner()
    report = runner.run(
        query         = "What chemotherapy is used for osteosarcoma?",
        retrieved_docs= ["Osteosarcoma is treated with ..."],
        graph_context = "CANCER –[HAS_DRUG]→ Cisplatin",
        vector_context= "Osteosarcoma treatment involves ...",
        fused_context = "<fused>",
        final_answer  = "Osteosarcoma is typically treated with ...",
    )
    print(report.to_json(indent=2))

Usage (full):
    runner = EvaluationRunner(use_llm=True, k=5)
    report = runner.run(
        query                 = "...",
        retrieved_docs        = [...],
        graph_context         = "...",
        vector_context        = "...",
        fused_context         = "...",
        final_answer          = "...",
        # Optional parameters:
        expected_answer       = "...",        # ground truth
        extracted_entities    = ["Osteosarcoma", "Cisplatin"],
        graph_results         = [...],        # raw GraphResult dicts
        multihop_paths        = [["Cancer", "Drug", "Side_Effect"]],
        selected_image_captions = ["Figure 1. ..."],
        all_image_captions    = [...],
        rrf_scores            = [0.9, 0.7, ...],
        relevant_doc_ids      = ["doc1", "doc2"],
        retrieved_doc_ids     = ["doc1", "doc3", "doc4"],
        is_web_fallback       = False,
    )

Saving reports:
    report.save("eval_reports/run_001.json")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.evaluators import (
    EvalResult,
    FallbackEvaluator,
    FusionEvaluator,
    GenerationEvaluator,
    GraphEvaluator,
    MultimodalEvaluator,
    RetrievalEvaluator,
)
from evaluation.ragas_eval import RagasEvaluator
from evaluation.trulens_integration import (
    FAST_FEEDBACK_SET,
    STANDARD_FEEDBACK_SET,
    TruLensRecorder,
)
from evaluation.metrics import letter_grade, weighted_average

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Report container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    """
    Full structured evaluation report for one pipeline call.

    All numeric scores are in [0, 1].
    """
    query:             str
    final_answer:      str

    retrieval:         Optional[dict] = None
    generation:        Optional[dict] = None
    graph:             Optional[dict] = None
    fusion:            Optional[dict] = None
    fallback:          Optional[dict] = None
    multimodal:        Optional[dict] = None
    ragas:             Optional[dict] = None
    trulens:           Optional[dict] = None

    overall_score:     float = 0.0
    overall_grade:     str   = ""
    evaluation_time_ms: float = 0.0
    timestamp:         float = field(default_factory=time.time)
    warnings:          List[str] = field(default_factory=list)

    # Weights for computing overall score
    COMPONENT_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "retrieval":   1.5,
        "generation":  2.5,
        "graph":       1.5,
        "fusion":      1.0,
        "fallback":    0.5,
        "multimodal":  0.5,
        "ragas":       2.0,
        "trulens":     1.5,
    })

    def _compute_overall(self) -> float:
        component_scores = {}
        for name, data in [
            ("retrieval",  self.retrieval),
            ("generation", self.generation),
            ("graph",      self.graph),
            ("fusion",     self.fusion),
            ("fallback",   self.fallback),
            ("multimodal", self.multimodal),
        ]:
            if data and "overall" in data:
                component_scores[name] = data["overall"]

        if self.ragas and "ragas_score" in self.ragas:
            component_scores["ragas"] = self.ragas["ragas_score"]

        if self.trulens and "composite_score" in self.trulens:
            component_scores["trulens"] = self.trulens["composite_score"]

        if not component_scores:
            return 0.0
        return weighted_average(component_scores, self.COMPONENT_WEIGHTS)

    def __post_init__(self):
        self.overall_score = self._compute_overall()
        self.overall_grade = letter_grade(self.overall_score)

    def to_dict(self) -> dict:
        return {
            "query":               self.query,
            "final_answer_preview": self.final_answer[:200] + "..." if len(self.final_answer) > 200
                                    else self.final_answer,
            "overall_score":       self.overall_score,
            "overall_grade":       self.overall_grade,
            "evaluation_time_ms":  round(self.evaluation_time_ms, 2),
            "timestamp":           self.timestamp,
            "warnings":            self.warnings,
            "components": {
                "retrieval":   self.retrieval,
                "generation":  self.generation,
                "graph":       self.graph,
                "fusion":      self.fusion,
                "fallback":    self.fallback,
                "multimodal":  self.multimodal,
                "ragas":       self.ragas,
                "trulens":     self.trulens,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path):
        """Save report to a JSON file. Creates parent directories if needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        log.info("[EvaluationReport] Saved to %s", p)

    def print_summary(self):
        """Print a compact human-readable summary to stdout."""
        print("=" * 60)
        print(f"  EVALUATION SUMMARY")
        print(f"  Query: {self.query[:70]}...")
        print(f"  Overall Score: {self.overall_score:.3f}  Grade: {self.overall_grade}")
        print("-" * 60)
        components = {
            "Retrieval":   self.retrieval,
            "Generation":  self.generation,
            "Graph":       self.graph,
            "Fusion":      self.fusion,
            "Fallback":    self.fallback,
            "Multimodal":  self.multimodal,
            "RAGAS":       self.ragas,
            "TruLens":     self.trulens,
        }
        for name, data in components.items():
            if data is None:
                continue
            score = data.get("overall") or data.get("ragas_score") or data.get("composite_score", 0)
            grade = data.get("grade", letter_grade(score))
            print(f"  {name:<12} {score:.3f}  [{grade}]")
        if self.warnings:
            print("-" * 60)
            print("  WARNINGS:")
            for w in self.warnings:
                print(f"    ⚠  {w}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationRunner
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationRunner:
    """
    Orchestrates all evaluators for the Medical GraphRAG pipeline.

    Parameters
    ----------
    use_llm : bool
        Enable LLM-based evaluation (faithfulness judge via Groq).
        Requires GROQ_API_KEY environment variable. Default False.
    k : int
        Rank cutoff for retrieval metrics P@k, R@k, nDCG@k. Default 5.
    enable_ragas : bool
        Run RAGAS-style evaluation. Default True.
    enable_trulens : bool
        Run TruLens-style feedback functions. Default True.
    fast_mode : bool
        Use only a subset of feedback functions for speed. Default False.
    log_level : str
        Logging level. Default 'WARNING'.
    """

    def __init__(
        self,
        use_llm:        bool = False,
        k:              int  = 5,
        enable_ragas:   bool = True,
        enable_trulens: bool = True,
        fast_mode:      bool = False,
        log_level:      str  = "WARNING",
    ):
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.WARNING))

        self.use_llm        = use_llm
        self.enable_ragas   = enable_ragas
        self.enable_trulens = enable_trulens

        # Instantiate evaluators once (reuse across run() calls)
        self._retrieval  = RetrievalEvaluator(k=k)
        self._generation = GenerationEvaluator(use_llm=use_llm)
        self._graph      = GraphEvaluator()
        self._fusion     = FusionEvaluator()
        self._fallback   = FallbackEvaluator()
        self._multimodal = MultimodalEvaluator()
        self._ragas      = RagasEvaluator(use_llm=use_llm) if enable_ragas else None

        if enable_trulens:
            fns = FAST_FEEDBACK_SET if fast_mode else STANDARD_FEEDBACK_SET
            self._trulens = TruLensRecorder(feedback_functions=fns, store_history=True)
        else:
            self._trulens = None

        log.info(
            "[EvaluationRunner] Initialised — use_llm=%s, ragas=%s, trulens=%s, fast=%s",
            use_llm, enable_ragas, enable_trulens, fast_mode,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        # Required
        query:                  str,
        final_answer:           str,
        retrieved_docs:         List[str],
        vector_context:         str,
        graph_context:          str,
        fused_context:          str,
        # Highly recommended
        expected_answer:        str             = "",
        extracted_entities:     List[str]       = None,
        graph_results:          List[dict]      = None,
        # Optional – retrieval ranking
        rrf_scores:             List[float]     = None,
        relevant_doc_ids:       List[str]       = None,
        retrieved_doc_ids:      List[str]       = None,
        # Optional – graph
        multihop_paths:         List[List[str]] = None,
        # Optional – multimodal
        selected_image_captions: List[str]      = None,
        all_image_captions:     List[str]       = None,
        ground_truth_captions:  List[str]       = None,
        # Optional – flags
        is_web_fallback:        bool            = False,
    ) -> EvaluationReport:
        """
        Run the full evaluation suite and return an EvaluationReport.

        Parameters
        ----------
        query                   : Original user query
        final_answer            : LLM-generated answer (may include WEB_FALLBACK_MARKER)
        retrieved_docs          : page_content strings from Qdrant retrieval
        vector_context          : formatted vector context string
        graph_context           : formatted graph context string
        fused_context           : fused context string (output of fuse_contexts())
        expected_answer         : ground-truth / reference answer (optional)
        extracted_entities      : canonical entity names from entity_extractor.py
        graph_results           : raw GraphResult dicts from graph_retrieval.py
        rrf_scores              : RRF fusion scores for each retrieved doc
        relevant_doc_ids        : IDs of ground-truth relevant docs
        retrieved_doc_ids       : IDs of actually retrieved docs (same order)
        multihop_paths          : list of [node1, node2, ...] paths from Neo4j
        selected_image_captions : captions of images selected for the answer
        all_image_captions      : captions of ALL candidate images
        ground_truth_captions   : ideal image captions (for multimodal recall)
        is_web_fallback         : whether the answer came from web search fallback
        """
        t_start   = time.perf_counter()
        all_warns = []

        extracted_entities   = extracted_entities   or []
        graph_results        = graph_results        or []
        selected_image_captions = selected_image_captions or []
        all_image_captions   = all_image_captions   or []

        # ── 1. Retrieval ─────────────────────────────────────────────────────
        ret_result  = self._retrieval.evaluate(
            query             = query,
            retrieved_docs    = retrieved_docs,
            expected_answer   = expected_answer,
            rrf_scores        = rrf_scores,
            relevant_doc_ids  = relevant_doc_ids,
            retrieved_doc_ids = retrieved_doc_ids,
        )
        all_warns.extend(ret_result.warnings)

        # ── 2. Generation ────────────────────────────────────────────────────
        gen_result = self._generation.evaluate(
            query           = query,
            answer          = final_answer,
            context         = fused_context or vector_context,
            reference       = expected_answer,
            is_web_fallback = is_web_fallback,
        )
        all_warns.extend(gen_result.warnings)

        # ── 3. Graph ─────────────────────────────────────────────────────────
        grp_result = self._graph.evaluate(
            query               = query,
            graph_context       = graph_context,
            graph_results       = graph_results,
            extracted_entities  = extracted_entities,
            multihop_paths      = multihop_paths,
        )
        all_warns.extend(grp_result.warnings)

        # ── 4. Fusion ────────────────────────────────────────────────────────
        fus_result = self._fusion.evaluate(
            query          = query,
            answer         = final_answer,
            vector_context = vector_context,
            graph_context  = graph_context,
            fused_context  = fused_context,
        )
        all_warns.extend(fus_result.warnings)

        # ── 5. Fallback ──────────────────────────────────────────────────────
        fb_result = self._fallback.evaluate(
            query            = query,
            final_answer     = final_answer,
            vector_ctx_empty = not bool(vector_context and vector_context.strip()),
            graph_ctx_empty  = not bool(graph_context  and graph_context.strip()),
        )
        all_warns.extend(fb_result.warnings)

        # ── 6. Multimodal ────────────────────────────────────────────────────
        mm_result = self._multimodal.evaluate(
            query                   = query,
            selected_captions       = selected_image_captions,
            all_candidate_captions  = all_image_captions,
            ground_truth_captions   = ground_truth_captions,
        )
        all_warns.extend(mm_result.warnings)

        # ── 7. RAGAS ─────────────────────────────────────────────────────────
        ragas_dict = None
        if self._ragas is not None:
            try:
                ragas_res  = self._ragas.run_all(
                    query        = query,
                    answer       = final_answer,
                    contexts     = retrieved_docs or [vector_context],
                    ground_truth = expected_answer,
                )
                ragas_dict = ragas_res.to_dict()
            except Exception as e:
                log.warning("[EvaluationRunner] RAGAS evaluation failed: %s", e)
                all_warns.append(f"RAGAS evaluation failed: {e}")

        # ── 8. TruLens ───────────────────────────────────────────────────────
        trulens_dict = None
        if self._trulens is not None:
            try:
                tl_record    = self._trulens.record(
                    query               = query,
                    final_answer        = final_answer,
                    retrieved_docs      = retrieved_docs,
                    vector_context      = vector_context,
                    graph_context       = graph_context,
                    fused_context       = fused_context,
                    graph_results       = graph_results,
                    extracted_entities  = extracted_entities,
                    selected_captions   = selected_image_captions,
                    multihop_paths      = multihop_paths,
                )
                trulens_dict = tl_record.to_dict()
            except Exception as e:
                log.warning("[EvaluationRunner] TruLens evaluation failed: %s", e)
                all_warns.append(f"TruLens evaluation failed: {e}")

        # ── Assemble report ──────────────────────────────────────────────────
        eval_ms = (time.perf_counter() - t_start) * 1000

        report = EvaluationReport(
            query             = query,
            final_answer      = final_answer,
            retrieval         = ret_result.to_dict(),
            generation        = gen_result.to_dict(),
            graph             = grp_result.to_dict(),
            fusion            = fus_result.to_dict(),
            fallback          = fb_result.to_dict(),
            multimodal        = mm_result.to_dict(),
            ragas             = ragas_dict,
            trulens           = trulens_dict,
            evaluation_time_ms= eval_ms,
            warnings          = list(set(all_warns)),
        )

        return report

    # ──────────────────────────────────────────────────────────────────────
    # Batch evaluation
    # ──────────────────────────────────────────────────────────────────────

    def run_batch(
        self,
        samples: List[dict],
        save_dir: Optional[str] = None,
    ) -> List[EvaluationReport]:
        """
        Evaluate a list of samples.

        Each sample dict must contain the same keys as run() parameters.
        Optionally saves individual JSON reports to `save_dir`.

        Returns list of EvaluationReport objects.
        """
        reports = []
        for i, sample in enumerate(samples):
            log.info("[EvaluationRunner] Batch sample %d/%d", i + 1, len(samples))
            try:
                report = self.run(**sample)
                reports.append(report)
                if save_dir:
                    fname = sample.get("id", f"sample_{i:04d}")
                    report.save(Path(save_dir) / f"{fname}.json")
            except Exception as e:
                log.error("[EvaluationRunner] Sample %d failed: %s", i, e)
                # Append a minimal error report
                reports.append(EvaluationReport(
                    query=sample.get("query", "UNKNOWN"),
                    final_answer="",
                    warnings=[f"Evaluation failed: {e}"],
                ))
        return reports

    def aggregate_batch(self, reports: List[EvaluationReport]) -> dict:
        """
        Compute aggregate statistics over a list of reports.
        Returns mean scores per component and overall.
        """
        def _mean(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        agg: Dict[str, Any] = {
            "num_samples":    len(reports),
            "overall":        _mean([r.overall_score for r in reports]),
            "retrieval":      {},
            "generation":     {},
            "graph":          {},
            "fusion":         {},
            "fallback":       {},
            "multimodal":     {},
            "ragas":          {},
            "trulens":        {},
        }

        for comp_name in ["retrieval", "generation", "graph", "fusion", "fallback", "multimodal"]:
            all_scores: Dict[str, List[float]] = {}
            for r in reports:
                comp_data = getattr(r, comp_name)
                if comp_data and "scores" in comp_data:
                    for k, v in comp_data["scores"].items():
                        all_scores.setdefault(k, []).append(v)
            agg[comp_name] = {k: _mean(v) for k, v in all_scores.items()}

        # RAGAS aggregation
        ragas_keys = [
            "faithfulness", "answer_relevance", "context_precision",
            "context_recall", "ragas_score",
        ]
        ragas_agg: Dict[str, List[float]] = {k: [] for k in ragas_keys}
        for r in reports:
            if r.ragas:
                for k in ragas_keys:
                    if k in r.ragas:
                        ragas_agg[k].append(r.ragas[k])
        agg["ragas"] = {k: _mean(v) for k, v in ragas_agg.items()}

        # TruLens aggregation
        if self._trulens:
            agg["trulens"] = self._trulens.aggregate_history()

        return agg

    # ──────────────────────────────────────────────────────────────────────
    # History access
    # ──────────────────────────────────────────────────────────────────────

    def get_trulens_history(self) -> List[dict]:
        """Return all TruLens records from this session."""
        if self._trulens:
            return self._trulens.get_history()
        return []

    def clear_history(self):
        """Clear TruLens session history."""
        if self._trulens:
            self._trulens.clear_history()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader: build a sample from live pipeline outputs
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_sample(
    query:               str,
    pipeline_output:     tuple,           # (answer, sources, followups) from generate_answer_graphrag
    retrieved_docs=None,                  # list of LangChain Document objects
    vector_context:      str = "",
    graph_context:       str = "",
    fused_context:       str = "",
    extracted_entities:  List[str] = None,
    graph_results:       List[dict] = None,
    expected_answer:     str = "",
    selected_image_captions: List[str] = None,
    sample_id:           str = "",
) -> dict:
    """
    Build a standardised evaluation sample dict from live pipeline outputs.

    Designed to wrap generate_answer_graphrag() output directly:

        answer, sources, followups = generate_answer_graphrag(query, ...)
        sample = build_eval_sample(
            query            = query,
            pipeline_output  = (answer, sources, followups),
            retrieved_docs   = retrieved,
            vector_context   = vector_ctx,
            graph_context    = graph_ctx,
            fused_context    = fused_ctx,
        )
        report = runner.run(**sample)
    """
    answer = pipeline_output[0] if pipeline_output else ""

    # Determine if this was a web fallback
    is_fallback = "<!-- WEB_FALLBACK_EMPTY_CONTEXT -->" in answer

    # Extract plain text from LangChain Documents if needed
    doc_texts = []
    if retrieved_docs:
        for doc in retrieved_docs:
            if hasattr(doc, "page_content"):
                doc_texts.append(doc.page_content)
            elif isinstance(doc, str):
                doc_texts.append(doc)

    return {
        "query":                  query,
        "final_answer":           answer,
        "retrieved_docs":         doc_texts,
        "vector_context":         vector_context,
        "graph_context":          graph_context,
        "fused_context":          fused_context,
        "expected_answer":        expected_answer,
        "extracted_entities":     extracted_entities or [],
        "graph_results":          graph_results or [],
        "is_web_fallback":        is_fallback,
        "selected_image_captions": selected_image_captions or [],
        "id":                     sample_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke-test / demo run.

    Usage:
        python -m evaluation.runner
    """
    import sys

    logging.basicConfig(level=logging.INFO)

    # ── Sample data (matches real pipeline output shapes) ──────────────────
    SAMPLE_QUERY = "What chemotherapy drugs are used to treat osteosarcoma?"

    SAMPLE_DOCS = [
        "Osteosarcoma is treated with neoadjuvant chemotherapy, surgery, and adjuvant chemotherapy. "
        "The four chemotherapy agents in nearly all treatment regimens include methotrexate with "
        "leucovorin rescue, doxorubicin, cisplatin, and ifosfamide.",
        "Prior to the 1970s, chemotherapy was not used for osteosarcoma and survival rates were "
        "dismal. The introduction of adjuvant chemotherapy increased survival rates to 50%.",
        "Patients who had greater than 90% tumor necrosis had an 82% five-year survival rate, "
        "while those who had less than 90% tumor necrosis had a 68% five-year survival rate.",
    ]

    SAMPLE_VECTOR_CTX = "\n\n".join(SAMPLE_DOCS)

    SAMPLE_GRAPH_CTX = (
        "GRAPH KNOWLEDGE BASE:\n"
        "Osteosarcoma –[HAS_TREATMENT]→ Chemotherapy\n"
        "Chemotherapy –[INCLUDES_DRUG]→ Cisplatin\n"
        "Chemotherapy –[INCLUDES_DRUG]→ Doxorubicin\n"
        "Cisplatin –[HAS_SIDE_EFFECT]→ Nephrotoxicity\n"
        "Doxorubicin –[HAS_SIDE_EFFECT]→ Cardiotoxicity\n"
    )

    SAMPLE_FUSED_CTX = SAMPLE_GRAPH_CTX + "\n\n" + SAMPLE_VECTOR_CTX

    SAMPLE_ANSWER = (
        "Osteosarcoma is treated with a combination of neoadjuvant chemotherapy, surgery, "
        "and adjuvant chemotherapy. The main drugs used are methotrexate with leucovorin rescue, "
        "doxorubicin, cisplatin, and ifosfamide. Patients with metastatic disease may also "
        "receive etoposide. Response to neoadjuvant chemotherapy guides overall treatment."
    )

    SAMPLE_EXPECTED = (
        "The standard chemotherapy for osteosarcoma includes methotrexate, doxorubicin, "
        "cisplatin, and ifosfamide, used in neoadjuvant and adjuvant settings."
    )

    SAMPLE_ENTITIES   = ["Osteosarcoma", "Cisplatin", "Doxorubicin", "Methotrexate", "Ifosfamide"]
    SAMPLE_GRAPH_RES  = [
        {"query_type": "drugs_for_cancer", "records": [
            {"drug": "Cisplatin"},
            {"drug": "Doxorubicin"},
        ]},
    ]
    SAMPLE_MULTIHOP   = [
        ["Osteosarcoma", "Chemotherapy", "Cisplatin"],
        ["Cisplatin",    "Side_Effect",  "Nephrotoxicity"],
    ]
    SAMPLE_IMG_CAPS   = ["Figure 1. Osteosarcoma X-ray showing metaphyseal lesion."]

    # ── Run evaluation ─────────────────────────────────────────────────────
    print("\n🔬 Running GraphRAG Evaluation Framework demo...\n")

    runner = EvaluationRunner(
        use_llm        = False,   # set True if GROQ_API_KEY is set
        k              = 3,
        enable_ragas   = True,
        enable_trulens = True,
        fast_mode      = False,
    )

    report = runner.run(
        query                   = SAMPLE_QUERY,
        final_answer            = SAMPLE_ANSWER,
        retrieved_docs          = SAMPLE_DOCS,
        vector_context          = SAMPLE_VECTOR_CTX,
        graph_context           = SAMPLE_GRAPH_CTX,
        fused_context           = SAMPLE_FUSED_CTX,
        expected_answer         = SAMPLE_EXPECTED,
        extracted_entities      = SAMPLE_ENTITIES,
        graph_results           = SAMPLE_GRAPH_RES,
        multihop_paths          = SAMPLE_MULTIHOP,
        selected_image_captions = SAMPLE_IMG_CAPS,
        all_image_captions      = SAMPLE_IMG_CAPS,
        rrf_scores              = [0.95, 0.80, 0.65],
        is_web_fallback         = False,
    )

    report.print_summary()

    # Save to file
    out_path = Path("eval_reports") / "demo_report.json"
    report.save(out_path)
    print(f"\n📄 Full report saved to: {out_path}\n")

    # Show JSON preview
    data = report.to_dict()
    print("JSON Preview (scores only):")
    for comp, val in data["components"].items():
        if val:
            score = val.get("overall") or val.get("ragas_score") or val.get("composite_score", "—")
            print(f"  {comp:<14} {score}")

    sys.exit(0)
