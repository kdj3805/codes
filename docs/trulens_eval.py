"""
trulens_eval.py
─────────────────────────────────────────────────────────────────────────────
RAGAS + TruLens evaluation for the Hybrid Medical RAG pipeline.

Reuses the same pipeline execution path as ragas_eval.py.
Adds TruLens feedback functions on top:
    • groundedness        (answer grounded in context)
    • answer_relevance    (answer relevant to question)
    • context_relevance   (contexts relevant to question)

Usage
-----
    python trulens_eval.py [--dataset PATH] [--limit N]
                           [--cancer-filter TYPE] [--output PATH]
                           [--sample-type TYPE] [--no-ragas] [--no-trulens]

Output
------
    evaluation_results_trulens.json

Comparison table is printed to stdout showing RAGAS vs TruLens scores
side-by-side for easy inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trulens_eval")

# ── RAGAS imports ─────────────────────────────────────────────────────────────
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _RAGAS_AVAILABLE = True
except ImportError as e:
    log.warning("RAGAS not available: %s", e)
    _RAGAS_AVAILABLE = False

# ── TruLens imports ───────────────────────────────────────────────────────────
try:
    from trulens.core import TruSession, Feedback
    from trulens.providers.openai import OpenAI as TruOpenAI
    _TRULENS_AVAILABLE = True
except ImportError as e:
    log.warning("TruLens not available: %s  —  install with: pip install trulens trulens-providers-openai", e)
    _TRULENS_AVAILABLE = False

# ── project pipeline ──────────────────────────────────────────────────────────
from graphrag_integration import generate_answer_graphrag

DEFAULT_DATASET = PROJECT_ROOT / "corrected_golden_dataset.json"
DEFAULT_OUTPUT  = PROJECT_ROOT / "evaluation_results_trulens.json"

RAGAS_LLM_MODEL       = "gpt-4o-mini"
RAGAS_EMBED_MODEL     = "text-embedding-3-small"
TRULENS_MODEL         = "gpt-4o-mini"
INTER_QUERY_DELAY_SEC = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner  (shared with ragas_eval.py logic — no duplication)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(question: str, cancer_filter: str = "") -> tuple[str, list[str]]:
    """Invoke existing GraphRAG pipeline; return (answer, context_strings)."""
    answer, _sources, _ = generate_answer_graphrag(
        query          = question,
        patient_report = "",
        chat_history   = [],
        cancer_filter  = cancer_filter,
        is_analysis    = False,
    )

    from Cancer_retrieval_v2_visual import get_hybrid_mmr_retriever, build_context
    retriever = get_hybrid_mmr_retriever(cancer_filter=cancer_filter)
    retrieved = retriever.invoke(question)
    contexts  = [doc.page_content for doc in retrieved] if retrieved else []

    try:
        from graph_retrieval import retrieve_graph_context
        g = retrieve_graph_context(question, "")
        if g and g.strip():
            contexts.append(g.strip())
    except Exception:
        pass

    return answer, contexts


def _load_dataset(path: Path, limit: Optional[int], sample_type: Optional[str]) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if sample_type:
        data = [s for s in data if s.get("metadata", {}).get("type") == sample_type]
    if limit:
        data = data[:limit]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ragas(ragas_ds: "EvaluationDataset", openai_key: str) -> dict:
    llm   = ChatOpenAI(model=RAGAS_LLM_MODEL, api_key=openai_key)
    embed = OpenAIEmbeddings(model=RAGAS_EMBED_MODEL, api_key=openai_key)
    result = ragas_evaluate(
        dataset          = ragas_ds,
        metrics          = [faithfulness, answer_relevancy, context_precision, context_recall],
        llm              = llm,
        embeddings       = embed,
        raise_exceptions = False,
    )
    out = {}
    for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
        val = result.get(m.name)
        out[m.name] = round(float(val), 4) if val is not None else None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TruLens scoring
# ─────────────────────────────────────────────────────────────────────────────

def _init_trulens_feedbacks(openai_key: str):
    """
    Build TruLens Feedback objects using OpenAI provider.
    Returns (groundedness_fb, answer_relevance_fb, context_relevance_fb).
    """
    provider = TruOpenAI(model_engine=TRULENS_MODEL, api_key=openai_key)

    groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons, name="groundedness")
        .on_input_output()
    )
    answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="answer_relevance")
        .on_input_output()
    )
    context_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons, name="context_relevance")
        .on_input()
        .on(lambda x: x.get("contexts", []))
        .aggregate(lambda scores: sum(scores) / len(scores) if scores else 0.0)
    )
    return groundedness, answer_relevance, context_relevance


def _score_trulens_sample(
    question:         str,
    answer:           str,
    contexts:         list[str],
    groundedness_fb,
    answer_rel_fb,
    ctx_rel_fb,
) -> dict[str, Optional[float]]:
    """
    Run TruLens feedback functions on a single sample without using
    TruLens app instrumentation (keeping it lightweight and compatible
    with the existing pipeline).
    """
    scores: dict[str, Optional[float]] = {
        "groundedness":      None,
        "answer_relevance":  None,
        "context_relevance": None,
    }

    ctx_joined = "\n\n".join(contexts)

    try:
        result = groundedness_fb.imp(answer, ctx_joined)
        # TruLens feedback functions return either a float or (float, reason)
        scores["groundedness"] = round(float(result[0] if isinstance(result, tuple) else result), 4)
    except Exception as exc:
        log.debug("groundedness failed: %s", exc)

    try:
        result = answer_rel_fb.imp(question, answer)
        scores["answer_relevance"] = round(float(result[0] if isinstance(result, tuple) else result), 4)
    except Exception as exc:
        log.debug("answer_relevance failed: %s", exc)

    try:
        per_ctx: list[float] = []
        for ctx in contexts[:5]:           # cap at 5 to control API cost
            r = ctx_rel_fb.imp(question, ctx)
            per_ctx.append(float(r[0] if isinstance(r, tuple) else r))
        if per_ctx:
            scores["context_relevance"] = round(sum(per_ctx) / len(per_ctx), 4)
    except Exception as exc:
        log.debug("context_relevance failed: %s", exc)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_evaluation(
    samples:        list[dict],
    cancer_filter:  str,
    openai_key:     str,
    run_ragas:      bool,
    run_trulens:    bool,
    verbose:        bool,
) -> tuple[list[dict], dict, dict]:
    """
    Iterate over samples, run pipeline, collect per-sample TruLens scores,
    build RAGAS dataset, and return (run_records, ragas_scores, avg_trulens).
    """
    # TruLens feedback functions (initialised once)
    tl_grd = tl_ar = tl_cr = None
    if run_trulens and _TRULENS_AVAILABLE:
        try:
            tl_grd, tl_ar, tl_cr = _init_trulens_feedbacks(openai_key)
            log.info("TruLens feedback functions initialised.")
        except Exception as exc:
            log.warning("TruLens init failed: %s — skipping TruLens scores.", exc)
            run_trulens = False

    run_records: list[dict] = []
    ragas_samples: list["SingleTurnSample"] = []

    total = len(samples)
    for idx, sample in enumerate(samples, 1):
        question     = sample["question"]
        ground_truth = sample["ground_truth"]
        ref_contexts = sample.get("contexts", [])
        meta         = sample.get("metadata", {})

        if verbose:
            print(f"\n[{idx:3d}/{total}] {meta.get('type','?'):8s} | "
                  f"{meta.get('difficulty','?'):6s} | {question[:70]}…")

        t0 = time.monotonic()
        tl_scores: dict[str, Optional[float]] = {
            "groundedness": None, "answer_relevance": None, "context_relevance": None
        }

        try:
            answer, contexts = _run_pipeline(question, cancer_filter)
            elapsed = round(time.monotonic() - t0, 2)

            eval_contexts = contexts if contexts else ref_contexts

            # ── RAGAS sample ──────────────────────────────────────────────────
            if run_ragas and _RAGAS_AVAILABLE:
                ragas_samples.append(SingleTurnSample(
                    user_input         = question,
                    response           = answer,
                    retrieved_contexts = eval_contexts,
                    reference          = ground_truth,
                ))

            # ── TruLens scores ─────────────────────────────────────────────────
            if run_trulens and tl_grd is not None:
                tl_scores = _score_trulens_sample(
                    question, answer, eval_contexts, tl_grd, tl_ar, tl_cr
                )

            record = {
                "index":              idx - 1,
                "question":           question,
                "generated_answer":   answer,
                "ground_truth":       ground_truth,
                "retrieved_contexts": eval_contexts,
                "metadata":           meta,
                "latency_sec":        elapsed,
                "trulens_scores":     tl_scores,
                "error":              None,
            }

            if verbose:
                tl_str = "  ".join(
                    f"{k[:3]}={v:.3f}" if v is not None else f"{k[:3]}=N/A"
                    for k, v in tl_scores.items()
                )
                print(f"          ✓ {elapsed}s | {tl_str}")

        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 2)
            log.error("Pipeline error on sample %d: %s", idx, exc)
            record = {
                "index":              idx - 1,
                "question":           question,
                "generated_answer":   "",
                "ground_truth":       ground_truth,
                "retrieved_contexts": [],
                "metadata":           meta,
                "latency_sec":        elapsed,
                "trulens_scores":     tl_scores,
                "error":              str(exc),
            }
            if verbose:
                print(f"          ✗ ERROR: {exc}")

        run_records.append(record)
        if idx < total:
            time.sleep(INTER_QUERY_DELAY_SEC)

    # ── RAGAS aggregate ────────────────────────────────────────────────────────
    ragas_scores: dict = {}
    if run_ragas and _RAGAS_AVAILABLE and ragas_samples:
        try:
            ragas_scores = _compute_ragas(EvaluationDataset(samples=ragas_samples), openai_key)
        except Exception as exc:
            log.error("RAGAS aggregate failed: %s", exc)

    # ── TruLens aggregate ─────────────────────────────────────────────────────
    avg_tl: dict[str, Optional[float]] = {
        "groundedness": None, "answer_relevance": None, "context_relevance": None
    }
    if run_trulens:
        for key in avg_tl:
            vals = [
                r["trulens_scores"][key]
                for r in run_records
                if r["trulens_scores"].get(key) is not None
            ]
            avg_tl[key] = round(sum(vals) / len(vals), 4) if vals else None

    return run_records, ragas_scores, avg_tl


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(ragas_scores: dict, tl_scores: dict, run_records: list[dict]) -> None:
    errors  = sum(1 for r in run_records if r["error"])
    n       = len(run_records)
    avg_lat = sum(r["latency_sec"] for r in run_records) / max(1, n)

    print("\n" + "═" * 70)
    print("  RAGAS + TRULENS COMPARISON SUMMARY")
    print("═" * 70)
    print(f"  Samples : {n}   Errors : {errors}   Avg latency : {avg_lat:.2f}s")
    print()
    print(f"  {'Metric':<30} {'RAGAS':>10}  {'TruLens':>10}")
    print(f"  {'─'*30} {'─'*10}  {'─'*10}")

    comparison_map = [
        ("Faithfulness",      "faithfulness",      "groundedness"),
        ("Answer Relevance",  "answer_relevancy",  "answer_relevance"),
        ("Context Relevance", "context_precision", "context_relevance"),
        ("Context Recall",    "context_recall",    "—"),
    ]
    for label, ragas_key, tl_key in comparison_map:
        r_val = ragas_scores.get(ragas_key)
        t_val = tl_scores.get(tl_key) if tl_key != "—" else None
        r_str = f"{r_val:.4f}" if r_val is not None else "N/A"
        t_str = f"{t_val:.4f}" if t_val is not None else "N/A"
        print(f"  {label:<30} {r_str:>10}  {t_str:>10}")

    print()
    print("  Note: RAGAS context_precision ≈ TruLens context_relevance (similar but not identical)")
    print("        RAGAS faithfulness ≈ TruLens groundedness (overlap; different LLM judge prompts)")
    print("═" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS + TruLens evaluation for MedChat RAG pipeline"
    )
    parser.add_argument("--dataset",       type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit",         type=int,  default=None)
    parser.add_argument("--cancer-filter", type=str,  default="")
    parser.add_argument("--output",        type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-type",   type=str,  default=None,
                        choices=["vector","graph","hybrid","image","fallback"])
    parser.add_argument("--no-ragas",      action="store_true")
    parser.add_argument("--no-trulens",    action="store_true")
    parser.add_argument("--quiet",         action="store_true")
    args = parser.parse_args(argv)

    run_ragas   = not args.no_ragas
    run_trulens = not args.no_trulens

    # ── OpenAI key check ──────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if (run_ragas or run_trulens) and not openai_key:
        log.warning(
            "OPENAI_API_KEY not set — both RAGAS and TruLens use OpenAI as judge LLM.\n"
            "Set it in .env or export OPENAI_API_KEY=sk-...  Metrics will be skipped."
        )
        run_ragas   = False
        run_trulens = False

    # ── load dataset ──────────────────────────────────────────────────────────
    dataset_path = args.dataset
    if not dataset_path.exists():
        candidate = PROJECT_ROOT / dataset_path.name
        if candidate.exists():
            dataset_path = candidate
        else:
            sys.exit(f"Dataset not found: {dataset_path}")

    samples = _load_dataset(dataset_path, args.limit, args.sample_type)
    if not samples:
        sys.exit("No samples after filtering.")

    log.info("Evaluating %d samples (ragas=%s, trulens=%s)", len(samples), run_ragas, run_trulens)

    print(f"\n{'─'*70}")
    print(f"  Running {len(samples)} samples (RAGAS={run_ragas}, TruLens={run_trulens}) …")
    print(f"{'─'*70}")

    t_start = time.monotonic()
    run_records, ragas_scores, avg_tl = _run_evaluation(
        samples       = samples,
        cancer_filter = args.cancer_filter,
        openai_key    = openai_key,
        run_ragas     = run_ragas,
        run_trulens   = run_trulens,
        verbose       = not args.quiet,
    )
    elapsed = time.monotonic() - t_start

    _print_comparison(ragas_scores, avg_tl, run_records)

    # ── save ──────────────────────────────────────────────────────────────────
    output = {
        "eval_type":      "ragas+trulens",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "dataset":        str(dataset_path),
        "n_samples":      len(run_records),
        "cancer_filter":  args.cancer_filter,
        "sample_type":    args.sample_type,
        "wall_time_sec":  round(elapsed, 2),
        "ragas_metrics":  ragas_scores,
        "trulens_avg":    avg_tl,
        "comparison": {
            "faithfulness_vs_groundedness": {
                "ragas":   ragas_scores.get("faithfulness"),
                "trulens": avg_tl.get("groundedness"),
            },
            "answer_relevancy_vs_answer_relevance": {
                "ragas":   ragas_scores.get("answer_relevancy"),
                "trulens": avg_tl.get("answer_relevance"),
            },
            "context_precision_vs_context_relevance": {
                "ragas":   ragas_scores.get("context_precision"),
                "trulens": avg_tl.get("context_relevance"),
            },
            "context_recall": {
                "ragas":   ragas_scores.get("context_recall"),
                "trulens": None,
            },
        },
        "samples": [
            {
                "index":              r["index"],
                "question":           r["question"],
                "generated_answer":   r["generated_answer"],
                "ground_truth":       r["ground_truth"],
                "retrieved_contexts": r["retrieved_contexts"],
                "metadata":           r["metadata"],
                "latency_sec":        r["latency_sec"],
                "trulens_scores":     r["trulens_scores"],
                "error":              r["error"],
            }
            for r in run_records
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"Results saved → {args.output}\n")


if __name__ == "__main__":
    main()
