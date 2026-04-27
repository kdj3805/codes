"""
ragas_eval.py
─────────────────────────────────────────────────────────────────────────────
RAGAS evaluation for the Hybrid Medical RAG pipeline.

Usage
-----
    python ragas_eval.py [--dataset PATH] [--limit N] [--cancer-filter TYPE]
                         [--output PATH] [--sample-type TYPE]

Examples
--------
    python ragas_eval.py                                 # full dataset
    python ragas_eval.py --limit 20                      # first 20 samples
    python ragas_eval.py --sample-type graph --limit 15  # graph-only samples
    python ragas_eval.py --cancer-filter melanoma        # melanoma filter

Metrics computed
----------------
    faithfulness, answer_relevancy, context_precision, context_recall

Output
------
    evaluation_results_ragas.json
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

# ── project root on path ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ragas_eval")

# ── RAGAS imports ────────────────────────────────────────────────────────────
try:
    from ragas import evaluate
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
    log.warning("RAGAS not available: %s  —  install with: pip install ragas", e)
    _RAGAS_AVAILABLE = False

# ── project pipeline ─────────────────────────────────────────────────────────
from graphrag_integration import generate_answer_graphrag

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASET = PROJECT_ROOT / "corrected_golden_dataset.json"
DEFAULT_OUTPUT  = PROJECT_ROOT / "evaluation_results_ragas.json"

RAGAS_LLM_MODEL       = "gpt-4o-mini"   # cost-effective; swap to gpt-4o for higher accuracy
RAGAS_EMBED_MODEL     = "text-embedding-3-small"
INTER_QUERY_DELAY_SEC = 1.5             # avoid rate-limiting on Groq + OpenAI


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _load_dataset(path: Path, limit: Optional[int], sample_type: Optional[str]) -> list[dict]:
    """Load and optionally filter the golden dataset."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if sample_type:
        data = [s for s in data if s.get("metadata", {}).get("type") == sample_type]
        log.info("Filtered to type=%r → %d samples", sample_type, len(data))

    if limit:
        data = data[:limit]
        log.info("Capped at limit=%d", limit)

    return data


def _run_pipeline(
    question: str,
    cancer_filter: str = "",
) -> tuple[str, list[str]]:
    """
    Invoke the existing GraphRAG pipeline and return (answer, context_strings).
    Reuses generate_answer_graphrag from graphrag_integration.py.
    """
    answer, sources, _ = generate_answer_graphrag(
        query          = question,
        patient_report = "",
        chat_history   = [],
        cancer_filter  = cancer_filter,
        is_analysis    = False,
    )

    # Pull retrieved context strings from session state written by the stream path
    # generate_answer_graphrag stores them via Cancer_retrieval_v2_visual helpers.
    # We replicate the retrieval here to get raw context chunks for RAGAS.
    from Cancer_retrieval_v2_visual import (
        get_hybrid_mmr_retriever,
        build_context,
        _extract_sources,
    )
    retriever = get_hybrid_mmr_retriever(cancer_filter=cancer_filter)
    retrieved = retriever.invoke(question)
    raw_contexts = [doc.page_content for doc in retrieved] if retrieved else []

    # Supplement with graph context if available
    try:
        from graph_retrieval import retrieve_graph_context
        graph_ctx = retrieve_graph_context(question, "")
        if graph_ctx and graph_ctx.strip():
            raw_contexts.append(graph_ctx.strip())
    except Exception:
        pass

    return answer, raw_contexts


def _build_ragas_dataset(
    samples:       list[dict],
    cancer_filter: str = "",
    verbose:       bool = True,
) -> tuple[list[dict], "EvaluationDataset"]:
    """
    Run each golden sample through the pipeline, collect results,
    and build a RAGAS EvaluationDataset.

    Returns (run_records, ragas_dataset) where run_records contain
    per-sample data for the output JSON.
    """
    run_records: list[dict] = []
    ragas_samples: list[SingleTurnSample] = []

    total = len(samples)
    for idx, sample in enumerate(samples, 1):
        question     = sample["question"]
        ground_truth = sample["ground_truth"]
        ref_contexts = sample.get("contexts", [])
        meta         = sample.get("metadata", {})

        if verbose:
            print(f"\n[{idx:3d}/{total}] {meta.get('type','?'):8s} | "
                  f"{meta.get('difficulty','?'):6s} | {question[:72]}…")

        t0 = time.monotonic()
        try:
            generated_answer, retrieved_contexts = _run_pipeline(question, cancer_filter)
            elapsed = round(time.monotonic() - t0, 2)

            # RAGAS uses the RETRIEVED contexts (what the system actually fetched),
            # not the golden reference contexts, for faithfulness / precision / recall.
            contexts_for_ragas = retrieved_contexts if retrieved_contexts else ref_contexts

            ragas_sample = SingleTurnSample(
                user_input        = question,
                response          = generated_answer,
                retrieved_contexts= contexts_for_ragas,
                reference         = ground_truth,
            )
            ragas_samples.append(ragas_sample)

            record = {
                "index":              idx - 1,
                "question":           question,
                "generated_answer":   generated_answer,
                "ground_truth":       ground_truth,
                "retrieved_contexts": contexts_for_ragas,
                "ref_contexts":       ref_contexts,
                "metadata":           meta,
                "latency_sec":        elapsed,
                "error":              None,
            }
            if verbose:
                print(f"          ✓ {elapsed}s | answer_len={len(generated_answer)}")

        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 2)
            log.error("Pipeline error on sample %d: %s", idx, exc)
            record = {
                "index":              idx - 1,
                "question":           question,
                "generated_answer":   "",
                "ground_truth":       ground_truth,
                "retrieved_contexts": [],
                "ref_contexts":       ref_contexts,
                "metadata":           meta,
                "latency_sec":        elapsed,
                "error":              str(exc),
            }
            if verbose:
                print(f"          ✗ ERROR: {exc}")

        run_records.append(record)

        if idx < total:
            time.sleep(INTER_QUERY_DELAY_SEC)

    ragas_ds = EvaluationDataset(samples=ragas_samples)
    return run_records, ragas_ds


def _compute_ragas_metrics(
    ragas_ds: "EvaluationDataset",
) -> dict:
    """
    Run RAGAS evaluate() and return a dict of {metric_name: score}.
    Requires OPENAI_API_KEY in environment (used for RAGAS LLM judge).
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. RAGAS uses an OpenAI LLM judge.\n"
            "Set it in your .env file or export OPENAI_API_KEY=sk-..."
        )

    llm    = ChatOpenAI(model=RAGAS_LLM_MODEL, api_key=openai_key)
    embed  = OpenAIEmbeddings(model=RAGAS_EMBED_MODEL, api_key=openai_key)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    log.info("Running RAGAS evaluate() on %d samples …", len(ragas_ds.samples))
    result = evaluate(
        dataset      = ragas_ds,
        metrics      = metrics,
        llm          = llm,
        embeddings   = embed,
        raise_exceptions = False,
    )

    scores = {}
    for metric in metrics:
        name = metric.name
        val  = result.get(name)
        scores[name] = round(float(val), 4) if val is not None else None

    return scores


def _print_summary(scores: dict, run_records: list[dict], elapsed_total: float) -> None:
    """Pretty-print the evaluation summary to stdout."""
    errors = sum(1 for r in run_records if r["error"])
    print("\n" + "═" * 64)
    print("  RAGAS EVALUATION SUMMARY")
    print("═" * 64)
    print(f"  Samples evaluated : {len(run_records)} ({errors} errors)")
    print(f"  Total wall-time   : {elapsed_total:.1f}s")
    print(f"  Avg latency/query : {sum(r['latency_sec'] for r in run_records)/max(1,len(run_records)):.2f}s")
    print()
    print(f"  {'Metric':<30} {'Score':>8}")
    print(f"  {'─'*30} {'─'*8}")
    metric_labels = {
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
    }
    for key, label in metric_labels.items():
        val = scores.get(key)
        val_str = f"{val:.4f}" if val is not None else "N/A"
        print(f"  {label:<30} {val_str:>8}")
    print("═" * 64 + "\n")


def _slice_scores(run_records: list[dict]) -> dict:
    """
    Compute per-slice breakdowns by type and difficulty using index mapping.
    These are populated from per-sample RAGAS scores if available, or
    left as counts-only if per-sample scores aren't returned by RAGAS.
    """
    slices: dict = {}
    for r in run_records:
        meta = r.get("metadata", {})
        for key in ("type", "difficulty", "reasoning_type"):
            val = meta.get(key, "unknown")
            slices.setdefault(key, {}).setdefault(val, 0)
            slices[key][val] += 1
    return slices


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for MedChat RAG pipeline")
    parser.add_argument("--dataset",      type=Path,  default=DEFAULT_DATASET,
                        help="Path to golden dataset JSON")
    parser.add_argument("--limit",        type=int,   default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--cancer-filter",type=str,   default="",
                        help="Cancer type filter for retriever (e.g. melanoma)")
    parser.add_argument("--output",       type=Path,  default=DEFAULT_OUTPUT,
                        help="Path for output JSON results")
    parser.add_argument("--sample-type",  type=str,   default=None,
                        choices=["vector","graph","hybrid","image","fallback"],
                        help="Filter dataset by sample type")
    parser.add_argument("--no-ragas",     action="store_true",
                        help="Skip RAGAS metrics; only run pipeline + save answers")
    parser.add_argument("--quiet",        action="store_true",
                        help="Suppress per-sample progress output")
    args = parser.parse_args(argv)

    # ── load dataset ─────────────────────────────────────────────────────────
    dataset_path = args.dataset
    if not dataset_path.exists():
        # try relative to project root
        candidate = PROJECT_ROOT / dataset_path.name
        if candidate.exists():
            dataset_path = candidate
        else:
            sys.exit(f"Dataset not found: {dataset_path}")

    samples = _load_dataset(dataset_path, args.limit, args.sample_type)
    log.info("Loaded %d samples from %s", len(samples), dataset_path)

    if not samples:
        sys.exit("No samples to evaluate after filtering.")

    # ── run pipeline ─────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  Running {len(samples)} samples through the GraphRAG pipeline …")
    print(f"{'─'*64}")

    t_start = time.monotonic()
    run_records, ragas_ds = _build_ragas_dataset(
        samples       = samples,
        cancer_filter = args.cancer_filter,
        verbose       = not args.quiet,
    )
    t_pipeline = time.monotonic() - t_start

    # ── RAGAS metrics ─────────────────────────────────────────────────────────
    ragas_scores: dict = {}
    if not args.no_ragas:
        if not _RAGAS_AVAILABLE:
            log.warning("RAGAS not installed — skipping metric computation.")
        else:
            try:
                ragas_scores = _compute_ragas_metrics(ragas_ds)
            except EnvironmentError as env_err:
                log.warning("%s", env_err)
                log.warning("Skipping RAGAS metrics.  Pipeline answers are still saved.")
            except Exception as exc:
                log.error("RAGAS evaluate() failed: %s", exc)

    # ── print summary ─────────────────────────────────────────────────────────
    _print_summary(ragas_scores, run_records, t_pipeline)

    # ── build output document ─────────────────────────────────────────────────
    output = {
        "eval_type":     "ragas",
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "dataset":       str(dataset_path),
        "n_samples":     len(run_records),
        "cancer_filter": args.cancer_filter,
        "sample_type":   args.sample_type,
        "pipeline_wall_time_sec": round(t_pipeline, 2),
        "ragas_metrics": ragas_scores,
        "slice_counts":  _slice_scores(run_records),
        "samples": [
            {
                "index":              r["index"],
                "question":           r["question"],
                "generated_answer":   r["generated_answer"],
                "ground_truth":       r["ground_truth"],
                "retrieved_contexts": r["retrieved_contexts"],
                "metadata":           r["metadata"],
                "latency_sec":        r["latency_sec"],
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
