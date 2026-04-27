"""
evaluation/evaluate.py
----------------------
CLI entry point for the Medical GraphRAG Evaluation Framework.

Supports two evaluation modes:
  --mode ragas      Run RAGAS-style metrics on a JSON dataset
  --mode trulens    Run TruLens-style feedback on a live query or demo

Usage:
    # RAGAS — evaluate a dataset
    python evaluate.py --mode ragas --dataset data/eval_dataset.json

    # RAGAS — with LLM judge and custom output path
    python evaluate.py --mode ragas --dataset data/eval_dataset.json \\
                       --output results/ragas.json --use-llm

    # TruLens — offline demo (no pipeline connection needed)
    python evaluate.py --mode trulens

    # TruLens — live query against the real pipeline
    python evaluate.py --mode trulens \\
                       --query "What drugs treat osteosarcoma?" \\
                       --patient-report "Patient has osteosarcoma stage IIB"

    # TruLens — summarise a saved session
    python evaluate.py --mode trulens --summary results/session.json

    # Run both modes sequentially
    python evaluate.py --mode both --dataset data/eval_dataset.json \\
                       --query "What is osteosarcoma?"

Environment variables:
    GROQ_API_KEY      Required for --use-llm and live TruLens pipeline calls
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Medical GraphRAG Evaluation Framework\n"
            "Supports RAGAS-style dataset evaluation and TruLens-style pipeline feedback."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Core mode ──────────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=["ragas", "trulens", "both"],
        default="ragas",
        help="Evaluation mode  (default: ragas)",
    )

    # ── RAGAS options ───────────────────────────────────────────────────────
    ragas_g = p.add_argument_group("RAGAS options")
    ragas_g.add_argument(
        "--dataset",
        default="",
        metavar="PATH",
        help="Path to JSON evaluation dataset (required for --mode ragas / both)",
    )
    ragas_g.add_argument(
        "--output",
        default="results/ragas_results.json",
        metavar="PATH",
        help="Where to write RAGAS JSON output  (default: results/ragas_results.json)",
    )
    ragas_g.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Groq LLM as faithfulness / relevance judge  (requires GROQ_API_KEY)",
    )
    ragas_g.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        metavar="MODEL",
        help="Groq model name  (default: llama-3.3-70b-versatile)",
    )

    # ── TruLens options ─────────────────────────────────────────────────────
    tl_g = p.add_argument_group("TruLens options")
    tl_g.add_argument(
        "--query",
        default="",
        metavar="TEXT",
        help="Single query to evaluate against the live pipeline",
    )
    tl_g.add_argument(
        "--patient-report",
        default="",
        metavar="TEXT",
        help="Patient report context for the live query",
    )
    tl_g.add_argument(
        "--cancer-filter",
        default="",
        metavar="TYPE",
        help="Cancer type filter for vector retrieval (e.g. 'breast', 'lung')",
    )
    tl_g.add_argument(
        "--trulens-output",
        default="results/trulens_record.json",
        metavar="PATH",
        help="Where to write the TruLens record JSON  (default: results/trulens_record.json)",
    )
    tl_g.add_argument(
        "--session-save",
        default="",
        metavar="PATH",
        help="Path to save the TruLens session file after all queries",
    )
    tl_g.add_argument(
        "--summary",
        default="",
        metavar="PATH",
        help="Print summary of an existing TruLens session file and exit",
    )
    tl_g.add_argument(
        "--demo",
        action="store_true",
        help="Run TruLens in offline demo mode (no pipeline connection needed)",
    )

    # ── General ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level  (default: WARNING)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all stdout output except errors",
    )

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Mode runners
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas_mode(args: argparse.Namespace) -> int:
    """Run RAGAS evaluation. Returns exit code (0 = success)."""
    from  evaluation.ragas_eval import RagasEvaluator

    if not args.dataset:
        print(
            "[ERROR] --dataset is required for --mode ragas.\n"
            "  Example: python evaluate.py --mode ragas --dataset data/eval_dataset.json",
            file=sys.stderr,
        )
        return 1

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset file not found: {dataset_path}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"\n{'='*55}")
        print(f"  RAGAS Evaluation")
        print(f"  Dataset : {dataset_path}")
        print(f"  Output  : {args.output}")
        print(f"  Mode    : {'LLM judge (Groq)' if args.use_llm else 'heuristic'}")
        print(f"{'='*55}\n")

    evaluator = RagasEvaluator(use_llm=args.use_llm, model=args.model)
    output    = evaluator.evaluate_dataset(
        dataset_path = str(dataset_path),
        output_path  = args.output,
        verbose      = not args.quiet,
    )

    if not args.quiet:
        _print_ragas_summary(output["aggregate"], output["n_samples"], output["mode"])

    return 0


def run_trulens_mode(args: argparse.Namespace) -> int:
    """Run TruLens evaluation. Returns exit code (0 = success)."""
    from evaluation.trulens_eval import TruLensEvaluator, _print_record, _offline_demo

    # ── Special case: summarise an existing session file ───────────────────
    if args.summary:
        _print_session_file(args.summary)
        return 0

    # ── Demo mode (no pipeline needed) ─────────────────────────────────────
    if args.demo or not args.query:
        if not args.quiet:
            print(
                "\n[TruLens] No --query provided. Running offline demo.\n"
                "  Use --query TEXT to evaluate against the live pipeline.\n"
            )
        _offline_demo()
        return 0

    # ── Live query ──────────────────────────────────────────────────────────
    if not args.quiet:
        print(f"\n{'='*55}")
        print(f"  TruLens Evaluation  (live pipeline)")
        print(f"  Query   : {args.query[:65]}")
        print(f"  Filter  : {args.cancer_filter or 'none'}")
        print(f"{'='*55}\n")

    evaluator = TruLensEvaluator(pipeline_available=True)
    record    = evaluator.evaluate(
        query          = args.query,
        patient_report = args.patient_report,
        cancer_filter  = args.cancer_filter,
    )

    if not args.quiet:
        _print_record(record)

    # Save single record
    _save_json(record, args.trulens_output, quiet=args.quiet)

    # Save session
    if args.session_save:
        evaluator.save_session(args.session_save)

    return 0


def run_both_mode(args: argparse.Namespace) -> int:
    """Run RAGAS then TruLens. Returns combined exit code."""
    if not args.quiet:
        print("\n" + "─" * 55)
        print("  Running RAGAS evaluation...")
        print("─" * 55)

    rc_ragas   = run_ragas_mode(args)

    if not args.quiet:
        print("\n" + "─" * 55)
        print("  Running TruLens evaluation...")
        print("─" * 55)

    rc_trulens = run_trulens_mode(args)

    return max(rc_ragas, rc_trulens)   # non-zero if either failed


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_ragas_summary(aggregate: dict, n_samples: int, mode: str):
    print(f"\n── RAGAS Summary ({'LLM judge' if mode == 'llm' else 'heuristic'}) ─────────────────")
    print(f"  Samples evaluated : {n_samples}")
    for k, v in aggregate.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<22} {v:.4f}  {bar}")
    print("─" * 55 + "\n")


def _print_session_file(path: str):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Session file not found: {p}", file=sys.stderr)
        return
    with open(p) as f:
        data = json.load(f)
    summary = data.get("summary", {})
    print(f"\n── TruLens Session: {p.name} ─────────────────────────")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"      {sk:<28} {sv}")
        else:
            print(f"  {k:<32} {v}")
    print("─" * 55)


def _save_json(data: dict, path: str, quiet: bool = False):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    if not quiet:
        print(f"\n[evaluate.py] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Sample dataset generator  (for quick local testing)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_sample_dataset(path: str = "data/sample_eval_dataset.json"):
    """
    Write a sample evaluation dataset to disk.
    Useful for first-time setup and CI testing.
    """
    samples = [
        {
            "id": "q001",
            "query": "What chemotherapy drugs are used to treat osteosarcoma?",
            "answer": (
                "Osteosarcoma is treated with neoadjuvant chemotherapy followed by surgery "
                "and adjuvant chemotherapy. The main drugs used are methotrexate with leucovorin "
                "rescue, doxorubicin, cisplatin, and ifosfamide. Please consult your oncologist."
            ),
            "contexts": [
                (
                    "Conventional treatment for OS consists of a combination of neoadjuvant "
                    "and adjuvant chemotherapy, and surgery. The four chemotherapy agents that "
                    "are in nearly all treatment regimens include methotrexate with leucovorin "
                    "rescue, doxorubicin, cisplatin, and ifosfamide."
                ),
                (
                    "Prior to the 1970s, chemotherapy was not used for osteosarcoma and survival "
                    "rates were dismal. The introduction of adjuvant chemotherapy increased "
                    "survival rates to 50%."
                ),
            ],
            "ground_truth": (
                "The standard chemotherapy for osteosarcoma includes methotrexate, doxorubicin, "
                "cisplatin, and ifosfamide, administered in neoadjuvant and adjuvant settings."
            ),
        },
        {
            "id": "q002",
            "query": "What is the 5-year survival rate for acute lymphoblastic leukemia in adults?",
            "answer": (
                "Despite a high rate of response to induction chemotherapy, only 30–40% of adult "
                "patients with ALL will achieve long-term remission. Elderly patients over 60 have "
                "particularly poor outcomes, with only 10–15% long-term survival. Please consult "
                "your oncologist for personalised information."
            ),
            "contexts": [
                (
                    "Despite a high rate of response to induction chemotherapy, only 30–40% of "
                    "adult patients with ALL will achieve long-term remission. Patients over the "
                    "age of 60 have particularly poor outcomes, with only 10–15% long-term survival."
                ),
                (
                    "The MRC UKALL XII/ECOG 2993 regimen yielded a complete response rate of 91% "
                    "and an overall 5-year survival of 38%."
                ),
            ],
            "ground_truth": (
                "Adult ALL has a 5-year survival of approximately 38%. Long-term remission is "
                "achieved in 30–40% of adults, with elderly patients having significantly worse "
                "outcomes (10–15% survival)."
            ),
        },
        {
            "id": "q003",
            "query": "What are the major subtypes of melanoma?",
            "answer": (
                "Melanoma is classified into four main subtypes: superficial spreading melanoma "
                "(SSM, ~70%), nodular melanoma (~15%), lentigo maligna melanoma (~10%), and acral "
                "lentiginous melanoma (~2%). SSM is the most common and grows outward first."
            ),
            "contexts": [
                (
                    "Clark divided melanoma into four main subtypes: superficial spreading melanoma "
                    "(70% of all melanomas), nodular melanoma (approximately 15%), lentigo maligna "
                    "melanoma (approximately 10%), and acral lentiginous melanoma (approximately 2%)."
                ),
            ],
            "ground_truth": (
                "The four main melanoma subtypes are SSM (70%), nodular melanoma (15%), "
                "lentigo maligna melanoma (10%), and acral lentiginous melanoma (2%)."
            ),
        },
        {
            "id": "q004",
            "query": "How does topical tamoxifen compare to oral tamoxifen for breast cancer?",
            "answer": (
                "Topical 4-OHT gel showed comparable anti-proliferative effects to oral tamoxifen "
                "with significantly lower plasma concentrations, suggesting equivalent local "
                "efficacy with reduced systemic exposure and potentially fewer side effects."
            ),
            "contexts": [
                (
                    "Lee et al. conducted a phase II randomized, double-blind, placebo-controlled "
                    "trial comparing oral tamoxifen (20 mg/day) with topical 4-OHT gel (4 mg/day). "
                    "The primary outcome, Ki-67 staining in DCIS lesions, showed post-therapy "
                    "decreases of 3.4% in the 4-OHT group and 5.1% in the oral tamoxifen group "
                    "(between-group p = 0.99). Mean 4-OHT concentrations in mammary adipose "
                    "tissue were comparable, while plasma levels differed significantly."
                ),
            ],
            "ground_truth": (
                "Topical 4-OHT gel achieves similar breast tissue concentrations and Ki-67 "
                "reduction compared to oral tamoxifen, but with much lower plasma levels, "
                "indicating reduced systemic toxicity."
            ),
        },
        {
            "id": "q005",
            "query": "What ALK inhibitors are approved for non-small cell lung cancer?",
            "answer": (
                "Five ALK tyrosine kinase inhibitors (TKIs) are approved for ALK-positive NSCLC: "
                "crizotinib (1st generation), ceritinib and alectinib and brigatinib (2nd "
                "generation), and lorlatinib (3rd generation). Each successive generation was "
                "developed to overcome resistance to the prior generation."
            ),
            "contexts": [
                (
                    "The therapeutic development in this domain has led to the implementation of "
                    "three generations of ALK tyrosine kinase inhibitors (TKI) with five molecules "
                    "approved into clinical practice: crizotinib, ceritinib, alectinib, brigatinib, "
                    "and lorlatinib."
                ),
                (
                    "Crizotinib was the first oral TKI molecule approved in 2011 by the FDA for "
                    "metastatic NSCLC patients with ALK mutation. Ceritinib was the initial ALK "
                    "TKI molecule of the second-generation class approved to overcome resistance "
                    "to crizotinib. Lorlatinib is a third-generation TKI approved in 2018."
                ),
            ],
            "ground_truth": (
                "Five ALK TKIs are FDA-approved for ALK-positive NSCLC: crizotinib (1st gen), "
                "ceritinib, alectinib, brigatinib (2nd gen), and lorlatinib (3rd gen)."
            ),
        },
    ]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"[evaluate.py] Sample dataset written → {out}  ({len(samples)} samples)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level    = getattr(logging, args.log_level, logging.WARNING),
        format   = "%(levelname)-8s  %(name)s  %(message)s",
        stream   = sys.stderr,
    )

    # ── Special: generate sample dataset if no dataset given and mode=ragas ─
    if args.mode in ("ragas", "both") and not args.dataset:
        sample_path = "data/sample_eval_dataset.json"
        if not Path(sample_path).exists():
            if not args.quiet:
                print(
                    f"[evaluate.py] No --dataset provided.  "
                    f"Generating sample dataset at {sample_path} ..."
                )
            _generate_sample_dataset(sample_path)
        args.dataset = sample_path

    # ── Dispatch ──────────────────────────────────────────────────────────
    dispatch = {
        "ragas":   run_ragas_mode,
        "trulens": run_trulens_mode,
        "both":    run_both_mode,
    }

    runner  = dispatch.get(args.mode, run_ragas_mode)
    exit_code = runner(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
