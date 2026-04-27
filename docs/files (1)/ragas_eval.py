"""
evaluation/ragas_eval.py
------------------------
RAGAS-style evaluation for the Medical GraphRAG pipeline.

Implements the four core RAGAS metrics WITHOUT the ragas package:
  1. faithfulness        — are answer claims grounded in the retrieved context?
  2. answer_relevance    — does the answer address the query?
  3. context_precision   — are retrieved chunks relevant to the answer?
  4. context_recall      — do the chunks cover the ground-truth answer?

Plus two bonus metrics tailored to this pipeline:
  5. answer_correctness  — token F1 + cosine vs ground truth
  6. noise_sensitivity   — answer stability when context is perturbed

Dataset format  (JSON file, list of records):
[
  {
    "id":           "q001",                        ← optional
    "query":        "What drugs treat osteosarcoma?",
    "answer":       "Osteosarcoma is treated with cisplatin ...",
    "contexts":     ["Osteosarcoma treatment ...", "Cisplatin ..."],
    "ground_truth": "The standard chemotherapy is ..."   ← optional
  },
  ...
]

Output format:
{
  "dataset":    "path/to/dataset.json",
  "mode":       "heuristic" | "llm",
  "n_samples":  42,
  "results":    [ { per-sample scores } ],
  "aggregate":  { mean scores across all samples }
}

Usage (standalone):
    python ragas_eval.py --dataset data/eval_dataset.json --output results/ragas.json
    python ragas_eval.py --dataset data/eval_dataset.json --use-llm   # Groq judge
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
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities  (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

_STOP = frozenset({
    "a","an","the","is","are","was","were","in","on","at","to","of",
    "for","with","by","from","and","or","but","not","it","its","be",
    "been","being","have","has","had","do","does","did","will","would",
})
_PUNCT = str.maketrans("", "", string.punctuation)


def _tokens(text: str) -> List[str]:
    return [w for w in text.lower().translate(_PUNCT).split() if w not in _STOP]


def _cosine(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    vocab  = list(set(ta + tb))
    ca, cb = Counter(ta), Counter(tb)
    la, lb = len(ta), len(tb)
    dot    = sum((ca[w] / la) * (cb[w] / lb) for w in vocab)
    na     = math.sqrt(sum((ca[w] / la) ** 2 for w in vocab))
    nb     = math.sqrt(sum((cb[w] / lb) ** 2 for w in vocab))
    return round(dot / (na * nb), 4) if (na and nb) else 0.0


def _token_f1(pred: str, ref: str) -> float:
    pc, rc = Counter(_tokens(pred)), Counter(_tokens(ref))
    common = sum((pc & rc).values())
    if not common:
        return 0.0
    prec = common / sum(pc.values())
    rec  = common / sum(rc.values())
    return round(2 * prec * rec / (prec + rec), 4)


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# LLM Judge  (optional Groq backend)
# ─────────────────────────────────────────────────────────────────────────────

class _GroqJudge:
    """Thin wrapper around Groq for LLM-based RAGAS scoring."""

    # System prompts kept short to stay within Groq token limits
    _SYS_FAITHFULNESS = (
        "You are a strict medical fact-checker. "
        "Given CONTEXT and CLAIMS, decide which claims are directly supported by the context. "
        "Reply with JSON only: "
        '{{"supported": [<claim>, ...], "unsupported": [<claim>, ...]}}'
    )
    _SYS_RELEVANCE = (
        "You are a QA evaluator. Given a QUESTION and ANSWER, "
        "generate 3 alternative phrasings of the question that the answer also responds to. "
        'Reply with JSON only: {{"questions": ["...", "...", "..."]}}'
    )
    _SYS_CORRECTNESS = (
        "Rate factual correctness of a GENERATED answer vs a GROUND TRUTH answer (0.0–1.0). "
        'Reply with JSON only: {{"score": <float>, "reason": "<one sentence>"}}'
    )

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set")
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model  = model

    def _call(self, system: str, user: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw   = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except Exception as exc:
            log.warning("[GroqJudge] call failed: %s", exc)
            return {}

    def faithfulness(self, context: str, claims: List[str]) -> Tuple[float, List[str]]:
        user   = f"CONTEXT:\n{context[:3000]}\n\nCLAIMS:\n" + "\n".join(f"- {c}" for c in claims)
        result = self._call(self._SYS_FAITHFULNESS, user)
        supported   = result.get("supported",   [])
        unsupported = result.get("unsupported", [])
        total = len(claims)
        score = len(supported) / total if total > 0 else 0.0
        return round(score, 4), unsupported

    def answer_relevance(self, query: str, answer: str) -> float:
        user   = f"QUESTION: {query}\nANSWER: {answer[:1500]}"
        result = self._call(self._SYS_RELEVANCE, user)
        qs     = result.get("questions", [])
        if not qs:
            return 0.0
        return round(sum(_cosine(query, q) for q in qs) / len(qs), 4)

    def answer_correctness(self, ground_truth: str, answer: str) -> float:
        user   = f"GROUND TRUTH:\n{ground_truth[:1500]}\n\nGENERATED:\n{answer[:1500]}"
        result = self._call(self._SYS_CORRECTNESS, user)
        return round(float(result.get("score", 0.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS Metrics  (heuristic implementations)
# ─────────────────────────────────────────────────────────────────────────────

_FAITHFULNESS_THRESHOLD = 0.25   # min cosine for a claim to be "grounded"
_CONTEXT_THRESHOLD      = 0.20   # min cosine for a chunk to be "relevant"


def compute_faithfulness(
    answer:   str,
    contexts: List[str],
    judge:    Optional[_GroqJudge] = None,
) -> Tuple[float, List[str]]:
    """
    Fraction of answer sentences (claims) that are grounded in the contexts.
    LLM mode: uses Groq to determine support.
    Heuristic mode: TF-IDF cosine ≥ threshold.
    """
    combined = " ".join(contexts)
    claims   = _sentences(answer)
    if not claims:
        return 0.0, []

    if judge:
        return judge.faithfulness(combined, claims)

    supported   = [c for c in claims if _cosine(c, combined) >= _FAITHFULNESS_THRESHOLD]
    unsupported = [c for c in claims if c not in supported]
    return round(len(supported) / len(claims), 4), unsupported


def compute_answer_relevance(
    query:  str,
    answer: str,
    judge:  Optional[_GroqJudge] = None,
) -> float:
    """
    How well does the answer address the query?
    LLM mode: generate reverse-questions and measure similarity to original.
    Heuristic mode: TF-IDF cosine(query, answer).
    """
    if judge:
        return judge.answer_relevance(query, answer)
    return _cosine(query, answer)


def compute_context_precision(
    contexts:       List[str],
    expected_answer: str,
    threshold:       float = _CONTEXT_THRESHOLD,
) -> float:
    """
    Fraction of retrieved chunks that are relevant to the expected answer.
    Relevance: cosine(chunk, expected_answer) ≥ threshold.
    """
    if not contexts or not expected_answer:
        return 0.0
    relevant = sum(1 for c in contexts if _cosine(c, expected_answer) >= threshold)
    return round(relevant / len(contexts), 4)


def compute_context_recall(
    contexts:    List[str],
    ground_truth: str,
    threshold:    float = _CONTEXT_THRESHOLD,
) -> float:
    """
    Fraction of ground-truth sentences attributable to at least one retrieved chunk.
    """
    if not ground_truth or not contexts:
        return 0.0
    gt_sentences = _sentences(ground_truth)
    if not gt_sentences:
        return 0.0
    combined = " ".join(contexts)
    covered  = sum(1 for s in gt_sentences if _cosine(s, combined) >= threshold)
    return round(covered / len(gt_sentences), 4)


def compute_answer_correctness(
    answer:       str,
    ground_truth: str,
    judge:        Optional[_GroqJudge] = None,
) -> float:
    """
    Factual correctness: token-F1 + cosine vs ground truth.
    LLM mode: Groq rates 0–1.
    """
    if not ground_truth:
        return 0.0
    if judge:
        return judge.answer_correctness(ground_truth, answer)
    return round(0.5 * _token_f1(answer, ground_truth)
                 + 0.5 * _cosine(answer, ground_truth), 4)


def compute_noise_sensitivity(
    answer:   str,
    contexts: List[str],
    ratio:    float = 0.30,
) -> float:
    """
    Measure answer stability by injecting noise into the context.
    Noise: reverse every token in `ratio` proportion of the text.
    Returns stability in [0, 1]  (1 = not sensitive; 0 = fully unstable).
    """
    import random, copy
    noisy_contexts = []
    for ctx in contexts:
        words   = ctx.split()
        n_noisy = int(len(words) * ratio)
        indices = random.sample(range(len(words)), min(n_noisy, len(words)))
        words_c = words[:]
        for idx in indices:
            words_c[idx] = words_c[idx][::-1]
        noisy_contexts.append(" ".join(words_c))

    orig_faith,  _ = compute_faithfulness(answer, contexts)
    noisy_faith, _ = compute_faithfulness(answer, noisy_contexts)
    drop = max(0.0, orig_faith - noisy_faith)
    return round(1.0 - min(drop, 1.0), 4)


def ragas_score(
    faithfulness:     float,
    answer_relevance: float,
    context_precision: float,
    context_recall:   float,
) -> float:
    """
    Harmonic mean of the four core RAGAS metrics.
    Following the original RAGAS paper definition.
    """
    values = [faithfulness, answer_relevance, context_precision, context_recall]
    nonzero = [v for v in values if v > 0]
    if not nonzero:
        return 0.0
    hm = len(values) / sum(1 / v for v in nonzero)
    return round(hm, 4)


# ─────────────────────────────────────────────────────────────────────────────
# RagasEvaluator  (dataset-level runner)
# ─────────────────────────────────────────────────────────────────────────────

class RagasEvaluator:
    """
    Evaluate a dataset of query/answer/context samples.

    Parameters
    ----------
    use_llm : bool
        Use Groq LLM judge for faithfulness, answer relevance, and correctness.
        Falls back to heuristics automatically if Groq is unavailable.
    model : str
        Groq model name. Default: llama-3.3-70b-versatile.
    """

    def __init__(
        self,
        use_llm: bool = False,
        model:   str  = "llama-3.3-70b-versatile",
    ):
        self.judge: Optional[_GroqJudge] = None
        self.mode  = "heuristic"

        if use_llm:
            try:
                self.judge = _GroqJudge(model=model)
                self.mode  = "llm"
                log.info("[RagasEvaluator] Groq judge enabled (%s)", model)
            except Exception as exc:
                log.warning("[RagasEvaluator] Groq unavailable (%s) — using heuristics", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Single-sample evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_sample(self, sample: dict) -> dict:
        """
        Evaluate one sample dict.

        Required keys : query, answer, contexts (list of str)
        Optional keys : ground_truth (str), id (str)
        """
        query        = sample["query"]
        answer       = sample["answer"]
        contexts     = sample.get("contexts", [])
        ground_truth = sample.get("ground_truth", "")

        faith, unsupported = compute_faithfulness(answer, contexts, self.judge)
        ans_rel            = compute_answer_relevance(query, answer, self.judge)
        ctx_prec           = compute_context_precision(contexts, answer)
        ctx_rec            = compute_context_recall(contexts, ground_truth)
        ans_corr           = compute_answer_correctness(answer, ground_truth, self.judge)
        noise_sens         = compute_noise_sensitivity(answer, contexts)
        overall            = ragas_score(faith, ans_rel, ctx_prec, ctx_rec)

        return {
            "id":                  sample.get("id", ""),
            "query":               query,
            "faithfulness":        faith,
            "answer_relevance":    ans_rel,
            "context_precision":   ctx_prec,
            "context_recall":      ctx_rec,
            "answer_correctness":  ans_corr,
            "noise_sensitivity":   noise_sens,
            "ragas_score":         overall,
            "unsupported_claims":  unsupported,
            "ground_truth_provided": bool(ground_truth),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Dataset-level evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_dataset(
        self,
        dataset_path: str,
        output_path:  Optional[str] = None,
        verbose:      bool = True,
    ) -> dict:
        """
        Load a JSON dataset, evaluate every sample, and save results.

        Parameters
        ----------
        dataset_path : path to JSON dataset file
        output_path  : where to write results (optional)
        verbose      : print progress to stdout

        Returns
        -------
        Full results dict (also written to output_path if provided)
        """
        # Load dataset
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if not isinstance(samples, list):
            raise ValueError("Dataset must be a JSON array of sample objects")

        if verbose:
            print(f"[RAGAS] Evaluating {len(samples)} samples  (mode={self.mode})")

        # Evaluate
        t0      = time.perf_counter()
        results = []
        for i, sample in enumerate(samples):
            if verbose:
                print(f"  [{i+1}/{len(samples)}] {sample.get('query', '')[:60]}...", end="\r")
            try:
                results.append(self.evaluate_sample(sample))
            except Exception as exc:
                log.error("Sample %d failed: %s", i, exc)
                results.append({"id": sample.get("id", i), "error": str(exc)})

        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        if verbose:
            print(f"\n[RAGAS] Done in {elapsed:.0f} ms")

        # Aggregate
        aggregate = self._aggregate(results)

        output = {
            "dataset":    str(dataset_path),
            "mode":       self.mode,
            "n_samples":  len(results),
            "elapsed_ms": elapsed,
            "results":    results,
            "aggregate":  aggregate,
        }

        # Save
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            if verbose:
                print(f"[RAGAS] Results saved → {out}")

        return output

    def _aggregate(self, results: List[dict]) -> dict:
        keys = [
            "faithfulness", "answer_relevance", "context_precision",
            "context_recall", "answer_correctness", "noise_sensitivity", "ragas_score",
        ]
        agg = {}
        for k in keys:
            vals = [r[k] for r in results if k in r and isinstance(r[k], (int, float))]
            agg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        return agg


# ─────────────────────────────────────────────────────────────────────────────
# CLI (when run directly: python ragas_eval.py ...)
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragas_eval.py",
        description="RAGAS-style evaluation for the Medical GraphRAG pipeline",
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to JSON evaluation dataset file",
    )
    p.add_argument(
        "--output", default="results/ragas_results.json",
        help="Where to write the JSON output (default: results/ragas_results.json)",
    )
    p.add_argument(
        "--use-llm", action="store_true",
        help="Use Groq LLM judge instead of heuristic metrics (requires GROQ_API_KEY)",
    )
    p.add_argument(
        "--model", default="llama-3.3-70b-versatile",
        help="Groq model name (default: llama-3.3-70b-versatile)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return p


def main_ragas(args: Optional[argparse.Namespace] = None):
    """Entry point — called by evaluate.py --mode ragas, or directly."""
    if args is None:
        parser = _build_parser()
        args   = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s  %(message)s",
    )

    evaluator = RagasEvaluator(use_llm=getattr(args, "use_llm", False),
                               model=getattr(args, "model", "llama-3.3-70b-versatile"))
    output    = evaluator.evaluate_dataset(
        dataset_path = args.dataset,
        output_path  = args.output,
        verbose      = not getattr(args, "quiet", False),
    )

    # Print aggregate summary
    print("\n── RAGAS Aggregate Scores ──────────────────────")
    for k, v in output["aggregate"].items():
        bar   = "█" * int(v * 20)
        print(f"  {k:<22} {v:.4f}  {bar}")
    print(f"  {'n_samples':<22} {output['n_samples']}")
    print(f"  {'mode':<22} {output['mode']}")
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    main_ragas()
