"""
live_eval.py
─────────────────────────────────────────────────────────────────────────────
Live query evaluation module for the MedChat RAG pipeline.

Runs lightweight, non-blocking RAGAS-compatible metrics on EVERY user query
immediately after an answer is generated.  Results are:
  • printed to terminal / logs
  • appended to  live_eval_logs.json  (one record per query)

This module is imported by cancer_app_v2.py — see integration note at the
bottom of this file for the two-line change required.

Scoring approach
─────────────────
For live evaluation we avoid calling an external LLM judge on every query
(too slow + costly for interactive use).  Instead we compute two lightweight
scores locally, and optionally run the full RAGAS faithfulness check if
OPENAI_API_KEY is set.

Local scores (always fast, no API calls)
  • context_coverage    : lexical overlap between answer and retrieved contexts
  • answer_length_ratio : proxy for answer completeness relative to question

Full RAGAS score (only if OPENAI_API_KEY is set, async/threaded)
  • faithfulness        : RAGAS LLM-judged faithfulness

Design constraints
──────────────────
  • MUST NOT block the Streamlit response loop (runs in a background thread)
  • MUST reuse the already-generated answer + already-retrieved contexts
    (no second pipeline call)
  • Saves to live_eval_logs.json with append semantics (one JSON object per
    line = NDJSON format so the file is always parseable even mid-write)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("live_eval")

PROJECT_ROOT   = Path(__file__).resolve().parent
LIVE_LOG_PATH  = PROJECT_ROOT / "live_eval_logs.json"
LIVE_LOG_LOCK  = threading.Lock()

# ── tune these constants to balance cost vs. coverage ────────────────────────
ENABLE_RAGAS_LIVE     = os.getenv("LIVE_EVAL_RAGAS", "0") == "1"  # off by default
RAGAS_LIVE_LLM_MODEL  = "gpt-4o-mini"
RAGAS_LIVE_EMBED      = "text-embedding-3-small"


# ─────────────────────────────────────────────────────────────────────────────
# Local lightweight metrics  (no external API)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    stop = {
        "the","a","an","is","are","was","were","of","in","on","at","to","and",
        "or","but","for","not","this","that","it","be","been","have","has","had",
        "with","by","from","as","its","their","they","we","our","you","your",
        "i","my","he","she","his","her","can","will","may","should","would",
    }
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return {t for t in tokens if t not in stop}


def _context_coverage(answer: str, contexts: list[str]) -> float:
    """
    Fraction of meaningful answer tokens that appear in at least one context.
    Range: 0.0 → 1.0.
    """
    if not answer or not contexts:
        return 0.0
    answer_tokens   = _tokenise(answer)
    context_tokens  = _tokenise(" ".join(contexts))
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return round(len(overlap) / len(answer_tokens), 4)


def _answer_length_ratio(question: str, answer: str) -> float:
    """
    Log-normalised ratio of answer length to question length.
    Proxy: very short answers on complex questions score low.
    Capped at 1.0.
    """
    q_len = max(len(question.split()), 1)
    a_len = len(answer.split())
    if a_len == 0:
        return 0.0
    ratio = math.log1p(a_len) / math.log1p(q_len * 3)
    return round(min(ratio, 1.0), 4)


def _is_fallback_answer(answer: str) -> bool:
    """Detect whether the answer is a web-fallback or insufficient-context response."""
    from graphrag_integration import WEB_FALLBACK_MARKER
    if WEB_FALLBACK_MARKER in answer:
        return True
    fallback_phrases = [
        "i do not have enough information",
        "not available in the provided context",
        "web search result",
    ]
    lower = answer.lower()
    return any(p in lower for p in fallback_phrases)


# ─────────────────────────────────────────────────────────────────────────────
# Optional: full RAGAS faithfulness (background thread, non-blocking)
# ─────────────────────────────────────────────────────────────────────────────

def _ragas_faithfulness_single(
    question:  str,
    answer:    str,
    contexts:  list[str],
) -> Optional[float]:
    """
    Run RAGAS faithfulness on a single sample using OpenAI judge.
    Returns None if RAGAS is not available or OPENAI_API_KEY is not set.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        return None

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        return None

    try:
        ds = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input         = question,
                response           = answer,
                retrieved_contexts = contexts,
                reference          = answer,   # no ground truth at live time
            )
        ])
        llm   = ChatOpenAI(model=RAGAS_LIVE_LLM_MODEL, api_key=openai_key)
        embed = OpenAIEmbeddings(model=RAGAS_LIVE_EMBED, api_key=openai_key)
        result = ragas_evaluate(
            dataset          = ds,
            metrics          = [faithfulness],
            llm              = llm,
            embeddings       = embed,
            raise_exceptions = False,
        )
        val = result.get("faithfulness")
        return round(float(val), 4) if val is not None else None
    except Exception as exc:
        log.debug("live RAGAS faithfulness failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Log writer
# ─────────────────────────────────────────────────────────────────────────────

def _append_log(record: dict) -> None:
    """Thread-safe append of one NDJSON record to live_eval_logs.json."""
    line = json.dumps(record, ensure_ascii=False)
    with LIVE_LOG_LOCK:
        with open(LIVE_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called from cancer_app_v2.py
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_live(
    question:      str,
    answer:        str,
    contexts:      list[str],
    cancer_filter: str = "",
    metadata:      Optional[dict] = None,
) -> None:
    """
    Evaluate a single live query result in a background thread.

    Parameters
    ----------
    question      : the user's query string
    answer        : the generated answer returned by the pipeline
    contexts      : list of raw context strings retrieved by the pipeline
    cancer_filter : the active cancer-type filter (for logging)
    metadata      : optional dict with extra fields to store in the log

    This function returns immediately.  All work happens in a daemon thread.
    """
    # capture arguments for the thread closure
    _q, _a, _c = question, answer, list(contexts)

    def _worker():
        t0 = time.monotonic()

        # ── local scores (always) ─────────────────────────────────────────────
        coverage = _context_coverage(_a, _c)
        length_r = _answer_length_ratio(_q, _a)
        is_fb    = _is_fallback_answer(_a)

        # ── optional full RAGAS faithfulness ─────────────────────────────────
        ragas_faith: Optional[float] = None
        if ENABLE_RAGAS_LIVE and not is_fb and _c:
            ragas_faith = _ragas_faithfulness_single(_q, _a, _c)

        elapsed = round(time.monotonic() - t0, 3)

        record = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "question":         _q,
            "answer_snippet":   _a[:200].replace("\n", " "),
            "n_contexts":       len(_c),
            "cancer_filter":    cancer_filter,
            "is_fallback":      is_fb,
            "scores": {
                "context_coverage":    coverage,
                "answer_length_ratio": length_r,
                "ragas_faithfulness":  ragas_faith,
            },
            "eval_latency_sec": elapsed,
            "metadata":         metadata or {},
        }

        # ── print to terminal ─────────────────────────────────────────────────
        faith_str = f"faith={ragas_faith:.3f}" if ragas_faith is not None else ""
        fb_tag    = " [FALLBACK]" if is_fb else ""
        log.info(
            "[LiveEval%s] cov=%.3f  len_ratio=%.3f  %s  (%ds, %d ctx)  Q: %s…",
            fb_tag, coverage, length_r, faith_str, elapsed, len(_c), _q[:55]
        )

        # ── append to log file ────────────────────────────────────────────────
        try:
            _append_log(record)
        except Exception as exc:
            log.warning("live_eval log write failed: %s", exc)

    thread = threading.Thread(target=_worker, daemon=True, name="live_eval")
    thread.start()


# ─────────────────────────────────────────────────────────────────────────────
# CLI utility: read and summarise the live log
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_live_log(path: Path = LIVE_LOG_PATH) -> None:
    if not path.exists():
        print(f"No live log found at {path}")
        return

    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        print("Live log is empty.")
        return

    n          = len(records)
    n_fallback = sum(1 for r in records if r.get("is_fallback"))
    covs       = [r["scores"]["context_coverage"]    for r in records if r["scores"].get("context_coverage")    is not None]
    lens       = [r["scores"]["answer_length_ratio"]  for r in records if r["scores"].get("answer_length_ratio") is not None]
    faiths     = [r["scores"]["ragas_faithfulness"]   for r in records if r["scores"].get("ragas_faithfulness")  is not None]

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    print("\n" + "═" * 60)
    print("  LIVE EVAL LOG SUMMARY")
    print("═" * 60)
    print(f"  Total queries logged : {n}")
    print(f"  Fallback answers     : {n_fallback} ({100*n_fallback/max(1,n):.1f}%)")
    print()
    print(f"  Avg context coverage    : {avg(covs)}")
    print(f"  Avg answer length ratio : {avg(lens)}")
    print(f"  Avg RAGAS faithfulness  : {avg(faiths) if faiths else 'not computed'}")
    print(f"  (faithfulness requires LIVE_EVAL_RAGAS=1 + OPENAI_API_KEY)")
    print("═" * 60 + "\n")
    print(f"  Log file: {path}")
    print(f"  Last 3 queries:")
    for r in records[-3:]:
        ts  = r.get("timestamp","")[:19]
        cov = r["scores"].get("context_coverage", "?")
        fb  = " [FB]" if r.get("is_fallback") else ""
        print(f"    {ts}{fb}  cov={cov}  Q: {r['question'][:60]}…")
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Summarise the live eval log")
    p.add_argument("--log", type=Path, default=LIVE_LOG_PATH)
    args = p.parse_args()
    _summarise_live_log(args.log)
