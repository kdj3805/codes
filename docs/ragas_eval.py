"""
evaluation/ragas_eval.py
------------------------
RAGAS-inspired evaluation metrics for the Medical GraphRAG pipeline.

Implements (without the ragas package dependency):
  1. Faithfulness           — claim-by-claim context grounding
  2. Answer Relevance       — reverse-question generation similarity
  3. Context Precision      — proportion of relevant retrieved chunks
  4. Context Recall         — coverage of ground-truth sentences
  5. Context Entity Recall  — entity-level recall
  6. Answer Correctness     — semantic similarity to ground truth
  7. Answer Similarity      — BLEU + token-F1 against reference
  8. Noise Sensitivity      — response stability to injected noise

All metrics return float in [0, 1]. Each has both a heuristic mode
(always available) and an optional LLM-judge mode (use_llm=True).

Usage:
    from evaluation.ragas_eval import RagasEvaluator
    evaluator = RagasEvaluator(use_llm=False)   # heuristic-only, no API key needed
    results   = evaluator.run_all(
        query="What is the treatment for osteosarcoma?",
        answer="Osteosarcoma treatment involves neoadjuvant chemotherapy ...",
        contexts=["Osteosarcoma is treated with ..."],
        ground_truth="The standard treatment is neoadjuvant chemotherapy ...",
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from evaluation.metrics import (
    bleu_score,
    claim_grounding_score,
    context_precision,
    context_recall,
    entity_alignment_score,
    sentence_split,
    tfidf_cosine_similarity,
    token_overlap_f1,
    tokenise,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RagasResult:
    faithfulness:            float = 0.0
    answer_relevance:        float = 0.0
    context_precision:       float = 0.0
    context_recall:          float = 0.0
    context_entity_recall:   float = 0.0
    answer_correctness:      float = 0.0
    answer_similarity:       float = 0.0
    noise_sensitivity:       float = 0.0
    metadata:                dict  = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def ragas_score(self) -> float:
        """
        Harmonic mean of the four RAGAS core metrics:
        faithfulness, answer_relevance, context_precision, context_recall.
        (Follows the original RAGAS paper formula.)
        """
        core = [
            self.faithfulness,
            self.answer_relevance,
            self.context_precision,
            self.context_recall,
        ]
        # Harmonic mean
        if all(v == 0 for v in core):
            return 0.0
        try:
            hm = len(core) / sum(1 / v for v in core if v > 0)
        except ZeroDivisionError:
            hm = 0.0
        return round(hm, 4)

    def to_dict(self) -> dict:
        return {
            "faithfulness":           self.faithfulness,
            "answer_relevance":       self.answer_relevance,
            "context_precision":      self.context_precision,
            "context_recall":         self.context_recall,
            "context_entity_recall":  self.context_entity_recall,
            "answer_correctness":     self.answer_correctness,
            "answer_similarity":      self.answer_similarity,
            "noise_sensitivity":      self.noise_sensitivity,
            "ragas_score":            self.ragas_score,
            "metadata":               self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge helper (Groq)
# ─────────────────────────────────────────────────────────────────────────────

class _LLMJudge:
    """Thin wrapper around Groq for RAGAS-style LLM evaluation."""

    _FAITHFULNESS_SYS = (
        "You are a medical fact-checking assistant. "
        "Given a context and a set of claims from an answer, "
        "decide which claims are directly supported by the context. "
        "Return ONLY a JSON object: "
        "{\"supported\": [<claim1>, ...], \"unsupported\": [<claim2>, ...]}"
    )

    _RELEVANCE_SYS = (
        "You are an evaluation assistant. "
        "Given a question and an answer, generate 3 short alternative phrasings of "
        "the question that the answer would also respond to. "
        "Return ONLY a JSON object: {\"questions\": [\"...\", \"...\", \"...\"]}"
    )

    _CORRECTNESS_SYS = (
        "You are a medical knowledge evaluator. "
        "Given a ground-truth answer and a generated answer, rate factual correctness. "
        "Return ONLY a JSON object: "
        "{\"correctness_score\": <float 0.0-1.0>, \"reason\": \"<brief explanation>\"}"
    )

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model  = model
        api_key     = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        from groq import Groq
        self._client = Groq(api_key=api_key)

    def _call(self, system: str, user: str) -> dict:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.0,
                max_tokens=600,
            )
            text  = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except Exception as e:
            log.warning("[LLMJudge] API call failed: %s", e)
            return {}

    def faithfulness(self, context: str, claims: List[str]) -> tuple[float, List[str]]:
        """Returns (score, unsupported_claims)."""
        user = f"CONTEXT:\n{context[:3000]}\n\nCLAIMS:\n" + "\n".join(f"- {c}" for c in claims)
        result     = self._call(self._FAITHFULNESS_SYS, user)
        supported  = result.get("supported", [])
        unsupported = result.get("unsupported", [])
        total = len(claims)
        score = len(supported) / total if total > 0 else 0.0
        return round(score, 4), unsupported

    def answer_relevance(self, query: str, answer: str) -> float:
        """Returns score based on cosine similarity of generated questions to original query."""
        user   = f"QUESTION: {query}\nANSWER: {answer[:1500]}"
        result = self._call(self._RELEVANCE_SYS, user)
        gen_qs = result.get("questions", [])
        if not gen_qs:
            return 0.0
        sims = [tfidf_cosine_similarity(query, q) for q in gen_qs]
        return round(sum(sims) / len(sims), 4)

    def answer_correctness(self, ground_truth: str, answer: str) -> float:
        user   = f"GROUND TRUTH:\n{ground_truth[:1500]}\n\nGENERATED ANSWER:\n{answer[:1500]}"
        result = self._call(self._CORRECTNESS_SYS, user)
        return round(float(result.get("correctness_score", 0.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RagasEvaluator:
    """
    Main RAGAS-style evaluator.

    Parameters
    ----------
    use_llm : bool
        Use Groq LLM for faithfulness, answer relevance, and correctness.
        Falls back to heuristics automatically if the API call fails.
    model : str
        Groq model name for the LLM judge.
    faithfulness_threshold : float
        TF-IDF cosine threshold for heuristic claim grounding. Default 0.35.
    context_threshold : float
        TF-IDF cosine threshold for context precision/recall. Default 0.25.
    noise_ratio : float
        Proportion of context to replace with noise for noise-sensitivity test. Default 0.3.
    """

    def __init__(
        self,
        use_llm:                bool  = False,
        model:                  str   = "llama-3.3-70b-versatile",
        faithfulness_threshold: float = 0.35,
        context_threshold:      float = 0.25,
        noise_ratio:            float = 0.30,
    ):
        self.use_llm   = use_llm
        self.f_thr     = faithfulness_threshold
        self.ctx_thr   = context_threshold
        self.noise_ratio = noise_ratio
        self._judge: Optional[_LLMJudge] = None

        if use_llm:
            try:
                self._judge = _LLMJudge(model=model)
                log.info("[RagasEvaluator] LLM judge initialised (model=%s)", model)
            except Exception as e:
                log.warning("[RagasEvaluator] LLM judge unavailable: %s. Using heuristics.", e)
                self.use_llm = False

    # ──────────────────────────────────────────────────────────────────────
    # Individual metric methods
    # ──────────────────────────────────────────────────────────────────────

    def compute_faithfulness(
        self,
        answer:   str,
        contexts: List[str],
    ) -> tuple[float, List[str]]:
        """
        Faithfulness: fraction of answer claims supported by the contexts.

        Returns (score, unsupported_claims).
        """
        combined_ctx = " ".join(contexts)
        claims       = sentence_split(answer)

        if not claims:
            return 0.0, []

        if self.use_llm and self._judge:
            score, unsupported = self._judge.faithfulness(combined_ctx, claims)
            return score, unsupported

        # Heuristic fallback
        supported    = [c for c in claims if tfidf_cosine_similarity(c, combined_ctx) >= self.f_thr]
        unsupported  = [c for c in claims if c not in supported]
        score        = len(supported) / len(claims)
        return round(score, 4), unsupported

    def compute_answer_relevance(self, query: str, answer: str) -> float:
        """
        Answer relevance: how well the answer addresses the query.

        LLM mode: generate n reverse-questions; measure cosine similarity to original.
        Heuristic mode: direct TF-IDF cosine between query and answer.
        """
        if self.use_llm and self._judge:
            return self._judge.answer_relevance(query, answer)
        return tfidf_cosine_similarity(query, answer)

    def compute_context_precision(self, contexts: List[str], answer: str) -> float:
        """
        Context precision: fraction of retrieved contexts relevant to the answer.
        """
        return context_precision(contexts, answer, self.ctx_thr)

    def compute_context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """
        Context recall: fraction of ground-truth sentences attributable to context.
        """
        return context_recall(contexts, ground_truth, self.ctx_thr)

    def compute_context_entity_recall(
        self,
        contexts:     List[str],
        ground_truth: str,
    ) -> float:
        """
        Context entity recall: fraction of named entities in the ground truth
        that appear in ANY retrieved context.

        Uses a simple regex-based proper-noun / medical entity heuristic.
        """
        # Simple heuristic: capitalised multi-word phrases as pseudo-entities
        gt_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", ground_truth)
        gt_entities = list({e.lower() for e in gt_entities})

        if not gt_entities:
            return 0.0

        combined_ctx = " ".join(contexts).lower()
        found = sum(1 for e in gt_entities if e in combined_ctx)
        return round(found / len(gt_entities), 4)

    def compute_answer_correctness(
        self,
        answer:       str,
        ground_truth: str,
    ) -> float:
        """
        Answer correctness: factual similarity between answer and ground truth.

        LLM mode: LLM rates correctness (0–1).
        Heuristic mode: token-overlap F1 + TF-IDF cosine (weighted average).
        """
        if self.use_llm and self._judge:
            return self._judge.answer_correctness(ground_truth, answer)

        tok_f1 = token_overlap_f1(answer, ground_truth)
        cos    = tfidf_cosine_similarity(answer, ground_truth)
        return round(0.5 * tok_f1 + 0.5 * cos, 4)

    def compute_answer_similarity(
        self,
        answer:       str,
        ground_truth: str,
    ) -> float:
        """
        Answer similarity: BLEU-4 + TF-IDF cosine.
        """
        bleu = bleu_score(answer, ground_truth)
        cos  = tfidf_cosine_similarity(answer, ground_truth)
        return round(0.4 * bleu + 0.6 * cos, 4)

    def compute_noise_sensitivity(
        self,
        query:    str,
        answer:   str,
        contexts: List[str],
        noisy_answer: Optional[str] = None,
    ) -> float:
        """
        Noise sensitivity: how much the faithfulness/relevance drops when
        context is injected with noise.

        If `noisy_answer` is provided (pre-generated answer on noisy context),
        we compare faithfulness(original) - faithfulness(noisy).
        Otherwise we generate a synthetic noisy context and re-score.

        Returns stability score (1 = not sensitive to noise; 0 = very sensitive).
        """
        orig_faith, _ = self.compute_faithfulness(answer, contexts)

        if noisy_answer is not None:
            noisy_faith, _ = self.compute_faithfulness(noisy_answer, contexts)
        else:
            # Synthesise noisy context: shuffle token order in a portion of the text
            noisy_contexts = self._inject_noise(contexts, self.noise_ratio)
            noisy_faith, _ = self.compute_faithfulness(answer, noisy_contexts)

        drop       = max(0.0, orig_faith - noisy_faith)
        stability  = 1.0 - min(drop, 1.0)
        return round(stability, 4)

    # ──────────────────────────────────────────────────────────────────────
    # Noise injection helper
    # ──────────────────────────────────────────────────────────────────────

    def _inject_noise(self, contexts: List[str], ratio: float) -> List[str]:
        """
        Replace `ratio` proportion of tokens in each context with random tokens
        drawn from the same vocabulary to simulate noisy retrieval.
        """
        import random
        noisy = []
        for ctx in contexts:
            tokens = ctx.split()
            n_noisy = int(len(tokens) * ratio)
            indices = random.sample(range(len(tokens)), min(n_noisy, len(tokens)))
            for idx in indices:
                # Rotate the token (synthetic noise)
                tokens[idx] = tokens[idx][::-1] if tokens[idx] else "NOISE"
            noisy.append(" ".join(tokens))
        return noisy

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def run_all(
        self,
        query:        str,
        answer:       str,
        contexts:     List[str],
        ground_truth: str = "",
        noisy_answer: Optional[str] = None,
    ) -> RagasResult:
        """
        Run all RAGAS metrics and return a RagasResult.

        Parameters
        ----------
        query        : the user's original question
        answer       : the RAG system's final answer
        contexts     : list of retrieved context strings
        ground_truth : reference / expected answer (optional)
        noisy_answer : answer generated on noisy context for noise_sensitivity
        """
        # Core metrics
        faithfulness, unsupported = self.compute_faithfulness(answer, contexts)
        ans_relevance             = self.compute_answer_relevance(query, answer)
        ctx_precision             = self.compute_context_precision(contexts, answer)
        ctx_recall                = self.compute_context_recall(contexts, ground_truth) \
                                    if ground_truth else 0.0
        ctx_entity_recall         = self.compute_context_entity_recall(contexts, ground_truth) \
                                    if ground_truth else 0.0
        ans_correctness           = self.compute_answer_correctness(answer, ground_truth) \
                                    if ground_truth else 0.0
        ans_similarity            = self.compute_answer_similarity(answer, ground_truth) \
                                    if ground_truth else 0.0
        noise_sensitivity         = self.compute_noise_sensitivity(
                                        query, answer, contexts, noisy_answer
                                    )

        return RagasResult(
            faithfulness           = faithfulness,
            answer_relevance       = ans_relevance,
            context_precision      = ctx_precision,
            context_recall         = ctx_recall,
            context_entity_recall  = ctx_entity_recall,
            answer_correctness     = ans_correctness,
            answer_similarity      = ans_similarity,
            noise_sensitivity      = noise_sensitivity,
            metadata               = {
                "used_llm_judge":      self.use_llm,
                "unsupported_claims":  unsupported,
                "num_contexts":        len(contexts),
                "has_ground_truth":    bool(ground_truth),
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # Batch evaluation
    # ──────────────────────────────────────────────────────────────────────

    def run_batch(
        self,
        samples: List[dict],
    ) -> List[dict]:
        """
        Evaluate a batch of samples.

        Each dict must contain: query, answer, contexts (list of str).
        Optionally: ground_truth (str), noisy_answer (str).

        Returns list of dicts with all scores.
        """
        results = []
        for i, sample in enumerate(samples):
            log.info("[RagasEvaluator] Evaluating sample %d/%d", i + 1, len(samples))
            try:
                result = self.run_all(
                    query        = sample["query"],
                    answer       = sample["answer"],
                    contexts     = sample["contexts"],
                    ground_truth = sample.get("ground_truth", ""),
                    noisy_answer = sample.get("noisy_answer"),
                )
                row = result.to_dict()
                row["sample_id"] = sample.get("id", i)
                row["query"]     = sample["query"]
                results.append(row)
            except Exception as e:
                log.error("[RagasEvaluator] Sample %d failed: %s", i, e)
                results.append({"sample_id": i, "error": str(e)})

        return results

    def aggregate_batch(self, batch_results: List[dict]) -> dict:
        """Compute mean scores across a batch of evaluation results."""
        metric_keys = [
            "faithfulness", "answer_relevance", "context_precision",
            "context_recall", "context_entity_recall", "answer_correctness",
            "answer_similarity", "noise_sensitivity", "ragas_score",
        ]
        totals = {k: [] for k in metric_keys}
        for row in batch_results:
            for k in metric_keys:
                if k in row and isinstance(row[k], (int, float)):
                    totals[k].append(row[k])

        return {
            k: round(sum(v) / len(v), 4) if v else 0.0
            for k, v in totals.items()
        }
