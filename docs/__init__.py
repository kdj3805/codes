"""
evaluation/
-----------
Medical GraphRAG Evaluation Framework

Public API:
    from evaluation import EvaluationRunner
    runner = EvaluationRunner()
    report = runner.run(query, retrieved_docs, graph_output, final_answer, ...)
"""

__version__ = "1.0.0"
__author__  = "GraphRAG Eval Framework"

from evaluation.runner import EvaluationRunner
from evaluation.evaluators import (
    RetrievalEvaluator,
    GenerationEvaluator,
    GraphEvaluator,
    FusionEvaluator,
    FallbackEvaluator,
    MultimodalEvaluator,
)

__all__ = [
    "EvaluationRunner",
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "GraphEvaluator",
    "FusionEvaluator",
    "FallbackEvaluator",
    "MultimodalEvaluator",
]
