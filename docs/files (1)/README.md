# Medical GraphRAG — Evaluation Framework

A lightweight, modular evaluation framework for the Medical GraphRAG system.
No heavy ML evaluation libraries required — only standard Python + `numpy`.

---

## Architecture

```
evaluation/
├── evaluate.py       ← CLI entry point  (python evaluate.py --mode ragas|trulens|both)
├── ragas_eval.py     ← RAGAS-style metrics on a JSON dataset
├── trulens_eval.py   ← TruLens-style pipeline wrapper + feedback functions
├── feedbacks.py      ← Custom metric functions (graph, fusion, fallback, multimodal)
└── README.md         ← This file
```

### How the pieces connect

```
                    ┌─────────────────────────────┐
                    │         evaluate.py          │
                    │  CLI --mode ragas|trulens    │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┴──────────────────┐
              │                                   │
   ┌──────────▼──────────┐           ┌────────────▼──────────┐
   │     ragas_eval.py   │           │    trulens_eval.py    │
   │  Dataset → metrics  │           │  Pipeline wrapper +   │
   │  (JSON in/out)      │           │  feedback functions   │
   └──────────┬──────────┘           └────────────┬──────────┘
              │                                   │
              └──────────────┬────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  feedbacks.py   │
                    │  Custom metrics │
                    │  (graph, fusion,│
                    │  fallback, MM)  │
                    └─────────────────┘
```

---

## Setup

### 1. Install dependencies

The evaluation framework itself only needs the Python standard library and `numpy`.
The live TruLens mode also requires the full pipeline environment:

```bash
# Minimal (RAGAS heuristic + TruLens offline demo)
pip install numpy

# Full pipeline evaluation (live TruLens mode)
pip install numpy groq python-dotenv qdrant-client langchain neo4j
```

### 2. Set environment variables

```bash
# Required for --use-llm (LLM judge) and live TruLens pipeline calls
export GROQ_API_KEY="gsk_..."
```

### 3. Verify installation

```bash
# Offline smoke-test — no API keys or DBs needed
python evaluate.py --mode ragas   # generates a sample dataset and runs it
python evaluate.py --mode trulens --demo
```

---

## Dataset Format (RAGAS)

Create a JSON file with a list of evaluation samples:

```json
[
  {
    "id":           "q001",
    "query":        "What drugs treat osteosarcoma?",
    "answer":       "Osteosarcoma is treated with cisplatin, doxorubicin ...",
    "contexts":     [
      "Conventional treatment for OS consists of neoadjuvant chemotherapy ...",
      "The four agents include methotrexate, doxorubicin, cisplatin, ifosfamide."
    ],
    "ground_truth": "Standard chemotherapy is methotrexate, doxorubicin, cisplatin, ifosfamide."
  }
]
```

| Key           | Required | Description                                                |
|---------------|----------|------------------------------------------------------------|
| `query`       | ✅       | The user's question                                        |
| `answer`      | ✅       | The system's generated answer to evaluate                  |
| `contexts`    | ✅       | List of retrieved text chunks (from Qdrant / graph)        |
| `ground_truth`| Optional | Reference answer — enables recall and correctness metrics  |
| `id`          | Optional | Sample identifier for tracing results                      |

A sample dataset with 5 oncology questions is auto-generated the first time
you run `python evaluate.py --mode ragas` with no `--dataset` argument.

---

## How to Run

### RAGAS Evaluation

```bash
# Basic — heuristic metrics, auto-generates sample dataset
python evaluate.py --mode ragas

# With your own dataset
python evaluate.py --mode ragas --dataset data/my_dataset.json

# With LLM judge (Groq) for better faithfulness + relevance scores
python evaluate.py --mode ragas --dataset data/my_dataset.json --use-llm

# Custom output location
python evaluate.py --mode ragas \
    --dataset  data/my_dataset.json \
    --output   results/ragas_$(date +%Y%m%d).json \
    --use-llm
```

### TruLens Evaluation

```bash
# Offline demo — no pipeline connection needed
python evaluate.py --mode trulens --demo

# Live query against the real pipeline (requires full stack)
python evaluate.py --mode trulens \
    --query "What side effects does cisplatin cause?" \
    --patient-report "Patient has osteosarcoma, taking cisplatin"

# Save the TruLens record and session
python evaluate.py --mode trulens \
    --query "What is melanoma?" \
    --trulens-output results/record.json \
    --session-save  results/session.json

# Summarise a saved session
python evaluate.py --mode trulens --summary results/session.json
```

### Run Both Modes

```bash
python evaluate.py --mode both \
    --dataset data/my_dataset.json \
    --query   "What drugs treat osteosarcoma?"
```

### Module-level usage (within your application)

```python
# RAGAS
from evaluation.ragas_eval import RagasEvaluator

evaluator = RagasEvaluator(use_llm=False)   # heuristic, no API key needed
results   = evaluator.evaluate_dataset(
    dataset_path = "data/eval_dataset.json",
    output_path  = "results/ragas.json",
)
print(results["aggregate"])


# TruLens — offline scoring of a pre-built trace
from evaluation.trulens_eval import TruLensEvaluator, PipelineTrace

trace                = PipelineTrace("What treats osteosarcoma?")
trace.vector_context = "Osteosarcoma treatment involves cisplatin ..."
trace.graph_context  = "Osteosarcoma → Cisplatin"
trace.fused_context  = trace.graph_context + "\n\n" + trace.vector_context
trace.final_answer   = "Cisplatin, doxorubicin and methotrexate are used."
trace.had_fallback   = False

evaluator = TruLensEvaluator(pipeline_available=False)
record    = evaluator.evaluate_trace(trace)
print(record["composite_score"], record["grade"])


# Custom feedbacks
from evaluation.feedbacks import run_all_feedbacks

scores = run_all_feedbacks(
    query               = "What drugs treat osteosarcoma?",
    final_answer        = "Cisplatin and doxorubicin ...",
    vector_context      = "Osteosarcoma treatment ...",
    graph_context       = "Osteosarcoma → Cisplatin",
    fused_context       = "...",
    graph_results       = [{"query_type": "cancer_drugs", "records": [...]}],
    extracted_entities  = ["Osteosarcoma", "Cisplatin"],
    selected_captions   = ["Figure 1. Osteosarcoma X-ray"],
)
print(scores["composite"])
```

---

## How Evaluation Works

### RAGAS Metrics

| Metric               | What it measures                                           | Mode            |
|----------------------|------------------------------------------------------------|-----------------|
| `faithfulness`       | Fraction of answer sentences grounded in retrieved context | Heuristic / LLM |
| `answer_relevance`   | How directly the answer addresses the query                | Heuristic / LLM |
| `context_precision`  | Fraction of retrieved chunks relevant to the answer        | Heuristic       |
| `context_recall`     | Fraction of ground-truth sentences covered by contexts     | Heuristic       |
| `answer_correctness` | Token-F1 + cosine similarity vs ground truth               | Heuristic / LLM |
| `noise_sensitivity`  | Answer stability when context is perturbed                 | Heuristic       |
| `ragas_score`        | Harmonic mean of the four core metrics                     | Derived          |

**Heuristic mode** uses TF-IDF cosine similarity (pure Python, no ML framework).  
**LLM mode** uses your Groq API key to run a zero-temperature judge — more accurate but slower and costs tokens.

### TruLens Feedback Functions

| Function                | Category   | Weight | What it measures                                    |
|-------------------------|------------|--------|-----------------------------------------------------|
| `context_relevance`     | Retrieval  | 1.5    | Retrieved docs relevant to the query                |
| `context_diversity`     | Retrieval  | 1.0    | Low redundancy among retrieved chunks               |
| `groundedness`          | Generation | 2.0    | Answer sentences supported by context               |
| `answer_relevance`      | Generation | 2.0    | Answer directly addresses the query                 |
| `medical_safety`        | Safety     | 3.0    | Absence of dangerous medical advice patterns        |
| `graph_entity_coverage` | Graph      | 1.0    | Query entities present in Neo4j results             |
| `source_balance`        | Fusion     | 1.0    | Neither vector nor graph source fully ignored       |
| `image_answer_alignment`| Multimodal | 0.5    | Selected images relevant to the answer              |
| `fallback_trigger_logic`| Fallback   | 1.5    | Fallback fires only when both contexts are empty    |

The **composite score** is a weighted average of all nine functions.  
`medical_safety` has the highest weight (3.0) — any dangerous advice pattern significantly lowers the score.

### Custom Feedback Functions (feedbacks.py)

| Function               | Composite sub-scores                                                       |
|------------------------|----------------------------------------------------------------------------|
| `graph_correctness`    | entity_hit_rate, context_richness, query_alignment, result_coverage        |
| `fusion_quality`       | vector_contribution, graph_contribution, source_balance, query_completeness|
| `fallback_correctness` | trigger_accuracy, response_relevance, marker_present, length_adequacy      |
| `multimodal_relevance` | caption_query_relevance, caption_answer_relevance, selection_precision,    |
|                        | caption_diversity                                                           |

---

## Output Format

### RAGAS output (JSON)

```json
{
  "dataset":    "data/eval_dataset.json",
  "mode":       "heuristic",
  "n_samples":  5,
  "elapsed_ms": 42.3,
  "results": [
    {
      "id":                  "q001",
      "query":               "What drugs treat osteosarcoma?",
      "faithfulness":        0.8333,
      "answer_relevance":    0.7142,
      "context_precision":   0.6667,
      "context_recall":      0.7500,
      "answer_correctness":  0.6211,
      "noise_sensitivity":   0.9166,
      "ragas_score":         0.7371,
      "unsupported_claims":  [],
      "ground_truth_provided": true
    }
  ],
  "aggregate": {
    "faithfulness":        0.7821,
    "answer_relevance":    0.6934,
    "context_precision":   0.6120,
    "context_recall":      0.7011,
    "answer_correctness":  0.5890,
    "noise_sensitivity":   0.9243,
    "ragas_score":         0.6921
  }
}
```

### TruLens record (JSON)

```json
{
  "query":           "What drugs treat osteosarcoma?",
  "latency_ms":      1234.5,
  "had_fallback":    false,
  "feedback_scores": {
    "context_relevance":      0.7123,
    "context_diversity":      0.8900,
    "groundedness":           0.8333,
    "answer_relevance":       0.7142,
    "medical_safety":         1.0000,
    "graph_entity_coverage":  0.6667,
    "source_balance":         0.8500,
    "image_answer_alignment": 0.5000,
    "fallback_trigger_logic": 1.0000
  },
  "composite_score": 0.8124,
  "grade":           "B",
  "by_category": {
    "retrieval":   0.8012,
    "generation":  0.7738,
    "safety":      1.0000,
    "graph":       0.6667,
    "fusion":      0.8500,
    "multimodal":  0.5000,
    "fallback":    1.0000
  }
}
```

### Grade scale

| Score   | Grade |
|---------|-------|
| ≥ 0.85  | A     |
| ≥ 0.70  | B     |
| ≥ 0.55  | C     |
| ≥ 0.40  | D     |
| < 0.40  | F     |

---

## Deployment Notes

### Running in CI / CD

The framework is safe to run without any external services:

```bash
# Zero-dependency smoke test (heuristic only, offline demo)
python evaluate.py --mode ragas --quiet
python evaluate.py --mode trulens --demo --quiet
```

### Adding evaluation to your deployment pipeline

```python
# In your deployment health-check or post-deploy test
from evaluation.ragas_eval import RagasEvaluator

evaluator = RagasEvaluator(use_llm=False)
output    = evaluator.evaluate_dataset("data/golden_set.json", verbose=False)

ragas = output["aggregate"]["ragas_score"]
if ragas < 0.60:
    raise RuntimeError(f"RAGAS score {ragas:.3f} below acceptable threshold 0.60")
```

### Integrating TruLens into your Streamlit app

```python
# In your Streamlit session state, after each answer generation:
from evaluation.trulens_eval import TruLensEvaluator, PipelineTrace

@st.cache_resource
def get_evaluator():
    return TruLensEvaluator(pipeline_available=False, store_history=True)

evaluator = get_evaluator()

# After calling generate_answer_graphrag():
trace                = PipelineTrace(query)
trace.vector_context = vector_ctx
trace.graph_context  = graph_ctx
trace.fused_context  = fused_ctx
trace.final_answer   = answer
trace.had_fallback   = WEB_FALLBACK_MARKER in answer
record               = evaluator.evaluate_trace(trace)

# Display in UI
st.sidebar.metric("Answer Quality", f"{record['composite_score']:.2f}", record["grade"])
```

### Scaling to large datasets

For datasets > 1000 samples with `--use-llm`, Groq rate limits apply.
Add a small sleep between samples:

```python
# In ragas_eval.py, add to evaluate_sample() if running large batches:
import time
time.sleep(0.2)   # ~5 req/sec stays well within free-tier limits
```

### Interpreting low scores

| Low score on…       | Likely cause                                              |
|---------------------|-----------------------------------------------------------|
| `faithfulness`      | Answer contains information not in retrieved documents    |
| `answer_relevance`  | Answer is off-topic or too generic                        |
| `context_precision` | Retrieval is returning irrelevant documents               |
| `context_recall`    | Retrieved chunks don't cover the full answer space        |
| `medical_safety`    | Answer contains potentially dangerous phrasing            |
| `graph_entity_coverage` | Entity extractor missing entities, or Neo4j empty   |
| `source_balance`    | One context source dominating (check fusion thresholds)   |
| `fallback_trigger_logic` | Fallback firing when context was available          |

---

## File Reference

| File              | Key classes / functions                                          |
|-------------------|------------------------------------------------------------------|
| `evaluate.py`     | `main()`, `run_ragas_mode()`, `run_trulens_mode()`, `build_parser()` |
| `ragas_eval.py`   | `RagasEvaluator`, `compute_faithfulness()`, `compute_context_precision()`, `ragas_score()` |
| `trulens_eval.py` | `TruLensEvaluator`, `PipelineTrace`, `run_feedback_functions()`, `weighted_composite()` |
| `feedbacks.py`    | `graph_correctness()`, `fusion_quality()`, `fallback_correctness()`, `multimodal_relevance()`, `run_all_feedbacks()` |

---

## Quick Reference

```bash
# Most common commands
python evaluate.py --mode ragas                                  # heuristic, auto dataset
python evaluate.py --mode ragas  --dataset FILE --use-llm        # LLM judge
python evaluate.py --mode trulens --demo                         # offline demo
python evaluate.py --mode trulens --query "..." --session-save S # live + save session
python evaluate.py --mode trulens --summary results/session.json # inspect session
python evaluate.py --mode both   --dataset FILE --query "..."    # everything
```
