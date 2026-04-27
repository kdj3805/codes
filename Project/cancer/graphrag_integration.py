"""
graphrag_integration.py
-----------------------
1. Fused context builder (Qdrant vector + Neo4j graph)
2. Modified generate_answer() with GraphRAG
3. Synthetic patient reports
4. Test queries with example expected outputs
"""

from __future__ import annotations

import os
import logging
from groq import Groq

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Imports from existing RAG pipeline (already in your codebase)
# ─────────────────────────────────────────────────────────────────
# from retriever import get_hybrid_mmr_retriever, build_context
# from followups import generate_followups
# Stubs shown here for clarity — wire to your actual modules.

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMP  = 0.1


# ══════════════════════════════════════════════════════════════════
# Section 1: Context fusion
# ══════════════════════════════════════════════════════════════════

def fuse_contexts(
    vector_context: str,
    graph_context:  str,
    max_total_chars: int = 6000,
) -> str:
    """
    Merge vector-retrieved context with graph-retrieved context.
    Graph context is appended as a structured knowledge section.
    Truncates to max_total_chars total.
    """
    if not graph_context:
        return vector_context

    separator = "\n\n" + "─" * 60 + "\n"

    fused = (
        "VECTOR RETRIEVAL CONTEXT (peer-reviewed literature):\n"
        + vector_context
        + separator
        + "GRAPH KNOWLEDGE BASE (structured oncology knowledge):\n"
        + graph_context
    )

    if len(fused) > max_total_chars:
        # Keep vector context whole; trim graph context
        max_graph = max_total_chars - len(vector_context) - len(separator) - 100
        if max_graph > 200:
            graph_context = graph_context[:max_graph] + "\n...[graph context truncated]"
            fused = (
                "VECTOR RETRIEVAL CONTEXT (peer-reviewed literature):\n"
                + vector_context
                + separator
                + "GRAPH KNOWLEDGE BASE (structured oncology knowledge):\n"
                + graph_context
            )
        else:
            fused = fused[:max_total_chars]

    return fused


# ══════════════════════════════════════════════════════════════════
# Section 2: Modified generate_answer with GraphRAG
# ══════════════════════════════════════════════════════════════════

def generate_answer_graphrag(
    query:          str,
    patient_report: str        = "",
    chat_history:   list[dict] = None,
    cancer_filter:  str        = "",
    is_analysis:    bool       = False,
) -> tuple[str, list, list]:
    """
    Enhanced generate_answer that fuses:
      - Qdrant hybrid MMR retrieval (vector)
      - Neo4j GraphRAG retrieval (graph)
    Then sends fused context to Groq LLaMA 3.3 70B.
    """
    if chat_history is None:
        chat_history = []

    try:
        log.info("[GraphRAG] Query: %s", query[:70])

        # ── Step 1: Retrieval query refinement ──────────────────
        retrieval_query = _build_retrieval_query(query, chat_history)

        # ── Step 2: Vector retrieval (Qdrant) ───────────────────
        from Cancer_retrieval_v2_visual import get_hybrid_mmr_retriever, build_context
        retriever  = get_hybrid_mmr_retriever(cancer_filter=cancer_filter)
        retrieved  = retriever.invoke(retrieval_query)
        vector_ctx = build_context(retrieved) if retrieved else ""
        sources    = _extract_sources(retrieved) if retrieved else []

        # ── Step 3: Graph retrieval (Neo4j) ─────────────────────
        try:
            from graph_retrieval import retrieve_graph_context
            graph_ctx = retrieve_graph_context(query, patient_report)
        except Exception as ge:
            log.warning("[GraphRAG] Graph retrieval failed: %s", ge)
            graph_ctx = ""

        # ── Step 4: Fallback if both empty ──────────────────────
        if not vector_ctx and not graph_ctx:
            empty_msg = (
                "According to the provided clinical context, "
                "no relevant information was found for this query."
            )
            if not is_analysis:
                return _web_search_fallback(empty_msg, query, patient_report, chat_history)
            return empty_msg, [], []

        # ── Step 5: Fuse contexts ────────────────────────────────
        fused_context = fuse_contexts(vector_ctx, graph_ctx)

        # ── Step 6: Build prompt ─────────────────────────────────
        system_message = (
            "You are an empathetic medical AI assistant helping "
            "cancer patients and clinicians understand medical information.\n"
            "You have access to both peer-reviewed literature AND a structured "
            "oncology knowledge graph containing verified drug-side effect-food relationships.\n"
            "CRITICAL SYSTEM DIRECTIVE: You are connected to a frontend UI capable of rendering images. "
            "NEVER state that you are a text-based AI or cannot display images. "
            "If the context provides an image reference relevant to the answer, "
            "you MUST use the exact format `[IMAGE: filename.ext]` to display it."
        )

        graph_instruction = ""
        if graph_ctx:
            graph_instruction = (
                "\n- The GRAPH KNOWLEDGE BASE section contains structured relationships "
                "(drug→side effect, food→helps/worsens, drug interactions). "
                "Prioritise graph data for specific drug/food/interaction questions."
            )

        current_user_msg = f"""PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

CLINICAL CONTEXT:
{fused_context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer using the clinical context provided above (both vector and graph sections).
- Explain clearly — avoid heavy jargon where possible.
- Cite source numbers [1], [2] etc. when referencing literature facts.
- For food recommendations, drug interactions, and side effects, also reference
  the Graph Knowledge Base section when available.{graph_instruction}
- IMPORTANT: If the clinical context contains a "Visual Assets Database"
  section with image references relevant to the question, include them as:
    [IMAGE: filename.png]
- If the answer is not in the context, clearly state you do not have enough information.
- End with a disclaimer advising consultation with a qualified doctor.
"""

        # ── Step 7: Build message history ───────────────────────
        groq_messages = [{"role": "system", "content": system_message}]
        groq_messages.extend(_format_history_for_groq(chat_history))
        groq_messages.append({"role": "user", "content": current_user_msg})

        # ── Step 8: Call Groq ────────────────────────────────────
        client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model       = GROQ_MODEL,
            temperature = GROQ_TEMP,
            messages    = groq_messages,
        )
        answer = response.choices[0].message.content or ""

        # ── Step 9: Fallback if still insufficient ───────────────
        if _rag_has_no_answer(answer) and not is_analysis:
            log.info("[GraphRAG] RAG insufficient → web search fallback")
            return _web_search_fallback(answer, query, patient_report, chat_history)

        # ── Step 10: Follow-ups ──────────────────────────────────
        from Cancer_retrieval_v2_visual import generate_followups
        followups = generate_followups(answer, fused_context, cancer_filter)

        return answer, sources, followups

    except Exception as e:
        log.exception("[GraphRAG] Error in generate_answer_graphrag")
        return f"Error: {str(e)}", [], []


# ──────────────────────────────────────────────
# Stubs — replace with your actual implementations
# ──────────────────────────────────────────────

def _build_retrieval_query(query: str, history: list[dict]) -> str:
    if not history:
        return query
    last = history[-1].get("content", "") if history else ""
    return f"{last} {query}".strip()

def _extract_sources(retrieved: list) -> list:
    sources = []
    for doc in retrieved:
        meta = getattr(doc, "metadata", {})
        url  = meta.get("source") or meta.get("url") or ""
        if url and url not in sources:
            sources.append(url)
    return sources

def _format_history_for_groq(history: list[dict]) -> list[dict]:
    return [{"role": m.get("role","user"), "content": m.get("content","")} for m in history[-6:]]

def _rag_has_no_answer(answer: str) -> bool:
    indicators = ["do not have enough information", "no relevant information", "cannot find"]
    return any(ind in answer.lower() for ind in indicators)

def _web_search_fallback(prev: str, query: str, report: str, history: list) -> tuple:
    # Wire to your existing web_search_fallback
    return prev + "\n\n[Web search fallback not configured]", [], []


# ══════════════════════════════════════════════════════════════════
# Section 3: Synthetic patient reports
# ══════════════════════════════════════════════════════════════════

SYNTHETIC_PATIENT_REPORTS: list[dict] = [
    {
        "patient_id": "PT-001",
        "name":       "Arjun Sharma",
        "age":        14,
        "diagnosis":  "Osteosarcoma",
        "stage":      "IIB",
        "treatment":  "MAP protocol (Methotrexate, Doxorubicin, Cisplatin)",
        "report": """
Patient: Arjun Sharma, 14-year-old male.
Diagnosis: High-grade osteosarcoma of the distal femur, Stage IIB.
Treatment: MAP protocol — high-dose Methotrexate (12 g/m²), Doxorubicin (75 mg/m²), Cisplatin (100 mg/m²).
Current symptoms: Severe nausea and vomiting post-cisplatin infusion, grade 3 mucositis,
                  significant appetite loss, weight loss of 4 kg over last cycle.
Medications: Ondansetron, Dexamethasone (antiemetic regimen).
Concurrent meds: None.
Labs: Neutropenia (ANC 800), Hb 9.2 g/dL (anemia).
""",
    },
    {
        "patient_id": "PT-002",
        "name":       "Priya Mehta",
        "age":        42,
        "diagnosis":  "Breast Cancer",
        "stage":      "III",
        "treatment":  "AC-T (Doxorubicin, Cyclophosphamide, Paclitaxel)",
        "report": """
Patient: Priya Mehta, 42-year-old female.
Diagnosis: HER2-negative ER/PR-positive invasive ductal carcinoma, Stage IIIA.
Treatment: AC-T regimen — Doxorubicin (60 mg/m²) + Cyclophosphamide (600 mg/m²) ×4 cycles,
           followed by Paclitaxel (175 mg/m²) ×4 cycles.
Current symptoms: Peripheral neuropathy (tingling fingers), alopecia, fatigue, taste changes.
Concurrent medications: Warfarin (for DVT prophylaxis), Omeprazole (GERD).
Labs: WBC 3.1, platelets 145k, Hb 10.8 g/dL.
""",
    },
    {
        "patient_id": "PT-003",
        "name":       "Rajan Pillai",
        "age":        67,
        "diagnosis":  "Lung Cancer",
        "stage":      "IV",
        "treatment":  "Pembrolizumab + Carboplatin + Paclitaxel",
        "report": """
Patient: Rajan Pillai, 67-year-old male.
Diagnosis: Stage IV non-small cell lung cancer (NSCLC), adenocarcinoma, PD-L1 TPS 65%.
Treatment: Pembrolizumab (200 mg) + Carboplatin (AUC 5) + Paclitaxel (200 mg/m²), q3w.
Current symptoms: Fatigue grade 2, decreased appetite, weight loss 6 kg, constipation.
Concurrent medications: Metformin (T2DM), Aspirin 75 mg (cardiac), Omeprazole.
Labs: Creatinine 1.4 mg/dL, Hb 10.1 g/dL, blood glucose 140 mg/dL.
""",
    },
    {
        "patient_id": "PT-004",
        "name":       "Kavitha Nair",
        "age":        58,
        "diagnosis":  "Acute Leukemia",
        "stage":      "AML",
        "treatment":  "7+3 (Cytarabine + Daunorubicin)",
        "report": """
Patient: Kavitha Nair, 58-year-old female.
Diagnosis: Acute Myeloid Leukemia (AML), intermediate risk.
Treatment: Standard 7+3 induction — Cytarabine 100 mg/m² CI days 1–7,
           Daunorubicin 60 mg/m² days 1–3.
Current symptoms: Severe mucositis grade 4, diarrhea, neutropenic fever,
                  complete appetite loss, severe oral pain.
Concurrent medications: Fluconazole (antifungal prophylaxis), Ciprofloxacin (antibacterial).
Labs: ANC 0 (aplastic), Hb 7.8 g/dL, Platelets 12k.
""",
    },
    {
        "patient_id": "PT-005",
        "name":       "Dev Krishnan",
        "age":        35,
        "diagnosis":  "Melanoma",
        "stage":      "IV",
        "treatment":  "Ipilimumab + Nivolumab (IO combination)",
        "report": """
Patient: Dev Krishnan, 35-year-old male.
Diagnosis: Stage IV metastatic melanoma, BRAF wild-type.
Treatment: Ipilimumab (3 mg/kg) + Nivolumab (1 mg/kg) q3w ×4 cycles, then Nivolumab maintenance.
Current symptoms: Immune-related diarrhea grade 2, fatigue, decreased appetite.
Concurrent medications: Dexamethasone (irAE management), Ibuprofen (joint pain).
Labs: LFTs mildly elevated (ALT 68 U/L).
""",
    },
]


# ══════════════════════════════════════════════════════════════════
# Section 4: Test queries with expected graph paths
# ══════════════════════════════════════════════════════════════════

TEST_QUERIES: list[dict] = [
    # ── Chemo side effects ─────────────────────────────────
    {
        "category":       "chemo_side_effects",
        "patient_id":     "PT-001",
        "query":          "What side effects should I expect from cisplatin in my osteosarcoma treatment?",
        "expected_graph_path": [
            "Osteosarcoma → TREATED_WITH → Cisplatin",
            "Cisplatin → CAUSES → Nausea [severe/very_common]",
            "Cisplatin → CAUSES → Nephrotoxicity [severe/common]",
            "Cisplatin → CAUSES → Ototoxicity [severe/common]",
        ],
        "expected_entities": {"cancers": ["Osteosarcoma"], "drugs": ["Cisplatin"]},
    },
    {
        "category":       "chemo_side_effects",
        "patient_id":     "PT-002",
        "query":          "My paclitaxel is causing tingling in my hands and feet. Is this normal?",
        "expected_graph_path": [
            "Paclitaxel → CAUSES → Peripheral Neuropathy [moderate/very_common]",
            "Peripheral Neuropathy → LEADS_TO → Tingling Hands/Feet",
        ],
        "expected_entities": {"drugs": ["Paclitaxel"], "side_effects": ["Peripheral Neuropathy"]},
    },
    # ── Food recommendations ───────────────────────────────
    {
        "category":       "food_recommendation",
        "patient_id":     "PT-001",
        "query":          "What foods can help me manage nausea and mouth sores from chemotherapy?",
        "expected_graph_path": [
            "Banana → HELPS → Nausea [bland BRAT food]",
            "Ginger Tea → HELPS → Nausea [gingerol effect]",
            "Ice Chips → HELPS → Mucositis [cold numbing]",
            "Custard → HELPS → Mucositis [soft high-calorie]",
        ],
        "expected_entities": {"side_effects": ["Nausea", "Mucositis"], "intents": ["food_recommendation"]},
    },
    {
        "category":       "food_recommendation",
        "patient_id":     "PT-003",
        "query":          "I have constipation during lung cancer treatment. What should I eat?",
        "expected_graph_path": [
            "Prune Juice → HELPS → Constipation [sorbitol + fiber]",
            "Oatmeal → HELPS → Constipation [soluble fiber]",
            "Whole Grain Bread → HELPS → Constipation [insoluble fiber]",
        ],
        "expected_entities": {"cancers": ["Lung Cancer"], "side_effects": ["Constipation"]},
    },
    {
        "category":       "food_recommendation",
        "patient_id":     "PT-004",
        "query":          "What foods should I absolutely avoid when I have severe mouth sores?",
        "expected_graph_path": [
            "Spicy Foods → WORSENS → Mucositis [capsaicin irritates inflamed mucosa]",
            "Citrus Juice → WORSENS → Mucositis [acid erodes damaged oral lining]",
            "Alcohol → WORSENS → Mucositis [ethanol desiccates and inflames]",
        ],
        "expected_entities": {"side_effects": ["Mucositis"], "intents": ["food_recommendation"]},
    },
    # ── Drug interactions ──────────────────────────────────
    {
        "category":       "drug_interaction",
        "patient_id":     "PT-002",
        "query":          "I'm on warfarin for blood clots and starting doxorubicin. Is there a dangerous interaction?",
        "expected_graph_path": [
            "Doxorubicin → INTERACTS_WITH → Warfarin [severe: Enhanced anticoagulation; bleeding risk]",
        ],
        "expected_entities": {"drugs": ["Doxorubicin"], "ncdrugs": ["Warfarin"], "intents": ["drug_interaction"]},
    },
    {
        "category":       "drug_interaction",
        "patient_id":     "PT-001",
        "query":          "Can I take ibuprofen for pain while on methotrexate?",
        "expected_graph_path": [
            "Methotrexate → INTERACTS_WITH → Ibuprofen [severe: NSAID inhibits renal MTX excretion]",
        ],
        "expected_entities": {"drugs": ["Methotrexate"], "ncdrugs": ["Ibuprofen"], "intents": ["drug_interaction"]},
    },
    {
        "category":       "drug_interaction",
        "patient_id":     "PT-005",
        "query":          "Will taking ibuprofen for joint pain affect my immunotherapy with ipilimumab?",
        "expected_graph_path": [
            "Ipilimumab → INTERACTS_WITH → Dexamethasone [moderate: steroids reduce checkpoint inhibitor efficacy]",
            "Ibuprofen NSAID class consideration → inflammation modulation",
        ],
        "expected_entities": {"drugs": ["Ipilimumab"], "ncdrugs": ["Ibuprofen"]},
    },
    # ── Multi-hop reasoning ────────────────────────────────
    {
        "category":       "multihop",
        "patient_id":     "PT-001",
        "query":          "What are the best foods for a child with osteosarcoma on MAP protocol?",
        "expected_graph_path": [
            "Osteosarcoma → TREATED_WITH → Cisplatin/Doxorubicin/Methotrexate",
            "Cisplatin → CAUSES → Nausea → Banana/Ginger Tea HELPS",
            "Doxorubicin → CAUSES → Mucositis → Ice Chips/Custard HELPS",
            "Methotrexate → CAUSES → Mucositis → Scrambled Eggs HELPS",
        ],
        "expected_entities": {"cancers": ["Osteosarcoma"], "intents": ["food_recommendation"]},
    },
    {
        "category":       "multihop",
        "patient_id":     "PT-004",
        "query":          "My leukemia treatment causes diarrhea. What foods help and what should I avoid?",
        "expected_graph_path": [
            "Acute Leukemia → TREATED_WITH → multiple drugs → CAUSES → Diarrhea",
            "White Rice → HELPS → Diarrhea",
            "Sports Drink → HELPS → Diarrhea [electrolytes]",
            "Spicy Foods → WORSENS → Diarrhea",
            "Dairy (full-fat) → WORSENS → Diarrhea [lactase deficiency]",
        ],
        "expected_entities": {"cancers": ["Acute Leukemia"], "side_effects": ["Diarrhea"]},
    },
]


# ══════════════════════════════════════════════════════════════════
# Section 5: Test runner (offline simulation)
# ══════════════════════════════════════════════════════════════════

def run_graph_retrieval_test(query_dict: dict) -> dict:
    """
    Runs graph retrieval for a test query and returns the context.
    Does NOT call Groq — tests only the graph retrieval layer.
    """
    from entity_extractor import extract_entities
    from graph_retrieval import query_graph, build_graph_context

    query  = query_dict["query"]
    report = ""
    for p in SYNTHETIC_PATIENT_REPORTS:
        if p["patient_id"] == query_dict.get("patient_id"):
            report = p["report"]
            break

    entities = extract_entities(query, report)
    results  = query_graph(entities)
    context  = build_graph_context(results)

    return {
        "query":            query,
        "patient_id":       query_dict.get("patient_id"),
        "category":         query_dict["category"],
        "extracted":        entities.to_dict(),
        "graph_results":    len(results),
        "graph_context_len": len(context),
        "graph_context":    context,
    }


def run_all_tests(verbose: bool = True) -> None:
    """Execute all test queries against the graph."""
    print("=" * 70)
    print("GRAPHRAG TEST SUITE")
    print("=" * 70)

    for i, tq in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Category: {tq['category']}")
        print(f"  Query: {tq['query'][:80]}...")
        try:
            result = run_graph_retrieval_test(tq)
            print(f"  Entities: {result['extracted']}")
            print(f"  Graph result sets: {result['graph_results']}")
            if verbose and result["graph_context"]:
                # Print first 400 chars of context
                snippet = result["graph_context"][:400].replace("\n", "\n    ")
                print(f"  Context snippet:\n    {snippet}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("Tests complete.")


# ══════════════════════════════════════════════════════════════════
# Section 6: Expected output examples (documentation)
# ══════════════════════════════════════════════════════════════════

EXPECTED_OUTPUTS = {
    "PT-001_nausea_food": {
        "query":   "What foods help with nausea from cisplatin?",
        "graph_context_example": """
=== KNOWLEDGE GRAPH CONTEXT ===

[Recommended Foods]
  • Banana (fruit) helps with Nausea — bland easy-to-digest BRAT food
  • White Rice (grain) helps with Nausea — low-fiber easily digestible
  • Toast (grain) helps with Nausea — bland starchy food
  • Crackers (grain) helps with Nausea — bland carbohydrate buffer
  • Ginger Tea (beverage) helps with Nausea — gingerol anti-nausea effect
  • Applesauce (fruit) helps with Nausea — soft, low-fiber, easy to digest

[Foods to Avoid]
  • AVOID Fried Foods (general) — worsens Nausea: high fat delays gastric emptying
  • AVOID Alcohol (beverage) — worsens Nausea: direct gastric mucosal irritant
""",
        "expected_llm_answer_keywords": [
            "banana", "ginger", "bland", "crackers",
            "small meals", "avoid fried", "avoid alcohol",
        ],
    },
    "PT-002_warfarin_interaction": {
        "query":   "Is doxorubicin safe with warfarin?",
        "graph_context_example": """
=== KNOWLEDGE GRAPH CONTEXT ===

[Serious Drug Interactions]
  • Doxorubicin + Warfarin [severe]: Enhanced anticoagulation; bleeding risk
    (confidence: 0.94)
""",
        "expected_llm_answer_keywords": [
            "severe interaction", "bleeding risk", "anticoagulation",
            "monitor INR", "consult doctor",
        ],
    },
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Uncomment to run tests against live Neo4j:
    # run_all_tests(verbose=True)

    # Print synthetic reports for review
    print("\n=== SYNTHETIC PATIENT REPORTS ===")
    for p in SYNTHETIC_PATIENT_REPORTS:
        print(f"\n{p['patient_id']} | {p['name']} | {p['diagnosis']} | Stage {p['stage']}")
        print(f"  Treatment: {p['treatment']}")

    print("\n=== TEST QUERIES ===")
    for i, tq in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}] [{tq['category']}] {tq['query']}")
        print(f"  Expected graph path: {tq['expected_graph_path'][0]}")
