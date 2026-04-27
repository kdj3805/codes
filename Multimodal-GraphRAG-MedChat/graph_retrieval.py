"""
graph_retrieval.py
------------------
Cypher-based retrieval layer for the cancer GraphRAG system.
Implements multi-hop reasoning over Neo4j knowledge graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from neo4j_client import get_client
from entity_extractor import ExtractedEntities

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════

@dataclass
class GraphResult:
    query_type:   str
    entities_in:  list[str]
    records:      list[dict]
    cypher_used:  str


# ══════════════════════════════════════════════════════════════════
# Individual Cypher retrieval functions
# ══════════════════════════════════════════════════════════════════

def get_drugs_for_cancer(cancer_names: list[str], client=None) -> GraphResult:
    """Cancer → TREATED_WITH → Drug"""
    client = client or get_client()
    cypher = """
    MATCH (c:Cancer)-[r:TREATED_WITH]->(d:Drug)
    WHERE c.name IN $names
    RETURN c.name AS cancer,
           d.name AS drug,
           d.drug_class AS drug_class,
           d.mechanism AS mechanism,
           r.line AS treatment_line,
           r.protocol AS protocol,
           r.confidence_score AS confidence
    ORDER BY r.confidence_score DESC
    LIMIT 20
    """
    records = client.run_query(cypher, {"names": cancer_names})
    return GraphResult("cancer_drugs", cancer_names, records, cypher)


def get_side_effects_for_drugs(drug_names: list[str], client=None) -> GraphResult:
    """Drug → CAUSES → SideEffect"""
    client = client or get_client()
    cypher = """
    MATCH (d:Drug)-[r:CAUSES]->(s:SideEffect)
    WHERE d.name IN $names
    RETURN d.name AS drug,
           s.name AS side_effect,
           s.description AS description,
           s.icd10_code AS icd10,
           r.severity AS severity,
           r.frequency AS frequency,
           r.confidence_score AS confidence
    ORDER BY r.confidence_score DESC
    LIMIT 30
    """
    records = client.run_query(cypher, {"names": drug_names})
    return GraphResult("drug_side_effects", drug_names, records, cypher)


def get_foods_that_help(side_effect_names: list[str], client=None) -> GraphResult:
    """Food → HELPS → SideEffect"""
    client = client or get_client()
    cypher = """
    MATCH (f:Food)-[r:HELPS]->(s:SideEffect)
    WHERE s.name IN $names
    RETURN f.name AS food,
           f.category AS category,
           f.description AS food_description,
           s.name AS helps_with,
           r.severity AS severity,
           r.mechanism AS mechanism,
           r.confidence_score AS confidence
    ORDER BY r.confidence_score DESC
    LIMIT 25
    """
    records = client.run_query(cypher, {"names": side_effect_names})
    return GraphResult("food_recommendations", side_effect_names, records, cypher)


def get_foods_to_avoid(side_effect_names: list[str], client=None) -> GraphResult:
    """Food → WORSENS → SideEffect"""
    client = client or get_client()
    cypher = """
    MATCH (f:Food)-[r:WORSENS]->(s:SideEffect)
    WHERE s.name IN $names
    RETURN f.name AS food,
           f.category AS category,
           s.name AS worsens,
           r.severity AS severity,
           r.mechanism AS mechanism,
           r.confidence_score AS confidence
    ORDER BY r.confidence_score DESC
    LIMIT 20
    """
    records = client.run_query(cypher, {"names": side_effect_names})
    return GraphResult("foods_to_avoid", side_effect_names, records, cypher)


def get_drug_interactions(drug_names: list[str], nc_drug_names: list[str], client=None) -> GraphResult:
    """Drug → INTERACTS_WITH → NonCancerDrug"""
    client = client or get_client()

    if nc_drug_names:
        cypher = """
        MATCH (d:Drug)-[r:INTERACTS_WITH]->(n:NonCancerDrug)
        WHERE d.name IN $drug_names AND n.name IN $nc_names
        RETURN d.name AS chemo_drug,
               n.name AS other_drug,
               n.indication AS indication,
               r.severity AS severity,
               r.effect AS interaction_effect,
               r.confidence_score AS confidence
        ORDER BY r.confidence_score DESC
        LIMIT 20
        """
        params = {"drug_names": drug_names, "nc_names": nc_drug_names}
    else:
        cypher = """
        MATCH (d:Drug)-[r:INTERACTS_WITH]->(n:NonCancerDrug)
        WHERE d.name IN $drug_names
        RETURN d.name AS chemo_drug,
               n.name AS other_drug,
               n.indication AS indication,
               r.severity AS severity,
               r.effect AS interaction_effect,
               r.confidence_score AS confidence
        ORDER BY r.confidence_score DESC
        LIMIT 20
        """
        params = {"drug_names": drug_names}

    records = client.run_query(cypher, params)
    entities = drug_names + nc_drug_names
    return GraphResult("drug_interactions", entities, records, cypher)


def get_multihop_cancer_food(cancer_names: list[str], client=None) -> GraphResult:
    """
    Multi-hop: Cancer → TREATED_WITH → Drug → CAUSES → SideEffect ← HELPS ← Food
    Answers: "What foods help patients with <cancer> on chemo?"
    """
    client = client or get_client()
    cypher = """
    MATCH (c:Cancer)-[:TREATED_WITH]->(d:Drug)-[rc:CAUSES]->(s:SideEffect)<-[rh:HELPS]-(f:Food)
    WHERE c.name IN $names
    RETURN c.name AS cancer,
           d.name AS drug,
           s.name AS side_effect,
           f.name AS food,
           f.category AS food_category,
           rh.mechanism AS mechanism,
           rc.severity AS se_severity,
           rh.confidence_score AS food_confidence
    ORDER BY rh.confidence_score DESC
    LIMIT 30
    """
    records = client.run_query(cypher, {"names": cancer_names})
    return GraphResult("multihop_cancer_food", cancer_names, records, cypher)


def get_multihop_drug_symptoms(drug_names: list[str], client=None) -> GraphResult:
    """
    Multi-hop: Drug → CAUSES → SideEffect → LEADS_TO → Symptom
    Answers: "What symptoms does <drug> eventually cause?"
    """
    client = client or get_client()
    cypher = """
    MATCH (d:Drug)-[:CAUSES]->(s:SideEffect)-[r:LEADS_TO]->(sy:Symptom)
    WHERE d.name IN $names
    RETURN d.name AS drug,
           s.name AS side_effect,
           sy.name AS symptom,
           sy.description AS symptom_description,
           r.confidence_score AS confidence
    ORDER BY r.confidence_score DESC
    LIMIT 25
    """
    records = client.run_query(cypher, {"names": drug_names})
    return GraphResult("multihop_drug_symptoms", drug_names, records, cypher)


def get_cancer_associated_side_effects(cancer_names: list[str], client=None) -> GraphResult:
    """Cancer → ASSOCIATED_WITH → SideEffect (intrinsic, non-chemo)"""
    client = client or get_client()
    cypher = """
    MATCH (c:Cancer)-[r:ASSOCIATED_WITH]->(s:SideEffect)
    WHERE c.name IN $names
    RETURN c.name AS cancer,
           s.name AS side_effect,
           s.description AS description,
           r.severity AS severity,
           r.frequency AS frequency,
           r.confidence_score AS confidence
    ORDER BY r.confidence_score DESC
    LIMIT 15
    """
    records = client.run_query(cypher, {"names": cancer_names})
    return GraphResult("cancer_associated_se", cancer_names, records, cypher)


def get_food_mechanism(food_names: list[str], client=None) -> GraphResult:
    """Food → all relationships with details"""
    client = client or get_client()
    cypher = """
    MATCH (f:Food)-[r]->(s:SideEffect)
    WHERE f.name IN $names
    RETURN f.name AS food,
           f.description AS food_description,
           type(r) AS relationship,
           s.name AS side_effect,
           r.mechanism AS mechanism,
           r.severity AS severity,
           r.confidence_score AS confidence
    ORDER BY f.name, type(r)
    LIMIT 20
    """
    records = client.run_query(cypher, {"names": food_names})
    return GraphResult("food_mechanism", food_names, records, cypher)


def get_severe_drug_interactions_all(drug_names: list[str], client=None) -> GraphResult:
    """All severe interactions for given chemo drugs."""
    client = client or get_client()
    cypher = """
    MATCH (d:Drug)-[r:INTERACTS_WITH]->(n:NonCancerDrug)
    WHERE d.name IN $names AND r.severity IN ['severe', 'moderate']
    RETURN d.name AS chemo_drug,
           n.name AS other_drug,
           n.drug_class AS other_class,
           n.indication AS indication,
           r.severity AS severity,
           r.effect AS effect,
           r.confidence_score AS confidence
    ORDER BY r.severity DESC, r.confidence_score DESC
    LIMIT 25
    """
    records = client.run_query(cypher, {"names": drug_names})
    return GraphResult("severe_interactions", drug_names, records, cypher)


# ══════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════

def query_graph(entities: ExtractedEntities, client=None) -> list[GraphResult]:
    """
    Run all relevant graph queries based on extracted entities.
    Returns a list of GraphResult objects.
    """
    client = client or get_client()
    results: list[GraphResult] = []

    # Drugs for cancers
    if entities.cancers:
        results.append(get_drugs_for_cancer(entities.cancers, client))
        results.append(get_cancer_associated_side_effects(entities.cancers, client))

    # Side effects for drugs
    all_drugs = entities.drugs
    if all_drugs:
        results.append(get_side_effects_for_drugs(all_drugs, client))
        results.append(get_multihop_drug_symptoms(all_drugs, client))

    # Food recommendations
    target_se = entities.side_effects
    if not target_se and entities.cancers:
        # derive side effects from cancer → drug path
        raw = get_multihop_cancer_food(entities.cancers, client)
        results.append(raw)
    elif target_se:
        results.append(get_foods_that_help(target_se, client))
        results.append(get_foods_to_avoid(target_se, client))

    # Direct food queries
    if entities.foods:
        results.append(get_food_mechanism(entities.foods, client))

    # Drug interactions
    if entities.drugs and (entities.ncdrugs or "drug_interaction" in entities.intents):
        results.append(get_drug_interactions(entities.drugs, entities.ncdrugs, client))
    elif entities.drugs and "drug_interaction" in entities.intents:
        results.append(get_severe_drug_interactions_all(entities.drugs, client))

    # Remove empty results
    results = [r for r in results if r.records]
    return results


# ══════════════════════════════════════════════════════════════════
# Context builder
# ══════════════════════════════════════════════════════════════════

def build_graph_context(results: list[GraphResult], max_chars: int = 3000) -> str:
    """
    Convert GraphResult list into a readable string for the LLM prompt.
    """
    if not results:
        return ""

    sections: list[str] = ["=== KNOWLEDGE GRAPH CONTEXT ===\n"]

    for res in results:
        if not res.records:
            continue

        label = _query_label(res.query_type)
        sections.append(f"\n[{label}]")

        for rec in res.records:
            line = _format_record(res.query_type, rec)
            sections.append(f"  • {line}")

    context = "\n".join(sections)

    # Truncate to max_chars
    if len(context) > max_chars:
        context = context[:max_chars] + "\n  ...[truncated]"

    return context


def _query_label(qt: str) -> str:
    labels = {
        "cancer_drugs":         "Treatment Protocols",
        "drug_side_effects":    "Drug Side Effects",
        "food_recommendations": "Recommended Foods",
        "foods_to_avoid":       "Foods to Avoid",
        "drug_interactions":    "Drug Interactions",
        "multihop_cancer_food": "Dietary Recommendations for Cancer Treatment",
        "multihop_drug_symptoms": "Drug → Symptom Pathway",
        "cancer_associated_se": "Cancer-Associated Side Effects",
        "food_mechanism":       "Food-Effect Relationships",
        "severe_interactions":  "Serious Drug Interactions",
    }
    return labels.get(qt, qt.replace("_", " ").title())


def _format_record(query_type: str, rec: dict) -> str:
    try:
        if query_type == "cancer_drugs":
            return (
                f"{rec.get('cancer')} → {rec.get('drug')} "
                f"({rec.get('drug_class')}); protocol: {rec.get('protocol')}; "
                f"line: {rec.get('treatment_line')}; confidence: {rec.get('confidence', 0):.2f}"
            )
        elif query_type == "drug_side_effects":
            return (
                f"{rec.get('drug')} causes {rec.get('side_effect')} "
                f"[{rec.get('severity')}/{rec.get('frequency')}]; {rec.get('description', '')}"
            )
        elif query_type == "food_recommendations":
            return (
                f"{rec.get('food')} ({rec.get('category')}) helps with "
                f"{rec.get('helps_with')} — {rec.get('mechanism', '')}"
            )
        elif query_type == "foods_to_avoid":
            return (
                f"AVOID {rec.get('food')} ({rec.get('category')}) — worsens "
                f"{rec.get('worsens')}: {rec.get('mechanism', '')}"
            )
        elif query_type in ("drug_interactions", "severe_interactions"):
            return (
                f"{rec.get('chemo_drug')} + {rec.get('other_drug')} "
                f"[{rec.get('severity')}]: {rec.get('interaction_effect') or rec.get('effect', '')}"
            )
        elif query_type == "multihop_cancer_food":
            return (
                f"{rec.get('cancer')} → {rec.get('drug')} → {rec.get('side_effect')} "
                f"→ eat {rec.get('food')} ({rec.get('mechanism', '')})"
            )
        elif query_type == "multihop_drug_symptoms":
            return (
                f"{rec.get('drug')} → {rec.get('side_effect')} → {rec.get('symptom')}: "
                f"{rec.get('symptom_description', '')}"
            )
        elif query_type == "cancer_associated_se":
            return (
                f"{rec.get('cancer')} associated with {rec.get('side_effect')} "
                f"[{rec.get('severity')}/{rec.get('frequency')}]"
            )
        elif query_type == "food_mechanism":
            return (
                f"{rec.get('food')} {rec.get('relationship')} {rec.get('side_effect')} "
                f"— {rec.get('mechanism', '')}"
            )
        else:
            return str(rec)
    except Exception:
        return str(rec)


# ══════════════════════════════════════════════════════════════════
# Convenience entry point
# ══════════════════════════════════════════════════════════════════

def retrieve_graph_context(
    query: str,
    patient_report: str = "",
    client=None,
) -> str:
    """
    Full pipeline: query → extract → graph query → context string.
    Drop-in function for integration with generate_answer().
    """
    from entity_extractor import extract_entities

    entities = extract_entities(query, patient_report)

    if entities.is_empty():
        log.debug("No entities extracted from query; skipping graph retrieval.")
        return ""

    log.info(
        "Graph retrieval | cancers=%s drugs=%s se=%s foods=%s",
        entities.cancers, entities.drugs, entities.side_effects, entities.foods,
    )

    results = query_graph(entities, client)
    context = build_graph_context(results)

    log.info(
        "Graph context built | %d result sets, %d chars",
        len(results), len(context),
    )
    return context
