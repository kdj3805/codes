"""
graph_schema.py
---------------
Defines all Neo4j node labels, relationship types, constraints, and indexes
for the cancer GraphRAG knowledge graph.
"""

from __future__ import annotations

import logging
from neo4j_client import get_client

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Node labels
# ─────────────────────────────────────────────
NODE_LABELS = [
    "Cancer",
    "Drug",
    "NonCancerDrug",
    "SideEffect",
    "Food",
    "Symptom",
    "NutritionPattern",
]

# ─────────────────────────────────────────────
# Relationship types
# ─────────────────────────────────────────────
RELATIONSHIP_TYPES = [
    "TREATED_WITH",
    "CAUSES",
    "WORSENS",
    "HELPS",
    "INTERACTS_WITH",
    "LEADS_TO",
    "ASSOCIATED_WITH",
    "RECOMMENDED_FOR",
    "CONTRAINDICATED_WITH",
]

# ─────────────────────────────────────────────
# Constraint definitions
# ─────────────────────────────────────────────
CONSTRAINTS: list[tuple[str, str, str]] = [
    # (label, property, constraint_name)
    ("Cancer",         "name", "cancer_name_unique"),
    ("Drug",           "name", "drug_name_unique"),
    ("NonCancerDrug",  "name", "noncancerdrug_name_unique"),
    ("SideEffect",     "name", "sideeffect_name_unique"),
    ("Food",           "name", "food_name_unique"),
    ("Symptom",        "name", "symptom_name_unique"),
    ("NutritionPattern", "name", "nutrition_name_unique"),
]

# ─────────────────────────────────────────────
# Full-text index definitions
# ─────────────────────────────────────────────
FULLTEXT_INDEXES: list[tuple[str, list[str], list[str]]] = [
    # (index_name, labels, properties)
    ("cancerFullText",    ["Cancer"],      ["name", "description", "icd10"]),
    ("drugFullText",      ["Drug"],        ["name", "generic_name", "drug_class"]),
    ("sideEffectFT",      ["SideEffect"],  ["name", "description", "icd10_code"]),
    ("foodFullText",      ["Food"],        ["name", "category", "description"]),
    ("symptomFullText",   ["Symptom"],     ["name", "description"]),
]

# ─────────────────────────────────────────────
# Vector index (Neo4j ≥ 5.11)
# ─────────────────────────────────────────────
VECTOR_INDEXES: list[dict] = [
    {
        "name":       "cancerEmbedding",
        "label":      "Cancer",
        "property":   "embedding",
        "dimensions": 384,
        "similarity": "cosine",
    },
    {
        "name":       "sideEffectEmbedding",
        "label":      "SideEffect",
        "property":   "embedding",
        "dimensions": 384,
        "similarity": "cosine",
    },
]


# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────

def create_constraints(client=None) -> None:
    client = client or get_client()
    for label, prop, name in CONSTRAINTS:
        cypher = (
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )
        try:
            client.run_write(cypher)
            log.info("Constraint ensured: %s", name)
        except Exception as exc:
            log.warning("Constraint %s skipped: %s", name, exc)


def create_fulltext_indexes(client=None) -> None:
    client = client or get_client()
    for idx_name, labels, props in FULLTEXT_INDEXES:
        label_str = "|".join(labels)
        prop_str  = ", ".join(f"n.{p}" for p in props)
        cypher = (
            f"CREATE FULLTEXT INDEX {idx_name} IF NOT EXISTS "
            f"FOR (n:{label_str}) ON EACH [{prop_str}]"
        )
        try:
            client.run_write(cypher)
            log.info("Fulltext index ensured: %s", idx_name)
        except Exception as exc:
            log.warning("Fulltext index %s skipped: %s", idx_name, exc)


def create_vector_indexes(client=None) -> None:
    client = client or get_client()
    for vi in VECTOR_INDEXES:
        cypher = f"""
        CREATE VECTOR INDEX {vi['name']} IF NOT EXISTS
        FOR (n:{vi['label']}) ON (n.{vi['property']})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`:       {vi['dimensions']},
                `vector.similarity_function`: '{vi['similarity']}'
            }}
        }}
        """
        try:
            client.run_write(cypher)
            log.info("Vector index ensured: %s", vi["name"])
        except Exception as exc:
            log.warning("Vector index %s skipped (Neo4j < 5.11?): %s", vi["name"], exc)


def bootstrap_schema(client=None) -> None:
    """Idempotently create all constraints and indexes."""
    client = client or get_client()
    log.info("Bootstrapping graph schema …")
    create_constraints(client)
    create_fulltext_indexes(client)
    create_vector_indexes(client)
    log.info("Schema bootstrap complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bootstrap_schema()
