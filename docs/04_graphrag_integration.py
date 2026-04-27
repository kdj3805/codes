"""
04_graphrag_integration.py
===========================
Integrates Neo4j (AuraDB) GraphRAG with existing Qdrant/FAISS vector DB pipeline.
Architecture:
  User Query
       │
       ▼
  Query Router ─── classifies intent (factual vs relational vs hybrid)
       │
   ┌───┴─────────────────────────┐
   ▼                             ▼
Vector Search               Graph Traversal
(Qdrant / FAISS)            (Neo4j / AuraDB)
   │                             │
   └──────────┬──────────────────┘
              ▼
       Context Fusion (RRF / weighted merge)
              │
              ▼
       LLM (Claude via Anthropic API)
              │
              ▼
         Final Answer
"""

import os
import json
import logging
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
from neo4j import GraphDatabase
from openai import OpenAI  # or your embedding provider

# ─── Optional imports (install as needed) ────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import anthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Neo4j / AuraDB
    neo4j_uri: str      = os.getenv("NEO4J_URI",      "neo4j+s://<your-auradb-id>.databases.neo4j.io")
    neo4j_user: str     = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    # Qdrant
    qdrant_url: str     = os.getenv("QDRANT_URL",     "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = "cancer_chemo_docs"      # existing collection name

    # FAISS
    faiss_index_path: str = os.getenv("FAISS_INDEX", "faiss_index.bin")
    faiss_meta_path:  str = os.getenv("FAISS_META",  "faiss_meta.json")

    # Embedding model (OpenAI or local)
    embedding_model: str = "text-embedding-3-small"   # 1536-dim
    embedding_dim:   int = 1536

    # Anthropic
    anthropic_model: str = "claude-sonnet-4-20250514"
    max_tokens:      int = 2048

    # Retrieval
    top_k_vector:  int = 5   # results from vector DB
    top_k_graph:   int = 10  # hops / results from graph
    rrf_k:         int = 60  # Reciprocal Rank Fusion constant


# ════════════════════════════════════════════════════════════
# 2. EMBEDDING UTILITY
# ════════════════════════════════════════════════════════════

class EmbeddingClient:
    def __init__(self, cfg: Config):
        self.model = cfg.embedding_model
        self.dim   = cfg.embedding_dim
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for the given text."""
        resp = self._client.embeddings.create(model=self.model, input=text)
        vec  = resp.data[0].embedding
        arr  = np.array(vec, dtype=np.float32)
        arr /= (np.linalg.norm(arr) + 1e-9)   # L2 normalise
        return arr.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        vecs = [r.embedding for r in resp.data]
        return vecs


# ════════════════════════════════════════════════════════════
# 3. NEO4J GRAPH RETRIEVER
# ════════════════════════════════════════════════════════════

class GraphRetriever:
    """
    Executes parameterised Cypher queries against Neo4j/AuraDB.
    Supports:
      - vector similarity search via Neo4j vector index
      - full-text keyword search
      - structured traversal queries for specific intents
    """

    QUERIES = {
        # Find all chemo drugs for a cancer + their eating effects
        "cancer_drugs_effects": """
            MATCH (c:Cancer {name: $cancer_name})-[:TREATED_WITH]->(d:ChemoDrug)
            OPTIONAL MATCH (d)-[r:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
            OPTIONAL MATCH (e)-[:WORSENED_BY]->(bad_food:FoodItem)
            OPTIONAL MATCH (e)-[:RELIEVED_BY]->(good_food:FoodItem)
            RETURN c.name AS cancer,
                   d.name AS drug, d.drug_class AS drug_class, d.mechanism AS mechanism,
                   e.name AS eating_effect, r.severity AS severity, r.frequency AS frequency,
                   collect(DISTINCT bad_food.name) AS foods_to_avoid,
                   collect(DISTINCT good_food.name) AS foods_to_eat
            ORDER BY d.name, e.name
        """,

        # Find drug interactions for a given chemo drug
        "chemo_drug_interactions": """
            MATCH (n:NonChemoDrug)-[:HAS_INTERACTION_WITH]->(c:ChemoDrug {name: $chemo_name})
            OPTIONAL MATCH (n)-[:DESCRIBED_BY]->(i:DrugInteraction)
            WHERE i.id CONTAINS toLower(replace($chemo_name, ' ', '_'))
               OR i.id CONTAINS toUpper(substring($chemo_name, 0, 3))
            OPTIONAL MATCH (n)-[:TREATS]->(a:Ailment)
            RETURN n.name AS non_chemo_drug, n.drug_class AS drug_class,
                   a.name AS treats_ailment,
                   i.severity AS severity, i.description AS interaction_description,
                   i.clinical_action AS recommended_action,
                   i.eating_relevance AS eating_relevance
            ORDER BY i.severity DESC
        """,

        # Foods to avoid and eat during a specific chemo drug
        "food_guidance_for_drug": """
            MATCH (d:ChemoDrug {name: $drug_name})-[:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
            OPTIONAL MATCH (e)-[:WORSENED_BY]->(avoid:FoodItem)
            OPTIONAL MATCH (e)-[:RELIEVED_BY]->(eat:FoodItem)
            OPTIONAL MATCH (g:NutritionGuideline)-[:REQUIRED_FOR|MANAGES]->(d)
            RETURN e.name AS eating_effect, e.management_tip AS management_tip,
                   collect(DISTINCT avoid.name) AS avoid_foods,
                   collect(DISTINCT eat.name)   AS recommended_foods,
                   collect(DISTINCT g.text)     AS nutrition_guidelines
        """,

        # Full treatment protocol: protocol → drugs → effects → guidance
        "protocol_full_detail": """
            MATCH (p:TreatmentProtocol {name: $protocol_name})-[:INCLUDES_DRUG]->(d:ChemoDrug)
            OPTIONAL MATCH (d)-[:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
            OPTIONAL MATCH (g:NutritionGuideline)-[:REQUIRED_FOR]->(d)
            RETURN p.name AS protocol, p.description AS protocol_description,
                   d.name AS drug, d.drug_class AS drug_class, d.notes AS drug_notes,
                   collect(DISTINCT e.name) AS eating_effects,
                   collect(DISTINCT g.text) AS mandatory_guidelines
        """,

        # Vector similarity search (requires embedding property set on nodes)
        "vector_search_chemo": """
            CALL db.index.vector.queryNodes('chemo_vector_index', $top_k, $query_vector)
            YIELD node AS drug, score
            OPTIONAL MATCH (drug)-[:CAUSES_EATING_EFFECT]->(effect:EatingAdverseEffect)
            RETURN drug.name AS drug_name, drug.mechanism AS mechanism,
                   drug.drug_class AS drug_class, score,
                   collect(DISTINCT effect.name) AS eating_effects
            ORDER BY score DESC
        """,

        # Full-text search across all text-indexed nodes
        "fulltext_search": """
            CALL db.index.fulltext.queryNodes('chemo_text_index', $query_text)
            YIELD node, score
            RETURN labels(node)[0] AS node_type,
                   node.name AS name,
                   CASE labels(node)[0]
                     WHEN 'ChemoDrug'          THEN node.mechanism
                     WHEN 'EatingAdverseEffect' THEN node.description
                     WHEN 'NutritionGuideline'  THEN node.text
                     ELSE ''
                   END AS content,
                   score
            ORDER BY score DESC LIMIT $limit
        """,

        # Interaction-aware dietary guidance (cross-DB query)
        "interaction_eating_impact": """
            MATCH (i:DrugInteraction)-[:COMPOUNDS_EATING_EFFECT]->(e:EatingAdverseEffect)
            MATCH (n:NonChemoDrug)-[:DESCRIBED_BY]->(i)
            MATCH (c:ChemoDrug)-[:DESCRIBED_BY]->(i)
            OPTIONAL MATCH (e)-[:RELIEVED_BY]->(good:FoodItem)
            RETURN i.id AS interaction_id,
                   n.name AS non_chemo_drug,
                   c.name AS chemo_drug,
                   e.name AS compounded_effect,
                   i.severity AS severity,
                   i.eating_relevance AS eating_note,
                   collect(DISTINCT good.name) AS mitigation_foods
            ORDER BY i.severity DESC
        """,

        # Biomarker-guided treatment + diet path
        "biomarker_treatment_diet": """
            MATCH (c:Cancer)-[:HAS_BIOMARKER]->(b:Biomarker {name: $biomarker_name})
            MATCH (c)-[:TREATED_WITH]->(d:ChemoDrug)
            OPTIONAL MATCH (d)-[:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
            RETURN c.name AS cancer, b.name AS biomarker,
                   b.therapeutic_relevance AS relevant_drugs,
                   d.name AS prescribed_drug,
                   collect(DISTINCT e.name) AS expected_eating_effects
        """,
    }

    def __init__(self, cfg: Config):
        self.driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
            max_connection_lifetime=300
        )

    def close(self):
        self.driver.close()

    def run(self, query_key: str, **params) -> list[dict]:
        cypher = self.QUERIES[query_key]
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def run_raw(self, cypher: str, **params) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def vector_search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        return self.run("vector_search_chemo", top_k=top_k, query_vector=query_vector)

    def fulltext_search(self, query_text: str, limit: int = 10) -> list[dict]:
        return self.run("fulltext_search", query_text=query_text, limit=limit)

    def store_embedding(self, label: str, name: str, embedding: list[float]):
        """Write embedding back to node property for vector index."""
        cypher = f"""
            MATCH (n:{label} {{name: $name}})
            CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
        """
        self.run_raw(cypher, name=name, embedding=embedding)


# ════════════════════════════════════════════════════════════
# 4. VECTOR DB RETRIEVER (Qdrant / FAISS)
# ════════════════════════════════════════════════════════════

class VectorRetriever:
    """Wraps either Qdrant or FAISS; returns ranked text chunks."""

    def __init__(self, cfg: Config, embedder: EmbeddingClient):
        self.cfg      = cfg
        self.embedder = embedder
        self._mode    = None

        if QDRANT_AVAILABLE and cfg.qdrant_url:
            self._qdrant = QdrantClient(
                url=cfg.qdrant_url,
                api_key=cfg.qdrant_api_key or None
            )
            self._mode = "qdrant"
            logger.info("VectorRetriever: using Qdrant at %s", cfg.qdrant_url)

        elif FAISS_AVAILABLE and os.path.exists(cfg.faiss_index_path):
            self._faiss_index = faiss.read_index(cfg.faiss_index_path)
            with open(cfg.faiss_meta_path) as fh:
                self._faiss_meta = json.load(fh)
            self._mode = "faiss"
            logger.info("VectorRetriever: using FAISS index at %s", cfg.faiss_index_path)
        else:
            logger.warning("No vector backend available; vector retrieval disabled.")

    def search(self, query: str, top_k: int = 5,
               filter_cancer: Optional[str] = None) -> list[dict]:
        if self._mode is None:
            return []

        vec = self.embedder.embed(query)

        if self._mode == "qdrant":
            return self._qdrant_search(vec, top_k, filter_cancer)
        elif self._mode == "faiss":
            return self._faiss_search(vec, top_k)
        return []

    def _qdrant_search(self, vec, top_k, filter_cancer):
        filt = None
        if filter_cancer:
            filt = Filter(must=[FieldCondition(
                key="metadata.cancer_type",
                match=MatchValue(value=filter_cancer)
            )])
        hits = self._qdrant.search(
            collection_name=self.cfg.qdrant_collection,
            query_vector=vec,
            limit=top_k,
            query_filter=filt,
            with_payload=True
        )
        return [
            {
                "id":      h.id,
                "score":   h.score,
                "text":    h.payload.get("text", ""),
                "source":  h.payload.get("source", ""),
                "metadata": h.payload.get("metadata", {})
            }
            for h in hits
        ]

    def _faiss_search(self, vec, top_k):
        arr = np.array([vec], dtype=np.float32)
        distances, indices = self._faiss_index.search(arr, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = self._faiss_meta[str(idx)]
            results.append({
                "id":     idx,
                "score":  float(1.0 / (1.0 + dist)),   # convert L2 distance to score
                "text":   meta.get("text", ""),
                "source": meta.get("source", ""),
                "metadata": meta
            })
        return results


# ════════════════════════════════════════════════════════════
# 5. CONTEXT FUSION – Reciprocal Rank Fusion (RRF)
# ════════════════════════════════════════════════════════════

def rrf_merge(
    vector_results: list[dict],
    graph_results:  list[dict],
    k: int = 60
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Each item scored as sum of 1/(k + rank) across lists.
    """
    scores: dict[str, float] = {}
    sources: dict[str, dict] = {}

    def key_of(item):
        return item.get("id") or item.get("name") or item.get("drug") or str(item)

    for rank, item in enumerate(vector_results):
        k_str = str(key_of(item))
        scores[k_str] = scores.get(k_str, 0.0) + 1.0 / (k + rank + 1)
        sources[k_str] = {**item, "_source_type": "vector"}

    for rank, item in enumerate(graph_results):
        k_str = str(key_of(item))
        scores[k_str] = scores.get(k_str, 0.0) + 1.0 / (k + rank + 1)
        if k_str not in sources:
            sources[k_str] = {**item, "_source_type": "graph"}
        else:
            sources[k_str]["_source_type"] = "both"

    merged = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [{**sources[k], "_rrf_score": scores[k]} for k in merged]


# ════════════════════════════════════════════════════════════
# 6. QUERY ROUTER
# ════════════════════════════════════════════════════════════

def classify_query(query: str) -> dict:
    """
    Lightweight rule-based classifier; in production replace with
    a small LLM classification call.
    Returns: {mode, graph_query_key, graph_params, needs_vector}
    """
    q = query.lower()

    # ── Cancer + drug + eating effects ──────────────────────
    if any(c in q for c in ["lung cancer","breast cancer","leukemia","osteosarcoma","skin cancer","melanoma"]):
        cancer_map = {
            "lung cancer":   "Lung Cancer",
            "breast cancer": "Breast Cancer",
            "leukemia":      "Acute Leukemia",
            "osteosarcoma":  "Osteosarcoma",
            "skin cancer":   "Skin Cancer",
            "melanoma":      "Skin Cancer",
        }
        cancer = next(v for k, v in cancer_map.items() if k in q)
        return {
            "mode": "graph_primary",
            "graph_query_key": "cancer_drugs_effects",
            "graph_params": {"cancer_name": cancer},
            "needs_vector": True,
        }

    # ── Drug interaction questions ───────────────────────────
    if any(kw in q for kw in ["interaction","interact","takes","on warfarin","on aspirin","on omeprazole","concurrent"]):
        drug_map = {
            "cisplatin":       "Cisplatin",
            "paclitaxel":      "Paclitaxel",
            "methotrexate":    "Methotrexate",
            "vincristine":     "Vincristine",
            "doxorubicin":     "Doxorubicin",
            "cyclophosphamide":"Cyclophosphamide",
            "etoposide":       "Etoposide",
            "capecitabine":    "Capecitabine",
        }
        chemo = next((v for k, v in drug_map.items() if k in q), None)
        if chemo:
            return {
                "mode": "graph_primary",
                "graph_query_key": "chemo_drug_interactions",
                "graph_params": {"chemo_name": chemo},
                "needs_vector": False,
            }

    # ── Food / diet questions ────────────────────────────────
    if any(kw in q for kw in ["eat","food","diet","avoid","nutrition","nausea","vomiting","diarrhoea","constipation","mucositis"]):
        drug_map = {
            "cisplatin":  "Cisplatin", "paclitaxel": "Paclitaxel",
            "methotrexate": "Methotrexate", "doxorubicin": "Doxorubicin",
            "fluorouracil": "Fluorouracil", "capecitabine": "Capecitabine",
        }
        drug = next((v for k, v in drug_map.items() if k in q), None)
        if drug:
            return {
                "mode": "graph_primary",
                "graph_query_key": "food_guidance_for_drug",
                "graph_params": {"drug_name": drug},
                "needs_vector": True,
            }

    # ── Protocol questions ───────────────────────────────────
    protocol_map = {
        "ac-t":       "AC-T (Breast)",
        "map":        "MAP (Osteosarcoma)",
        "7+3":        "7+3 AML Induction",
        "hyper-cvad": "Hyper-CVAD (ALL)",
    }
    for k, v in protocol_map.items():
        if k in q:
            return {
                "mode": "graph_primary",
                "graph_query_key": "protocol_full_detail",
                "graph_params": {"protocol_name": v},
                "needs_vector": True,
            }

    # ── Default: hybrid search ───────────────────────────────
    return {"mode": "hybrid", "graph_query_key": "fulltext_search",
            "graph_params": {"query_text": query, "limit": 10},
            "needs_vector": True}


# ════════════════════════════════════════════════════════════
# 7. CONTEXT FORMATTER
# ════════════════════════════════════════════════════════════

def format_context(merged_results: list[dict], max_chars: int = 8000) -> str:
    """Serialise merged graph + vector results into LLM context."""
    lines = []
    for i, r in enumerate(merged_results[:20]):
        src = r.get("_source_type", "unknown")
        lines.append(f"\n[Result {i+1} | source:{src} | rrf:{r.get('_rrf_score', 0):.4f}]")

        # Graph result fields
        for field in ["cancer", "drug", "drug_class", "mechanism", "eating_effect",
                      "severity", "frequency", "foods_to_avoid", "foods_to_eat",
                      "management_tip", "avoid_foods", "recommended_foods",
                      "nutrition_guidelines", "interaction_description",
                      "clinical_action", "eating_relevance", "mandatory_guidelines",
                      "non_chemo_drug", "compounded_effect", "mitigation_foods"]:
            val = r.get(field)
            if val is not None and val != [] and val != "":
                lines.append(f"  {field}: {val}")

        # Vector chunk text
        if "text" in r and r["text"]:
            snippet = r["text"][:500]
            lines.append(f"  text_chunk: {snippet}...")
            if "source" in r:
                lines.append(f"  source_doc: {r['source']}")

    context = "\n".join(lines)
    return context[:max_chars]


# ════════════════════════════════════════════════════════════
# 8. MAIN GraphRAG PIPELINE
# ════════════════════════════════════════════════════════════

class GraphRAGPipeline:
    """
    Unified GraphRAG pipeline:
      Neo4j (structured) + Qdrant/FAISS (semantic) → Claude (generation)
    """

    SYSTEM_PROMPT = """You are an expert oncology clinical assistant with deep knowledge of:
- Chemotherapy regimens and mechanisms
- Drug-drug interactions between chemotherapy and other medications
- Nutritional management during cancer treatment
- Evidence-based dietary guidance for specific chemotherapy side effects

When answering, always:
1. Cite the specific cancer type and drug involved
2. State the severity of any eating problems (High/Moderate/Low)
3. Provide actionable food recommendations (what to eat AND what to avoid)
4. Flag any critical drug interactions that affect eating (e.g. warfarin-capecitabine)
5. Note mandatory nutritional supplementation (e.g. B12/folate for pemetrexed)

Use the retrieved context as your primary source. If information is absent, say so clearly."""

    def __init__(self, cfg: Config = None):
        self.cfg       = cfg or Config()
        self.embedder  = EmbeddingClient(self.cfg)
        self.graph     = GraphRetriever(self.cfg)
        self.vector    = VectorRetriever(self.cfg, self.embedder)
        self.anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

    def query(self, user_query: str, verbose: bool = False) -> dict:
        logger.info("Query: %s", user_query)

        # ── 1. Route query ───────────────────────────────────
        route = classify_query(user_query)
        if verbose:
            logger.info("Route: %s", route)

        graph_results = []
        vector_results = []

        # ── 2. Graph retrieval ───────────────────────────────
        try:
            graph_results = self.graph.run(
                route["graph_query_key"],
                **route["graph_params"]
            )
            logger.info("Graph returned %d rows", len(graph_results))
        except Exception as exc:
            logger.warning("Graph query failed: %s", exc)

        # ── 3. Vector retrieval ──────────────────────────────
        if route.get("needs_vector"):
            try:
                vector_results = self.vector.search(user_query, top_k=self.cfg.top_k_vector)
                logger.info("Vector returned %d results", len(vector_results))
            except Exception as exc:
                logger.warning("Vector search failed: %s", exc)

        # ── 4. Merge (RRF) ───────────────────────────────────
        merged = rrf_merge(vector_results, graph_results, k=self.cfg.rrf_k)

        # ── 5. Format context ────────────────────────────────
        context = format_context(merged)

        # ── 6. Generate answer (Claude) ──────────────────────
        prompt = f"""<retrieved_context>
{context}
</retrieved_context>

<user_question>
{user_query}
</user_question>

Using the retrieved context above, provide a comprehensive, clinically accurate answer."""

        message = self.anthropic.messages.create(
            model=self.cfg.anthropic_model,
            max_tokens=self.cfg.max_tokens,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = message.content[0].text

        return {
            "query":          user_query,
            "answer":         answer,
            "graph_results":  graph_results,
            "vector_results": vector_results,
            "merged_context": merged,
            "route":          route,
        }

    def close(self):
        self.graph.close()


# ════════════════════════════════════════════════════════════
# 9. BULK NODE EMBEDDING SCRIPT
#    Run once to populate `embedding` properties in Neo4j
#    so the vector index can be used for similarity search
# ════════════════════════════════════════════════════════════

def embed_all_nodes(cfg: Config = None):
    """
    Fetches all ChemoDrug, Cancer, EatingAdverseEffect, NonChemoDrug nodes
    from Neo4j, generates embeddings, and writes them back.
    Run this script once (or when new nodes are added).
    """
    cfg      = cfg or Config()
    embedder = EmbeddingClient(cfg)
    graph    = GraphRetriever(cfg)

    node_configs = [
        {
            "label": "ChemoDrug",
            "cypher": "MATCH (n:ChemoDrug) RETURN n.name AS name, (n.name + ' ' + coalesce(n.mechanism,'') + ' ' + coalesce(n.drug_class,'') + ' ' + coalesce(n.notes,'')) AS text",
        },
        {
            "label": "Cancer",
            "cypher": "MATCH (n:Cancer) RETURN n.name AS name, (n.name + ' ' + coalesce(n.description,'') + ' ' + coalesce(n.subtype,'')) AS text",
        },
        {
            "label": "EatingAdverseEffect",
            "cypher": "MATCH (n:EatingAdverseEffect) RETURN n.name AS name, (n.name + ' ' + coalesce(n.description,'') + ' ' + coalesce(n.management_tip,'')) AS text",
        },
        {
            "label": "NonChemoDrug",
            "cypher": "MATCH (n:NonChemoDrug) RETURN n.name AS name, (n.name + ' ' + coalesce(n.mechanism,'') + ' ' + coalesce(n.drug_class,'')) AS text",
        },
    ]

    for nc in node_configs:
        rows = graph.run_raw(nc["cypher"])
        logger.info("Embedding %d %s nodes...", len(rows), nc["label"])
        texts = [r["text"] for r in rows]
        names = [r["name"] for r in rows]
        vecs  = embedder.embed_batch(texts)
        for name, vec in zip(names, vecs):
            graph.store_embedding(nc["label"], name, vec)
        logger.info("Done embedding %s.", nc["label"])

    graph.close()
    logger.info("All node embeddings stored in Neo4j.")


# ════════════════════════════════════════════════════════════
# 10. SAMPLE USAGE / DEMO
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = GraphRAGPipeline()

    DEMO_QUERIES = [
        "What are the eating problems caused by cisplatin in lung cancer patients and what foods should be avoided?",
        "A breast cancer patient on AC-T is also taking warfarin. What are the risks and dietary precautions?",
        "What nutritional guidelines are mandatory for a patient receiving pemetrexed?",
        "What foods can help manage mucositis caused by methotrexate in osteosarcoma treatment?",
        "What is the interaction between voriconazole and vincristine in leukemia treatment and how does it affect eating?",
        "A melanoma patient on nivolumab is also on levothyroxine for hypothyroidism. What should I monitor?",
    ]

    for q in DEMO_QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        print('='*70)
        result = pipeline.query(q, verbose=True)
        print(result["answer"])

    pipeline.close()
