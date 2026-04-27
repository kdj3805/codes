# GraphRAG: Cancer–Chemo–Nutrition Knowledge Graph
### Neo4j / AuraDB + Qdrant / FAISS ← → Claude Integration Guide

---

## 📋 What's in This Package

| File | Purpose |
|---|---|
| `01_schema_constraints.cypher` | Uniqueness constraints, full-text indexes, vector indexes |
| `02_cancer_chemo_eating.cypher` | Core graph: 5 cancers, 28 chemo drugs, 12 eating effects, 18 foods, 10 guidelines, 7 biomarkers, 6 protocols (~120 relationships) |
| `03_drug_interactions.cypher` | 12 ailments, 20 non-chemo drugs, 25 drug interaction nodes + all relationships |
| `04_graphrag_integration.py` | Python GraphRAG pipeline integrating Neo4j with Qdrant/FAISS and Claude |
| `README.md` | This guide |

**Total data entry points: 155+ nodes, 200+ relationships**

---

## 🗺️ Graph Schema

```
Cancer
  ├─[:TREATED_WITH]──────────► ChemoDrug
  │                                │
  ├─[:HAS_BIOMARKER]──────► Biomarker     ├─[:CAUSES_EATING_EFFECT]──► EatingAdverseEffect
  │                                │                                         │
  └─[:TREATED_BY_PROTOCOL]─► TreatmentProtocol                   ├─[:WORSENED_BY]────► FoodItem
        │                      │                                  ├─[:RELIEVED_BY]────► FoodItem
        └─[:INCLUDES_DRUG]────►┘                                  └─[:MANAGED_BY]─────► NutritionGuideline
                                │
                        ├─[:MAY_CAUSE]──────────────────────► SideEffect
                        │                                          │
                        └─[:DESCRIBED_BY]────► DrugInteraction    └─[:LEADS_TO_EATING_EFFECT]──► EatingAdverseEffect
                                                    ▲
NonChemoDrug ─────[:HAS_INTERACTION_WITH]──► ChemoDrug
    │
    └─[:TREATS]──────────────────────────► Ailment
```

---

## 🚀 Step-by-Step Setup

### Step 1 — Create Neo4j AuraDB Instance

1. Go to [console.neo4j.io](https://console.neo4j.io)
2. Click **New Instance** → Choose **AuraDB Free** (5GB) or **Professional**
3. Save the connection URI and credentials  
   Format: `neo4j+s://<id>.databases.neo4j.io`
4. Set environment variables:

```bash
export NEO4J_URI="neo4j+s://xxxxxxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
```

### Step 2 — Install Python Dependencies

```bash
pip install neo4j anthropic openai numpy

# For Qdrant vector store (optional – use if your existing pipeline uses it):
pip install qdrant-client

# For FAISS vector store (optional – use if your existing pipeline uses FAISS):
pip install faiss-cpu   # or faiss-gpu
```

### Step 3 — Run Cypher Scripts in Order

Use **Neo4j Browser**, **Neo4j Desktop**, or `cypher-shell`:

```bash
# Option A: Neo4j Browser (paste content directly)
# Open: https://<your-instance>.databases.neo4j.io/browser/

# Option B: cypher-shell (CLI)
cypher-shell -a $NEO4J_URI -u $NEO4J_USER -p $NEO4J_PASSWORD \
  --file 01_schema_constraints.cypher

cypher-shell -a $NEO4J_URI -u $NEO4J_USER -p $NEO4J_PASSWORD \
  --file 02_cancer_chemo_eating.cypher

cypher-shell -a $NEO4J_URI -u $NEO4J_USER -p $NEO4J_PASSWORD \
  --file 03_drug_interactions.cypher

# Option C: Python driver
python - << 'EOF'
from neo4j import GraphDatabase
driver = GraphDatabase.driver(
    "neo4j+s://xxx.databases.neo4j.io",
    auth=("neo4j", "password")
)
for fname in ["01_schema_constraints.cypher",
              "02_cancer_chemo_eating.cypher",
              "03_drug_interactions.cypher"]:
    with open(fname) as f:
        statements = [s.strip() for s in f.read().split(";") if s.strip()]
    with driver.session() as session:
        for stmt in statements:
            session.run(stmt)
    print(f"Loaded {fname}")
driver.close()
EOF
```

### Step 4 — Verify the Graph

Run these verification Cypher queries in Neo4j Browser:

```cypher
// Node counts
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC;

// Sample cancer-drug-effect path
MATCH (c:Cancer {name:"Lung Cancer"})-[:TREATED_WITH]->(d:ChemoDrug)
      -[:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
RETURN c.name, d.name, e.name LIMIT 20;

// Sample drug interaction path
MATCH (n:NonChemoDrug)-[:HAS_INTERACTION_WITH]->(c:ChemoDrug)
      -[:DESCRIBED_BY]->(i:DrugInteraction)
RETURN n.name, c.name, i.severity, i.description LIMIT 10;

// Full path: Cancer → Drug → Interaction → Eating Effect
MATCH path = (cancer:Cancer)-[:TREATED_WITH]->(chemo:ChemoDrug)
             <-[:HAS_INTERACTION_WITH]-(non_chemo:NonChemoDrug)
RETURN path LIMIT 5;
```

### Step 5 — Generate and Store Node Embeddings

This populates the `embedding` property for the Neo4j vector index:

```bash
export OPENAI_API_KEY="sk-..."  # for text-embedding-3-small

python - << 'EOF'
from graphrag_integration import embed_all_nodes, Config
cfg = Config()
embed_all_nodes(cfg)
EOF
```

> **Alternative**: Use a local sentence-transformer model to avoid OpenAI costs:
> ```python
> from sentence_transformers import SentenceTransformer
> model = SentenceTransformer("all-MiniLM-L6-v2")
> vec = model.encode("text here").tolist()
> ```

### Step 6 — Connect to Your Existing Vector DB

#### If using Qdrant:

```python
# Your existing collection is already set in Config:
# cfg.qdrant_collection = "cancer_chemo_docs"
# cfg.qdrant_url = "http://localhost:6333"

# Ensure your Qdrant collection has payload fields:
# - text (string): the chunk content
# - source (string): filename / document source
# - metadata.cancer_type (string): optional filter
```

#### If using FAISS:

```python
# Your FAISS index should be at cfg.faiss_index_path
# The metadata JSON at cfg.faiss_meta_path should contain:
# { "0": {"text": "...", "source": "...", ...}, "1": {...}, ... }
```

#### Ingesting the Markdown Files into Your Vector DB:

```python
import json
from pathlib import Path

# Example for Qdrant ingestion of the uploaded cancer review markdowns
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from graphrag_integration import EmbeddingClient, Config

cfg      = Config()
embedder = EmbeddingClient(cfg)
qdrant   = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key or None)

# Create collection if not exists
try:
    qdrant.create_collection(
        collection_name=cfg.qdrant_collection,
        vectors_config=VectorParams(size=cfg.embedding_dim, distance=Distance.COSINE),
    )
except Exception:
    pass  # collection exists

# Ingest each markdown file
source_files = {
    "lung-cancer-review.md":          "Lung Cancer",
    "breast-cancer-review.md":        "Breast Cancer",
    "acute-leukemia-review.md":       "Acute Leukemia",
    "osteosarcoma-review.md":         "Osteosarcoma",
    "melanoma-skin-cancer-review.md": "Skin Cancer",
    "skin-cancer-types-review.md":    "Skin Cancer",
}

point_id = 0
for fname, cancer_type in source_files.items():
    text = Path(fname).read_text(encoding="utf-8")
    # Chunk by paragraphs (simple splitter; use LangChain splitter in production)
    chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
    vecs   = embedder.embed_batch(chunks[:50])  # limit per call
    points = [
        PointStruct(
            id=point_id + i,
            vector=v,
            payload={
                "text":     chunks[i],
                "source":   fname,
                "metadata": {"cancer_type": cancer_type, "chunk_idx": i}
            }
        )
        for i, v in enumerate(vecs)
    ]
    qdrant.upsert(collection_name=cfg.qdrant_collection, points=points)
    point_id += len(points)
    print(f"Ingested {len(points)} chunks from {fname}")
```

### Step 7 — Run the GraphRAG Pipeline

```python
from graphrag_integration import GraphRAGPipeline

pipeline = GraphRAGPipeline()

result = pipeline.query(
    "A breast cancer patient receiving AC-T chemotherapy is also on warfarin. "
    "What are the drug interactions and what dietary modifications are needed?"
)
print(result["answer"])
```

---

## 🔄 Architecture Deep-Dive

### Query Flow

```
User Query
    │
    ▼
classify_query()  ──────────────────────────────────────────────────────
    │                                                                    │
    │ graph_primary: structured query                     hybrid: both  │
    ▼                                                                    ▼
GraphRetriever.run()                               GraphRetriever.fulltext_search()
    │                                              VectorRetriever.search()
    │   Cypher result rows (graph facts)               │  Semantic chunks
    │                                                  │
    └───────────────────────┬──────────────────────────┘
                            ▼
                     rrf_merge() ← Reciprocal Rank Fusion
                            │
                            ▼
                     format_context() ← Serialise to string
                            │
                            ▼
                     Claude API call
                      (system prompt + context + question)
                            │
                            ▼
                      Final Answer
```

### Node Type Reference

| Node Label | Count | Key Properties |
|---|---|---|
| `Cancer` | 5 | name, subtype, icd10, five_year_survival |
| `ChemoDrug` | 28 | name, drug_class, mechanism, route, typical_dose |
| `EatingAdverseEffect` | 12 | name, description, management_tip, onset |
| `FoodItem` | 18 | name, category, notes |
| `NutritionGuideline` | 10 | id, text, applicable_to, evidence_level |
| `Biomarker` | 7 | name, cancer, therapeutic_relevance |
| `TreatmentProtocol` | 6 | name, description, cancer, setting |
| `NonChemoDrug` | 20 | name, drug_class, mechanism |
| `Ailment` | 12 | name, icd10, prevalence |
| `DrugInteraction` | 25 | id, severity, mechanism, clinical_action, eating_relevance |
| `SideEffect` | 5 | name, nutrition_impact, management |

### Relationship Type Reference

| Relationship | From → To | Key Properties |
|---|---|---|
| `TREATED_WITH` | Cancer → ChemoDrug | line, evidence, regimen |
| `CAUSES_EATING_EFFECT` | ChemoDrug → EatingAdverseEffect | severity, frequency, onset |
| `WORSENED_BY` | EatingAdverseEffect → FoodItem | evidence |
| `RELIEVED_BY` | EatingAdverseEffect → FoodItem | evidence, notes |
| `MANAGED_BY` / `MANAGES` | NutritionGuideline ↔ Effect | — |
| `REQUIRED_FOR` | NutritionGuideline → ChemoDrug | — |
| `HAS_BIOMARKER` | Cancer → Biomarker | — |
| `TREATED_BY_PROTOCOL` | Cancer → TreatmentProtocol | — |
| `INCLUDES_DRUG` | TreatmentProtocol → ChemoDrug | — |
| `TREATS` | NonChemoDrug → Ailment | — |
| `HAS_INTERACTION_WITH` | NonChemoDrug → ChemoDrug | — |
| `DESCRIBED_BY` | Drug → DrugInteraction | — |
| `COMPOUNDS_EATING_EFFECT` | DrugInteraction → EatingAdverseEffect | note |
| `MAY_CAUSE` | ChemoDrug → SideEffect | — |
| `LEADS_TO_EATING_EFFECT` | SideEffect → EatingAdverseEffect | — |
| `WARNS_ABOUT` | NutritionGuideline → FoodItem | — |

---

## 🔧 Extending the Graph

### Adding New Drug Nodes

```cypher
MERGE (d:ChemoDrug {name: 'Olaparib'})
SET d.drug_class   = 'PARP inhibitor / targeted therapy',
    d.mechanism    = 'Inhibits PARP1/2, preventing DNA single-strand break repair in BRCA-deficient cells',
    d.route        = 'Oral',
    d.typical_dose = '300 mg twice daily';

// Link to cancer
MATCH (c:Cancer {name:'Breast Cancer'}),(d:ChemoDrug {name:'Olaparib'})
MERGE (c)-[:TREATED_WITH {line:'metastatic BRCA+', evidence:'Level 1A'}]->(d);
```

### Adding New Interaction Nodes

```cypher
MERGE (i:DrugInteraction {id:'DI_OMEP_CAPEC'})
SET i.description = 'Omeprazole + Capecitabine: PPIs may reduce capecitabine absorption by raising gastric pH',
    i.severity    = 'Mild',
    i.mechanism   = 'Altered gastric pH changes capecitabine dissolution kinetics',
    i.clinical_action = 'Take capecitabine with a meal; avoid PPI within 2h of dose if possible';
```

### Bulk Ingestion via Python

```python
from neo4j import GraphDatabase
import csv

driver = GraphDatabase.driver(NEO4J_URI, auth=(USER, PASSWORD))
with driver.session() as session:
    with open("new_drugs.csv") as f:
        for row in csv.DictReader(f):
            session.run("""
                MERGE (d:ChemoDrug {name: $name})
                SET d.drug_class = $drug_class,
                    d.mechanism  = $mechanism
            """, **row)
driver.close()
```

---

## 🐛 Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ServiceUnavailable` | Wrong URI or network | Check `NEO4J_URI`; test with Neo4j Browser first |
| `AuthError` | Wrong credentials | Reset password in AuraDB console |
| Vector index error | Neo4j < 5.11 | Upgrade to AuraDB Enterprise or use fulltext index only |
| `CALL db.index.vector` fails | Index not yet populated | Run `embed_all_nodes()` script |
| Qdrant connection refused | Qdrant not running | `docker run -p 6333:6333 qdrant/qdrant` |
| Empty graph results | Query typo in node names | Node names are case-sensitive; check exact spelling |
| High latency | Complex Cypher traversal | Add `LIMIT` clauses; use indexed properties in `MATCH` |

---

## 📊 Sample Queries for Testing

```cypher
-- Q1: What eating problems does cisplatin cause?
MATCH (d:ChemoDrug {name:'Cisplatin'})-[r:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
RETURN d.name, e.name, r.severity, r.frequency;

-- Q2: What foods relieve nausea?
MATCH (e:EatingAdverseEffect {name:'Nausea'})-[:RELIEVED_BY]->(f:FoodItem)
RETURN f.name, f.notes;

-- Q3: Severe drug interactions with methotrexate
MATCH (n:NonChemoDrug)-[:HAS_INTERACTION_WITH]->(c:ChemoDrug {name:'Methotrexate'})
MATCH (i:DrugInteraction) WHERE i.id CONTAINS 'MTX' AND i.severity = 'Severe'
RETURN n.name, i.description, i.clinical_action;

-- Q4: Full path from cancer to dietary guidance
MATCH (c:Cancer {name:'Breast Cancer'})-[:TREATED_WITH]->(d:ChemoDrug)
      -[:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
      -[:RELIEVED_BY]->(f:FoodItem)
RETURN c.name, d.name, e.name, f.name;

-- Q5: All mandatory nutritional supplements
MATCH (g:NutritionGuideline)-[:REQUIRED_FOR]->(d:ChemoDrug)
RETURN g.text, d.name, g.evidence_level;
```

---

## 🔒 Production Checklist

- [ ] Use AuraDB Professional or Enterprise (not Free tier) for production workloads
- [ ] Store credentials in secrets manager (AWS Secrets Manager / Vault), not environment variables
- [ ] Add connection pooling (`max_connection_pool_size` in driver config)
- [ ] Set query timeouts to prevent runaway traversals
- [ ] Use read replicas for retrieval queries
- [ ] Implement exponential backoff retry logic around Neo4j calls
- [ ] Monitor with Neo4j Metrics endpoint or Datadog integration
- [ ] Refresh node embeddings when graph data changes (schedule weekly job)
- [ ] Version-control all Cypher scripts and treat schema changes like database migrations
