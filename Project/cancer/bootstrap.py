"""
bootstrap.py
------------
One-shot setup: schema + data load.
Run once after Neo4j is available.

Usage:
    python bootstrap.py
"""

import logging
from graph_schema import bootstrap_schema
from graph_data_loader import load_all
from neo4j_client import get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

log = logging.getLogger(__name__)


def main() -> None:
    log.info("Connecting to Neo4j …")
    client = get_client()

    if not client.ping():
        log.error("Cannot reach Neo4j. Check NEO4J_URI / credentials.")
        return

    log.info("Schema bootstrap …")
    bootstrap_schema(client)

    log.info("Loading graph data …")
    load_all(client)

    # Verify counts
    counts_cypher = """
    MATCH (n) 
    RETURN labels(n)[0] AS label, count(*) AS count
    ORDER BY label
    """
    rel_cypher = """
    MATCH ()-[r]->()
    RETURN type(r) AS rel_type, count(*) AS count
    ORDER BY rel_type
    """

    log.info("\n=== NODE COUNTS ===")
    for row in client.run_query(counts_cypher):
        log.info("  %-20s %d", row["label"], row["count"])

    log.info("\n=== RELATIONSHIP COUNTS ===")
    for row in client.run_query(rel_cypher):
        log.info("  %-25s %d", row["rel_type"], row["count"])

    log.info("\nBootstrap complete. GraphRAG ready.")


if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────
# requirements.txt contents (reference)
# ──────────────────────────────────────────────
REQUIREMENTS = """
neo4j>=5.14.0
groq>=0.4.0
streamlit>=1.35.0
python-dotenv>=1.0.0
"""

# ──────────────────────────────────────────────
# .env template
# ──────────────────────────────────────────────
ENV_TEMPLATE = """
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
"""
