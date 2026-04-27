"""
bootstrap.py
------------
One-shot setup: schema + data load.
Run once after Neo4j is available.

Usage:
    python bootstrap.py
"""

# ─────────────────────────────────────────────
# Load environment variables FIRST (CRITICAL)
# ─────────────────────────────────────────────
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# ─────────────────────────────────────────────
# Imports (AFTER env is loaded)
# ─────────────────────────────────────────────
import logging
from graph_schema import bootstrap_schema
from graph_data_loader import load_all
from neo4j_client import get_client

# ─────────────────────────────────────────────
# Logging config
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Debug: Verify .env is loaded
# ─────────────────────────────────────────────
print("\n=== ENV DEBUG ===")
print("NEO4J_URI:", os.getenv("NEO4J_URI"))
print("NEO4J_USER:", os.getenv("NEO4J_USER"))
print("NEO4J_PASSWORD:", "SET" if os.getenv("NEO4J_PASSWORD") else "NOT SET")
print("=================\n")


def main() -> None:
    # ─────────────────────────────────────────
    # Step 1: Connect to Neo4j
    # ─────────────────────────────────────────
    log.info("Connecting to Neo4j …")
    client = get_client()

    if not client.ping():
        log.error("❌ Cannot reach Neo4j. Check NEO4J_URI / credentials.")
        return

    log.info("✅ Connected to Neo4j")

    # ─────────────────────────────────────────
    # Step 2: Create schema
    # ─────────────────────────────────────────
    log.info("Schema bootstrap …")
    bootstrap_schema(client)

    # ─────────────────────────────────────────
    # Step 3: Load data
    # ─────────────────────────────────────────
    log.info("Loading graph data …")
    load_all(client)

    # ─────────────────────────────────────────
    # Step 4: Verify nodes
    # ─────────────────────────────────────────
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

    log.info("\n🚀 Bootstrap complete. GraphRAG ready.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()