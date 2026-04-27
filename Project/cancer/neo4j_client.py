"""
neo4j_client.py
---------------
Thread-safe Neo4j driver wrapper for the cancer GraphRAG system.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Session
from neo4j.exceptions import AuthError, ServiceUnavailable

log = logging.getLogger(__name__)


class Neo4jClient:
    """Singleton-style Neo4j driver wrapper."""

    _instance: "Neo4jClient | None" = None

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._uri      = uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        self._user     = user     or os.getenv("NEO4J_USER",     "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._driver   = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_pool_size=20,
                connection_acquisition_timeout=30,
            )
            self._driver.verify_connectivity()
            log.info("Neo4j connected → %s", self._uri)
        except (AuthError, ServiceUnavailable) as exc:
            log.error("Neo4j connection failed: %s", exc)
            raise

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            log.info("Neo4j driver closed.")

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    @contextmanager
    def session(self, database: str = "neo4j") -> Generator[Session, None, None]:
        with self._driver.session(database=database) as s:
            yield s

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def run_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[dict]:
        with self.session(database=database) as s:
            result = s.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def run_write(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> None:
        with self.session(database=database) as s:
            s.execute_write(lambda tx: tx.run(cypher, parameters or {}))

    def run_write_batch(
        self,
        cypher: str,
        rows: list[dict],
        database: str = "neo4j",
    ) -> None:
        """Execute a parameterised write using UNWIND for bulk inserts."""
        with self.session(database=database) as s:
            s.execute_write(lambda tx: tx.run(cypher, {"rows": rows}))

    def ping(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "Neo4jClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_client() -> Neo4jClient:
    return Neo4jClient.get_instance()
