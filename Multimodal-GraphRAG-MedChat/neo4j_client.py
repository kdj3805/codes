"""
neo4j_client.py
---------------
Thread-safe Neo4j driver wrapper for the cancer GraphRAG system.
Fully compatible with Neo4j Aura + Windows SSL fix.
"""

from __future__ import annotations

import logging
import os
import ssl
import certifi
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Session
from neo4j.exceptions import AuthError, ServiceUnavailable

log = logging.getLogger(__name__)


class Neo4jClient:
    _instance: "Neo4jClient | None" = None

    def __init__(self) -> None:
        # ✅ CORRECT ENV VARIABLES
        self._uri      = os.getenv("NEO4J_URI")
        self._user     = os.getenv("NEO4J_USER")   # FIXED
        self._password = os.getenv("NEO4J_PASSWORD")

        if not self._uri or not self._user or not self._password:
            raise ValueError(
                "Missing Neo4j credentials. Check .env:\n"
                "NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"
            )

        self._driver = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            log.info("Initializing Neo4j driver...")

            # 🔥 FIX SSL CERTIFICATE ISSUE (WINDOWS)
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                ssl_context=ssl_context,   # ✅ KEY FIX
            )

            self._driver.verify_connectivity()

            log.info("✅ Neo4j connected successfully → %s", self._uri)

        except AuthError:
            log.error("❌ Authentication failed. Check credentials.")
            raise

        except ServiceUnavailable as exc:
            log.error("❌ Neo4j unavailable: %s", exc)
            raise

        except Exception as exc:
            log.error("❌ Unexpected Neo4j error: %s", exc)
            raise

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    @contextmanager
    def session(self, database: str = None) -> Generator[Session, None, None]:
        if database:
            with self._driver.session(database=database) as s:
                yield s
        else:
            with self._driver.session() as s:
                yield s

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def run_query(self, cypher: str, parameters=None) -> list[dict]:
        with self.session() as s:
            result = s.run(cypher, parameters or {})
            return [dict(r) for r in result]

    def run_write(self, cypher: str, parameters=None) -> None:
        with self.session() as s:
            s.execute_write(lambda tx: tx.run(cypher, parameters or {}))

    def run_write_batch(self, cypher: str, rows: list[dict]) -> None:
        with self.session() as s:
            s.execute_write(lambda tx: tx.run(cypher, {"rows": rows}))

    def ping(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "Neo4jClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_client() -> Neo4jClient:
    return Neo4jClient.get_instance()