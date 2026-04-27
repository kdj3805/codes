# ingest.py — CLEAN + DETERMINISTIC VERSION

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional

import fitz
import requests
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from groq import Groq
import easyocr

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

CHROMA_PERSIST_DIR = "./chroma_store"
CHROMA_COLLECTION = "enterprise_docs"

SEVERITIES = ["Critical", "High", "Medium", "Low"]

TIME_PATTERN = re.compile(
    r'(\d+)\s*(h|hr|hrs|hour|hours|days|day)',
    re.IGNORECASE
)

_OCR_READER = None


# ─────────────────────────────────────────────────────────────
# CHROMA
# ─────────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


# ─────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────

def embed_text(text: str):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        return resp.json()["embedding"]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# ESCALATION BLOCK PARSER (DETERMINISTIC)
# ─────────────────────────────────────────────────────────────

def parse_escalation_blocks(page_text: str):
    """
    Deterministically slice text between severity markers.
    """
    chunks = []

    lower_text = page_text.lower()

    if "escalation" not in lower_text:
        return []

    # Find all severity positions
    severity_positions = []

    for sev in SEVERITIES:
        for match in re.finditer(rf"\b{sev}\b", page_text):
            severity_positions.append((match.start(), sev))

    if not severity_positions:
        return []

    severity_positions.sort()

    for i, (start_pos, severity) in enumerate(severity_positions):
        end_pos = severity_positions[i + 1][0] if i + 1 < len(severity_positions) else len(page_text)
        block_text = page_text[start_pos:end_pos].strip()

        times = TIME_PATTERN.findall(block_text)
        time_values = [f"{t[0]} {t[1]}" for t in times]

        content = f"Severity: {severity}\n\n{block_text}"

        row_data = {
            "severity": severity,
            "time_thresholds": time_values,
            "raw_block": block_text
        }

        chunks.append({
            "chunk_type": "table_row",
            "severity": severity,
            "content": content,
            "raw_row_data": json.dumps(row_data, ensure_ascii=False)
        })

    return chunks


# ─────────────────────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────────────────────

def list_ingested_files() -> List[str]:
    """Return a sorted list of unique source_file values in the collection."""
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    files: set = set()
    for meta in (results.get("metadatas") or []):
        if meta and meta.get("source_file"):
            files.add(meta["source_file"])
    return sorted(files)


def delete_file_chunks(source_file: str) -> int:
    """Delete all chunks belonging to a given source file. Returns count deleted."""
    collection = get_collection()
    results = collection.get(
        where={"source_file": source_file},
        include=[]
    )
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def get_collection_stats() -> Dict[str, Any]:
    """Return total chunk count and per-file chunk counts."""
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []
    file_counts: Dict[str, int] = {}
    for meta in metadatas:
        if meta and meta.get("source_file"):
            fname = meta["source_file"]
            file_counts[fname] = file_counts.get(fname, 0) + 1
    return {
        "total_chunks": len(metadatas),
        "file_counts": file_counts,
    }


def ingest_pdf(pdf_path: str, source_name: Optional[str] = None, force_reingest: bool = False):

    source_file = source_name or os.path.basename(pdf_path)
    collection = get_collection()

    # If not force_reingest, check if already ingested
    if not force_reingest:
        existing = collection.get(
            where={"source_file": source_file},
            include=[]
        )
        if existing.get("ids"):
            return {"status": "already_ingested", "chunks_added": 0}

    # If force_reingest, delete existing chunks first
    if force_reingest:
        delete_file_chunks(source_file)

    doc = fitz.open(pdf_path)

    chunks = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")

        if not text.strip():
            continue

        # Always create section chunk
        chunks.append({
            "chunk_type": "section",
            "content": text,
            "page": page_idx + 1,
            "source_file": source_file,
            "severity": ""
        })

        # Try escalation parser
        escalation_chunks = parse_escalation_blocks(text)

        for esc in escalation_chunks:
            esc["page"] = page_idx + 1
            esc["source_file"] = source_file
            chunks.append(esc)

    doc.close()

    if not chunks:
        return {"status": "empty", "chunks_added": 0}

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        emb = embed_text(chunk["content"])
        if not emb:
            continue

        chunk_id = f"{source_file}::{idx}"

        ids.append(chunk_id)
        embeddings.append(emb)
        documents.append(chunk["content"])

        metadatas.append({
            "source_file": source_file,
            "page": chunk["page"],
            "chunk_type": chunk["chunk_type"],
            "severity": chunk.get("severity", ""),
            "raw_row_data": chunk.get("raw_row_data", "")
        })

    if not ids:
        return {"status": "error_no_embeddings", "chunks_added": 0}

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return {
        "status": "ingested",
        "chunks_added": len(ids)
    }
