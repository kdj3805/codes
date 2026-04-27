"""
ingest.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enterprise Multi-Document Ingestion Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles:
  - PDF text + table extraction via PyMuPDF
  - Image extraction + OCR via EasyOCR
  - AI image description via Groq
  - Section-level chunking (preserving original logic)
  - Embedding via Ollama nomic-embed-text
  - Persistent storage in ChromaDB
  - Deduplication: skips files already in the vector store
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import io
import re
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import fitz                      # PyMuPDF
import requests
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from groq import Groq
import easyocr

load_dotenv()

# Disable ChromaDB telemetry to avoid KeyError race condition in Streamlit reruns
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL:   str = "llama-3.1-8b-instant"

OLLAMA_URL:   str = "http://localhost:11434/api/embeddings"
EMBED_MODEL:  str = "nomic-embed-text"

# ChromaDB will persist data here across sessions
CHROMA_PERSIST_DIR: str = "./chroma_store"
CHROMA_COLLECTION:  str = "enterprise_docs"

IMAGE_SYSTEM_PROMPT: str = """You are an enterprise compliance assistant.
Describe the UI screenshot using ONLY the OCR text provided.
Rules:
- Identify screen type
- List visible fields or options
- Explain policy purpose
- Do not guess
- If unclear, say so
- 3-5 sentences max"""

SECTION_HEADING_PATTERNS: List[str] = [
    r'^Best\s+Practice\s*#?\d+',
    r'^Introduction\s*:?\s*$',
    r'^Our\s+Recommendations?\s*:?\s*$',
    r'^How\s+MaaS360\s+Helps?\s*:?\s*$',
    r'^The\s+Options?\s*:?\s*$',
    r'^Chapter\s+\d+',
    r'^About\s+this\s+publication\s*$',
    r'^Conclusion\s*:?\s*$',
    r'^Summary\s*:?\s*$',
    r'^Overview\s*:?\s*$',
]

# Lazily initialised — EasyOCR takes a few seconds to boot
_OCR_READER: Optional[easyocr.Reader] = None


def _ocr_reader() -> easyocr.Reader:
    global _OCR_READER
    if _OCR_READER is None:
        _OCR_READER = easyocr.Reader(["en"], gpu=False)
    return _OCR_READER


# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    """Open (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def is_already_ingested(collection: chromadb.Collection, source_file: str) -> bool:
    """Return True if at least one chunk for source_file exists in the DB."""
    result = collection.get(
        where={"source_file": source_file},
        limit=1,
        include=[]          # IDs only — fastest check
    )
    return len(result["ids"]) > 0


def delete_file_chunks(source_file: str) -> int:
    """Remove all chunks for source_file. Returns count deleted."""
    collection = get_collection()
    existing = collection.get(where={"source_file": source_file}, include=[])
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_ingested_files() -> List[str]:
    """Return sorted list of unique source_file names in the DB."""
    collection = get_collection()
    all_meta = collection.get(include=["metadatas"])
    files: set = set()
    for meta in (all_meta.get("metadatas") or []):
        if meta and "source_file" in meta:
            files.add(meta["source_file"])
    return sorted(files)


def get_collection_stats() -> Dict[str, Any]:
    """Return total chunk count, per-file counts, chunk-type breakdown."""
    collection = get_collection()
    total = collection.count()
    all_meta = collection.get(include=["metadatas"])

    file_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}

    for meta in (all_meta.get("metadatas") or []):
        if not meta:
            continue
        fname = meta.get("source_file", "unknown")
        ctype = meta.get("chunk_type", "section")
        file_counts[fname] = file_counts.get(fname, 0) + 1
        type_counts[ctype] = type_counts.get(ctype, 0) + 1

    return {
        "total_chunks": total,
        "file_counts":  file_counts,
        "type_counts":  type_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    """Embed a string via Ollama. Returns empty list on failure."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        return resp.json()["embedding"]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES  (identical logic to original single-file app)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_image(img_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if min(img.size) < 300:
            scale = 300 / min(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = img.convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return img_bytes


def _ocr_image(img_bytes: bytes) -> str:
    processed = _preprocess_image(img_bytes)
    result = _ocr_reader().readtext(processed, detail=0, paragraph=True)
    return "\n".join(result).strip() if result else ""


def _is_logo_image(ocr_text: str) -> bool:
    text = ocr_text.lower().strip()
    return len(text) < 20 and ("ibm" in text or "maas360" in text)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION  (logic preserved exactly from original)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all text blocks and markdown tables from the PDF.
    Returns items sorted by (page, y0) for correct reading order.
    """
    doc  = fitz.open(pdf_path)
    items: List[Dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # Text blocks
        for b in page.get_text("blocks"):
            text = b[4].strip()
            if len(text) < 5:
                continue
            items.append({
                "page":    page_idx,
                "type":    "text",
                "content": text,
                "y0":      float(b[1])
            })

        # Tables → markdown
        for table in page.find_tables():
            md = table.to_markdown()
            if md.strip():
                y0 = float(table.bbox[1]) if hasattr(table, "bbox") else 0.0
                items.append({
                    "page":    page_idx,
                    "type":    "table",
                    "content": md,
                    "y0":      y0
                })

    doc.close()
    return sorted(items, key=lambda x: (x["page"], x["y0"]))


def extract_images(pdf_path: str, groq_client: Groq) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extract all images, run OCR, generate Groq descriptions.
    Deduplicates repeated logo images.
    Returns dict keyed by page index.
    """
    doc         = fitz.open(pdf_path)
    page_images: Dict[int, List[Dict[str, Any]]] = {}
    seen_logos: set = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_images[page_idx] = []

        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                ocr  = _ocr_image(base["image"])
            except Exception:
                continue

            if len(ocr) < 3:
                continue

            # Skip duplicate logos
            if _is_logo_image(ocr):
                h = hashlib.md5(ocr.encode()).hexdigest()
                if h in seen_logos:
                    continue
                seen_logos.add(h)

            # AI description
            try:
                res  = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    temperature=0.1,
                    max_tokens=400,
                    messages=[
                        {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                        {"role": "user",   "content": ocr}
                    ]
                )
                desc = res.choices[0].message.content.strip()
                if len(desc) < 10:
                    desc = f"Image contains: {ocr[:80]}"
            except Exception:
                desc = f"Image contains: {ocr[:80]}"

            page_images[page_idx].append({
                "ocr":         ocr,
                "description": desc,
                "img_no":      i + 1
            })

    doc.close()
    return page_images


# ─────────────────────────────────────────────────────────────────────────────
# HEADING DETECTION  (logic preserved exactly from original)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_heading(heading: str) -> str:
    heading = re.sub(r'\s+', ' ', heading).strip()
    heading = re.sub(r'[:\.]$', '', heading)
    return heading


def _is_toc_block(text: str) -> bool:
    lines = text.splitlines()
    dotted = sum(1 for line in lines if re.search(r'\.{4,}', line))
    if dotted >= 2:
        return True
    if "table of contents" in text.lower():
        return True
    return False


def _is_heading(text: str) -> bool:
    text = text.strip()
    for pattern in SECTION_HEADING_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    # Generic: short line ending with colon that starts with a capital
    if len(text) < 100 and text.endswith(":") and text[0].isupper():
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING  (logic preserved from original; source_file injected per chunk)
# ─────────────────────────────────────────────────────────────────────────────

def build_chunks(
    items:       List[Dict[str, Any]],
    page_images: Dict[int, List[Dict[str, Any]]],
    source_file: str
) -> List[Dict[str, Any]]:
    """
    Section-level chunking that mirrors the original logic exactly.
    source_file is embedded in each chunk for multi-doc attribution.

    Images on the same page as a section are appended to that section's
    chunk content (as <imagedesc> blocks) — preserving original behaviour.
    """
    chunks: List[Dict[str, Any]] = []

    current_heading:  Optional[str]       = None
    current_content:  List[str]           = []
    current_page:     Optional[int]       = None
    section_id: int = 0

    def _flush_section() -> None:
        nonlocal section_id
        if not current_heading or not current_content:
            return

        section_text = "\n\n".join(current_content)

        # Append image descriptions for images on the same page
        if current_page is not None and current_page in page_images:
            for img in page_images[current_page]:
                section_text += (
                    f"\n\n<imagedesc>\n"
                    f"OCR: {img['ocr']}\n"
                    f"Description: {img['description']}\n"
                    f"</imagedesc>"
                )

        chunks.append({
            "chunk_id":   f"{source_file}::section_{section_id}",
            "chunk_type": "section",
            "heading":    _normalize_heading(current_heading),
            "content":    f"{_normalize_heading(current_heading)}\n\n{section_text}",
            "page":       (current_page or 0) + 1,
            "source_file": source_file
        })
        section_id += 1  # noqa: nonlocal incremented via closure

    # Walk items in reading order
    for item in sorted(items, key=lambda x: (x["page"], x["y0"])):
        content = item["content"].strip()
        page    = item["page"]

        # Drop TOC noise
        if _is_toc_block(content):
            continue

        if _is_heading(content):
            _flush_section()
            current_heading = content
            current_content = []
            current_page    = page
            continue

        # Accumulate into current section
        if current_heading is not None:
            current_content.append(content)

    _flush_section()   # flush final section
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INGESTION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path:      str,
    source_name:   Optional[str] = None,
    force_reingest: bool         = False
) -> Dict[str, Any]:
    """
    Ingest a single PDF into ChromaDB.

    Args:
        pdf_path:       Path to the PDF file on disk.
        source_name:    Human-readable filename stored as metadata.
                        Defaults to os.path.basename(pdf_path).
        force_reingest: If True, remove existing chunks and re-ingest.

    Returns:
        {"source_file": str, "status": str, "chunks_added": int}
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    source_file = source_name or os.path.basename(pdf_path)
    collection  = get_collection()

    # ── Skip duplicates ────────────────────────────────────────────────────
    if not force_reingest and is_already_ingested(collection, source_file):
        return {"source_file": source_file, "status": "skipped", "chunks_added": 0}

    # ── Remove stale chunks before re-ingesting ────────────────────────────
    if force_reingest:
        old_ids = collection.get(
            where={"source_file": source_file}, include=[]
        )["ids"]
        if old_ids:
            collection.delete(ids=old_ids)

    groq_client = Groq(api_key=GROQ_API_KEY)

    # ── Stage 1: Extract ──────────────────────────────────────────────────
    text_items  = extract_text_tables(pdf_path)
    page_images = extract_images(pdf_path, groq_client)

    # ── Stage 2: Chunk ────────────────────────────────────────────────────
    chunks = build_chunks(text_items, page_images, source_file)

    if not chunks:
        return {"source_file": source_file, "status": "empty", "chunks_added": 0}

    # ── Stage 3: Embed + Store ────────────────────────────────────────────
    ids:        List[str]              = []
    embeddings: List[List[float]]      = []
    documents:  List[str]              = []
    metadatas:  List[Dict[str, Any]]   = []

    for chunk in chunks:
        emb = embed_text(chunk["content"])
        if not emb:
            # Skip chunks that failed to embed rather than storing zero vectors
            continue

        ids.append(chunk["chunk_id"])
        embeddings.append(emb)
        documents.append(chunk["content"])
        metadatas.append({
            "source_file": chunk["source_file"],
            "page":        chunk["page"],
            "heading":     chunk.get("heading", ""),
            "chunk_type":  chunk.get("chunk_type", "section"),
        })

    if ids:
        # upsert is idempotent — safe to call even on re-ingest
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    return {
        "source_file":  source_file,
        "status":       "ingested",
        "chunks_added": len(ids)
    }


def ingest_multiple(
    pdf_paths:    List[str],
    source_names: Optional[List[str]] = None,
    force_reingest: bool              = False
) -> List[Dict[str, Any]]:
    """
    Ingest a list of PDFs, skipping already-ingested ones unless forced.

    Args:
        pdf_paths:     Paths to PDF files.
        source_names:  Display names aligned with pdf_paths (optional).
        force_reingest: Replace existing data when True.

    Returns list of result dicts (one per file).
    """
    results = []
    for idx, path in enumerate(pdf_paths):
        name   = source_names[idx] if source_names and idx < len(source_names) else None
        result = ingest_pdf(path, source_name=name, force_reingest=force_reingest)
        results.append(result)
    return results