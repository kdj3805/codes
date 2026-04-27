import os
import io
import re
import math
import hashlib
import fitz
import streamlit as st
import requests
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from groq import Groq
import easyocr
from rank_bm25 import BM25Okapi

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

OCR_READER = easyocr.Reader(["en"], gpu=False)

# Maximum rows per table sub-chunk to avoid embedding dilution
TABLE_ROWS_PER_CHUNK = 15

# RRF constant
RRF_K = 60

# Table chunk type boost for structured queries
TABLE_QUERY_BOOST = 2.0

IMAGE_SYSTEM_PROMPT = """You are an enterprise compliance assistant.
Describe the UI screenshot using ONLY the OCR text provided.
Rules:
- Identify screen type
- List visible fields or options
- Explain policy purpose
- Do not guess
- If unclear, say so
- 3-5 sentences max"""


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        return response.json()["embedding"]
    except Exception:
        return []


# ─────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────

def preprocess_image(img_bytes: bytes) -> bytes:
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


def ocr_image(img_bytes: bytes) -> str:
    img_bytes = preprocess_image(img_bytes)
    result = OCR_READER.readtext(img_bytes, detail=0, paragraph=True)
    return "\n".join(result).strip() if result else ""


def is_logo_image(ocr_text: str) -> bool:
    text = ocr_text.lower().strip()
    return len(text) < 20 and ("ibm" in text or "maas360" in text)


# ─────────────────────────────────────────────
# EXTRACTION  (unchanged per constraints)
# ─────────────────────────────────────────────

def extract_images(pdf_path: str, groq_client: Groq) -> Dict[int, List[Dict[str, Any]]]:
    doc = fitz.open(pdf_path)
    page_images: Dict[int, List[Dict[str, Any]]] = {}
    seen_logos: set = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_images[page_idx] = []

        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                ocr = ocr_image(base["image"])
            except Exception:
                continue

            if len(ocr) < 3:
                continue

            if is_logo_image(ocr):
                h = hashlib.md5(ocr.encode()).hexdigest()
                if h in seen_logos:
                    continue
                seen_logos.add(h)

            try:
                res = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    temperature=0.1,
                    max_tokens=400,
                    messages=[
                        {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                        {"role": "user", "content": ocr}
                    ]
                )
                desc = res.choices[0].message.content.strip()
                if len(desc) < 10:
                    desc = f"Image contains: {ocr[:80]}"
            except Exception:
                desc = f"Image contains: {ocr[:80]}"

            page_images[page_idx].append({
                "ocr": ocr,
                "description": desc,
                "img_no": i + 1
            })

    doc.close()
    return page_images


def extract_text_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text blocks and tables from every page, preserving position metadata."""
    doc = fitz.open(pdf_path)
    items = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        for b in page.get_text("blocks"):
            text = b[4].strip()
            if len(text) < 5:
                continue
            items.append({
                "page": page_idx,
                "type": "text",
                "content": text,
                "y0": float(b[1])
            })

        for table in page.find_tables():
            md = table.to_markdown()
            if md.strip():
                y0 = float(table.bbox[1]) if hasattr(table, 'bbox') else 0.0
                items.append({
                    "page": page_idx,
                    "type": "table",
                    "content": md,
                    "y0": y0,
                    # Store raw rows so we can split large tables later
                    "rows": _parse_markdown_table_rows(md)
                })

    doc.close()
    return sorted(items, key=lambda x: (x["page"], x["y0"]))


# ─────────────────────────────────────────────
# TABLE UTILITIES (ROW-LEVEL CHUNKING)
# ─────────────────────────────────────────────

def _parse_markdown_table_rows(md: str) -> List[str]:
    """Return raw lines of a markdown table, skipping the separator line."""
    lines = [l for l in md.splitlines() if l.strip()]
    result = []
    for line in lines:
        if re.match(r'^\s*\|[\s\-:]+\|\s*$', line):
            continue
        result.append(line)
    return result


def _parse_row_cells(row: str) -> List[str]:
    """
    Parse a markdown table row into individual cells.
    
    Args:
        row: Markdown row string like "| Cell1 | Cell2 | Cell3 |"
    
    Returns:
        List of cleaned cell values
    """
    return [c.strip() for c in row.strip().strip('|').split('|')]


def _extract_table_structure(md: str) -> Dict[str, Any]:
    """
    Parse markdown table and extract header + data rows.
    
    Returns:
        {
            "headers": List[str],
            "data_rows": List[List[str]]  # Each row is a list of cell values
        }
    """
    rows = _parse_markdown_table_rows(md)
    if not rows:
        return {" headers": [], "data_rows": []}
    
    headers = _parse_row_cells(rows[0])
    data_rows = [_parse_row_cells(row) for row in rows[1:]]
    
    return {
        "headers": headers,
        "data_rows": data_rows
    }


def _detect_escalation_matrix_headers(headers: List[str]) -> Optional[Dict[str, int]]:
    """
    Detect if this is an escalation matrix table and return column mappings.
    
    Returns:
        Dict mapping semantic field names to column indices, or None if not an escalation table.
        Example: {"severity": 0, "escalation": 1, "trigger": 2, "actions": 3}
    """
    headers_lower = [h.lower() for h in headers]
    
    # Look for escalation-specific patterns
    has_severity = any("severity" in h or "priority" in h for h in headers_lower)
    has_escalation = any("escalation" in h or "level" in h for h in headers_lower)
    has_trigger = any("trigger" in h or "condition" in h or "if" in h or "when" in h for h in headers_lower)
    has_action = any("action" in h or "step" in h or "response" in h for h in headers_lower)
    
    if not (has_severity or has_escalation or has_trigger):
        return None
    
    # Map column indices
    mapping = {}
    for i, h in enumerate(headers_lower):
        if "severity" in h or "priority" in h:
            mapping["severity"] = i
        elif "escalation" in h or ("level" in h and "escalation" not in mapping):
            mapping["escalation"] = i
        elif "trigger" in h or "condition" in h or ("if" in h and i > 0) or "when" in h:
            mapping["trigger"] = i
        elif "action" in h or "step" in h or "response" in h:
            mapping["actions"] = i
    
    return mapping if mapping else None


def _row_to_natural_language(
    row_cells: List[str],
    headers: List[str],
    parent_heading: str,
    escalation_mapping: Optional[Dict[str, int]] = None
) -> str:
    """
    Convert a table row into structured natural-language text for embedding.
    
    For escalation matrices, creates a semantic description like:
        "Severity: High
         Escalation Level: Third
         Trigger: If the ticket is not resolved in 48 hours
         Actions: Assignee is contacted for status; Customer Service Director..."
    
    For generic tables, creates key-value pairs:
        "Table row from {parent_heading}:
         Column1: Value1
         Column2: Value2"
    
    Args:
        row_cells: List of cell values for this row
        headers: List of column headers
        parent_heading: Section heading this table belongs to
        escalation_mapping: Optional column index mappings for escalation tables
    
    Returns:
        Natural language description of the row
    """
    if not row_cells or all(not c.strip() for c in row_cells):
        return ""
    
    # Escalation matrix handling
    if escalation_mapping:
        parts = [f"Escalation Matrix Entry (from {parent_heading}):"]
        
        if "severity" in escalation_mapping and escalation_mapping["severity"] < len(row_cells):
            severity = row_cells[escalation_mapping["severity"]].strip()
            if severity:
                parts.append(f"Severity: {severity}")
        
        if "escalation" in escalation_mapping and escalation_mapping["escalation"] < len(row_cells):
            escalation = row_cells[escalation_mapping["escalation"]].strip()
            if escalation:
                parts.append(f"Escalation Level: {escalation}")
        
        if "trigger" in escalation_mapping and escalation_mapping["trigger"] < len(row_cells):
            trigger = row_cells[escalation_mapping["trigger"]].strip()
            if trigger:
                parts.append(f"Trigger Condition: {trigger}")
        
        if "actions" in escalation_mapping and escalation_mapping["actions"] < len(row_cells):
            actions = row_cells[escalation_mapping["actions"]].strip()
            if actions:
                # Split multiple actions if they're semicolon or bullet separated
                action_items = re.split(r'[;\n•]', actions)
                action_items = [a.strip() for a in action_items if a.strip()]
                if len(action_items) > 1:
                    parts.append("Actions:")
                    for action in action_items:
                        parts.append(f"  - {action}")
                else:
                    parts.append(f"Actions: {actions}")
        
        return "\n".join(parts)
    
    # Generic table handling
    parts = [f"Table row from {parent_heading}:"]
    for i, cell in enumerate(row_cells):
        if cell.strip():
            header = headers[i] if i < len(headers) else f"Column {i+1}"
            parts.append(f"{header}: {cell.strip()}")
    
    return "\n".join(parts)


def split_table_by_rows(
    table_item: Dict[str, Any],
    parent_heading: str,
    chunk_id_prefix: str
) -> List[Dict[str, Any]]:
    """
    Split a markdown table into individual row-level chunks for better semantic embedding.
    
    Each row becomes its own chunk with:
    1. Original markdown row preserved (for display)
    2. Natural-language semantic extraction (for embedding quality)
    3. Clear tagging of severity/escalation/triggers (for escalation matrices)
    
    This approach solves the problem where large tables embedded as single blocks
    cause the LLM to mix rows and give incorrect answers.
    
    Args:
        table_item: Dict with "content" (markdown table), "page", "y0"
        parent_heading: The section heading this table belongs to
        chunk_id_prefix: Prefix for generating unique chunk IDs
    
    Returns:
        List of chunk dicts, one per table row
    """
    md_content = table_item.get("content", "")
    structure = _extract_table_structure(md_content)
    
    headers = structure["headers"]
    data_rows = structure["data_rows"]
    
    if not headers or not data_rows:
        return []
    
    # Detect if this is an escalation matrix
    escalation_mapping = _detect_escalation_matrix_headers(headers)
    
    chunks = []
    
    for row_idx, row_cells in enumerate(data_rows):
        # Skip empty rows
        if all(not c.strip() for c in row_cells):
            continue
        
        # Build natural language description
        nl_description = _row_to_natural_language(
            row_cells,
            headers,
            parent_heading,
            escalation_mapping
        )
        
        if not nl_description:
            continue
        
        # Build markdown snippet for this row (header + this data row)
        row_md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            "| " + " | ".join(row_cells) + " |"
        ]
        row_md = "\n".join(row_md_lines)
        
        # Combine both representations for maximum retrieval quality
        combined_content = f"""{nl_description}

--- Original Table Row (Page {table_item['page'] + 1}) ---
{row_md}"""
        
        # Determine chunk heading
        if escalation_mapping and "severity" in escalation_mapping:
            severity_idx = escalation_mapping["severity"]
            severity = row_cells[severity_idx].strip() if severity_idx < len(row_cells) else ""
            escalation_idx = escalation_mapping.get("escalation", -1)
            escalation = row_cells[escalation_idx].strip() if escalation_idx >= 0 and escalation_idx < len(row_cells) else ""
            
            if severity and escalation:
                heading = f"Escalation: {severity} Severity - {escalation}"
            elif severity:
                heading = f"Escalation: {severity} Severity"
            else:
                heading = f"Table Row {row_idx + 1} - {parent_heading}"
        else:
            heading = f"Table Row {row_idx + 1} - {parent_heading}"
        
        chunks.append({
            "chunk_id": f"{chunk_id_prefix}_row_{row_idx}",
            "chunk_type": "table_row",
            "heading": heading,
            "parent_heading": parent_heading,
            "content": combined_content,
            "page": table_item["page"] + 1,
            # Metadata for debugging/filtering
            "row_index": row_idx,
            "is_escalation_matrix": bool(escalation_mapping)
        })
    
    return chunks



# ─────────────────────────────────────────────
# HEADING DETECTION
# ─────────────────────────────────────────────

def normalize_heading(heading: str) -> str:
    heading = re.sub(r'\s+', ' ', heading).strip()
    heading = re.sub(r'[:\.]$', '', heading)
    return heading


def is_toc_block(text: str) -> bool:
    lines = text.splitlines()
    dotted_lines = sum(1 for line in lines if re.search(r'\.{4,}', line))
    if dotted_lines >= 2:
        return True
    if "table of contents" in text.lower():
        return True
    return False


def is_heading(text: str) -> bool:
    text = text.strip()
    heading_patterns = [
        r'^Best\s+Practice\s*#?\d+',
        r'^Introduction\s*:?\s*$',
        r'^Our\s+Recommendations?\s*:?\s*$',
        r'^How\s+MaaS360\s+Helps?\s*:?\s*$',
        r'^The\s+Options?\s*:?\s*$',
        r'^Chapter\s+\d+',
        r'^About\s+this\s+publication\s*$',
        r'^Conclusion\s*:?\s*$',
        r'^Summary\s*:?\s*$',
        r'^Overview\s*:?\s*$'
    ]
    for pattern in heading_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    if len(text) < 100 and text.endswith(":") and text[0].isupper():
        return True
    return False


# ─────────────────────────────────────────────
# CHUNKING  (rewritten)
# ─────────────────────────────────────────────

def build_chunks(
    items: List[Dict[str, Any]],
    page_images: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Produce four chunk types:
      - section   : heading + following text paragraphs (no tables/images mixed in)
      - table     : every markdown table gets its own chunk(s); table-only pages
                    are always preserved under a synthetic heading
      - image     : one chunk per image with OCR + AI description
      - orphan    : text blocks that precede the first heading

    Tables are split into sub-chunks when they exceed TABLE_ROWS_PER_CHUNK rows.
    Each table chunk contains both the markdown and a prose conversion to maximise
    both BM25 keyword recall and dense embedding recall.
    """
    all_chunks: List[Dict[str, Any]] = []

    section_heading: Optional[str] = None
    section_page: Optional[int] = None
    section_text_blocks: List[str] = []

    section_id = 0
    table_id = 0
    image_chunk_id = 0
    orphan_id = 0

    # Track which pages have at least one heading so we can detect table-only pages
    pages_with_heading: set = set()

    def flush_section() -> None:
        nonlocal section_id
        if not section_heading or not section_text_blocks:
            return
        body = "\n\n".join(section_text_blocks)
        all_chunks.append({
            "chunk_id": f"section_{section_id}",
            "chunk_type": "section",
            "heading": normalize_heading(section_heading),
            "parent_heading": "",
            "content": f"{normalize_heading(section_heading)}\n\n{body}",
            "page": (section_page or 0) + 1
        })
        section_id += 1

    items_sorted = sorted(items, key=lambda x: (x["page"], x["y0"]))

    for item in items_sorted:
        content = item["content"].strip()
        page = item["page"]

        # ── skip TOC noise ──────────────────────────────────────────────
        if item["type"] == "text" and is_toc_block(content):
            continue

        # ── heading detection ───────────────────────────────────────────
        if item["type"] == "text" and is_heading(content):
            flush_section()
            section_heading = content
            section_page = page
            section_text_blocks = []
            pages_with_heading.add(page)
            continue

        # ── table items (ROW-LEVEL CHUNKING) ───────────────────────────
        if item["type"] == "table":
            # Use the current section heading as parent; fall back to a
            # synthetic label so table-only pages are never orphaned.
            parent = (
                normalize_heading(section_heading)
                if section_heading
                else f"Page {page + 1}"
            )

            prefix = f"table_{table_id}"
            # NEW: Use row-level splitting for semantic extraction
            sub_chunks = split_table_by_rows(item, parent, prefix)
            all_chunks.extend(sub_chunks)
            table_id += len(sub_chunks)
            continue

        # ── normal text blocks ──────────────────────────────────────────
        if item["type"] == "text":
            if section_heading:
                section_text_blocks.append(content)
            else:
                # Preserve pre-heading text as orphan chunks so nothing is dropped
                all_chunks.append({
                    "chunk_id": f"orphan_{orphan_id}",
                    "chunk_type": "section",
                    "heading": f"Preamble — Page {page + 1}",
                    "parent_heading": "",
                    "content": content,
                    "page": page + 1
                })
                orphan_id += 1

    # Flush the final section
    flush_section()

    # ── image chunks ────────────────────────────────────────────────────
    for page_idx, imgs in page_images.items():
        for img in imgs:
            image_content = (
                f"<imagedesc>\n"
                f"Page {page_idx + 1}, Image {img['img_no']}\n\n"
                f"OCR Text:\n{img['ocr']}\n\n"
                f"Description:\n{img['description']}\n"
                f"</imagedesc>"
            )
            all_chunks.append({
                "chunk_id": f"image_{image_chunk_id}",
                "chunk_type": "image",
                "heading": f"Image — Page {page_idx + 1}",
                "parent_heading": "",
                "content": image_content,
                "page": page_idx + 1
            })
            image_chunk_id += 1

    return all_chunks


# ─────────────────────────────────────────────
# TOKENISATION
# ─────────────────────────────────────────────

def tokenize_text(text: str) -> List[str]:
    """
    Use re.findall(r'\w+') for BM25 tokenisation.
    This correctly handles table cell content, pipes, numbers, and
    hyphenated words that .split() misses.
    """
    return re.findall(r'\w+', text.lower())


# ─────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────
# HEADING BOOST
# ─────────────────────────────────────────────

def heading_match_boost(query: str, chunk_heading: str, chunk_type: str) -> float:
    """
    Boost score when:
    - Query mentions a numbered best practice that matches the chunk heading
    - Query is table-oriented and the chunk is a table
    - Query keywords overlap with heading
    """
    query_lower = query.lower()
    heading_lower = chunk_heading.lower()
    boost = 0.0

    # Numbered best-practice match
    match = re.search(r'best\s+practice\s*#?(\d+)', query_lower, re.IGNORECASE)
    if match:
        query_num = match.group(1)
        if query_num in heading_lower and "best practice" in heading_lower:
            boost += 10.0

    # Generic keyword overlap with heading
    keywords = ["introduction", "recommendations", "maas360", "conclusion", "overview"]
    for keyword in keywords:
        if keyword in query_lower and keyword in heading_lower:
            boost += 5.0

    # Table-oriented query boost (works for both 'table' and 'table_row')
    table_keywords = [
        "table", "role", "responsibility", "severity", "escalation",
        "response time", "contact", "matrix", "types", "compare",
        "difference", "list", "columns", "resolved", "hours", "ticket"
    ]
    if chunk_type in ["table", "table_row"] and any(kw in query_lower for kw in table_keywords):
        boost += TABLE_QUERY_BOOST

    # General token overlap
    query_tokens = set(tokenize_text(query))
    heading_tokens = set(tokenize_text(chunk_heading))
    if query_tokens:
        overlap = len(query_tokens & heading_tokens) / len(query_tokens)
        if overlap > 0.5:
            boost += 3.0

    return boost


# ─────────────────────────────────────────────
# RRF
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_ranks: List[Tuple[int, float]],
    bm25_ranks: List[Tuple[int, float]],
    k: int = RRF_K
) -> List[Tuple[int, float]]:
    rrf_scores: Dict[int, float] = {}
    for rank, (chunk_idx, _) in enumerate(dense_ranks):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk_idx, _) in enumerate(bm25_ranks):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# HYBRID RETRIEVAL
# ─────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval with Dense + BM25 + RRF.

    Key improvements over the original:
    1. BM25 uses re.findall(r'\\w+') tokenisation — handles table cell text.
    2. Heading + type-aware boost applied after RRF so table chunks surface
       when the query is structured / comparison oriented.
    3. BM25 index is built fresh per query (fast at this scale) from all chunks
       including table chunks — previously tables were mixed into sections and
       their content was diluted.
    """
    query_embedding = embed_text(query)

    # Dense ranking
    dense_scores: List[Tuple[int, float]] = []
    for idx, chunk in enumerate(chunks):
        sim = cosine_similarity(query_embedding, chunk["embedding"])
        dense_scores.append((idx, sim))
    dense_scores.sort(key=lambda x: x[1], reverse=True)

    # BM25 ranking — uses corrected tokenisation
    tokenized_corpus = [tokenize_text(chunk["content"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize_text(query)
    raw_bm25 = bm25.get_scores(query_tokens)

    bm25_ranked = sorted(
        [(idx, float(score)) for idx, score in enumerate(raw_bm25)],
        key=lambda x: x[1], reverse=True
    )

    # RRF fusion
    hybrid_scores = reciprocal_rank_fusion(dense_scores, bm25_ranked, k=RRF_K)

    # Heading + type-aware boost
    boosted: List[Tuple[int, float]] = []
    for chunk_idx, rrf_score in hybrid_scores:
        boost = heading_match_boost(
            query,
            chunks[chunk_idx].get("heading", ""),
            chunks[chunk_idx].get("chunk_type", "section")
        )
        boosted.append((chunk_idx, rrf_score + boost))

    boosted.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk_idx, final_score in boosted[:top_k]:
        dense_score = next((s for i, s in dense_scores if i == chunk_idx), 0.0)
        bm25_score = float(raw_bm25[chunk_idx])
        rrf_score = next((s for i, s in hybrid_scores if i == chunk_idx), 0.0)
        boost = heading_match_boost(
            query,
            chunks[chunk_idx].get("heading", ""),
            chunks[chunk_idx].get("chunk_type", "section")
        )
        results.append({
            "chunk_id": chunks[chunk_idx]["chunk_id"],
            "chunk_type": chunks[chunk_idx].get("chunk_type", "section"),
            "heading": chunks[chunk_idx].get("heading", ""),
            "content": chunks[chunk_idx]["content"],
            "page": chunks[chunk_idx].get("page"),
            "dense_score": round(dense_score, 4),
            "bm25_score": round(bm25_score, 4),
            "hybrid_score": round(rrf_score, 6),
            "heading_boost": round(boost, 4),
            "final_score": round(final_score, 6)
        })

    return results


# ─────────────────────────────────────────────
# CONTEXT ASSEMBLY
# ─────────────────────────────────────────────

def assemble_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Build the LLM context string.

    For table chunks the prose summary is already embedded in the content,
    so we pass the full content.  We label each block clearly so the LLM
    knows what type of content it is reading.
    """
    parts = []
    for c in retrieved_chunks:
        type_label = c.get("chunk_type", "section").upper()
        header = f"[{type_label} | {c['heading']} | Page {c.get('page', '?')}]"
        parts.append(f"{header}\n{c['content']}")
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────

def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    all_headings: List[str]
) -> str:
    groq_client = Groq(api_key=GROQ_API_KEY)
    query_lower = query.lower()

    # Hard-coded TOC shortcut — avoids false "not found" on contents queries
    if re.search(r'\b(main\s+sections?|all\s+sections?|headings?|topics?|table\s+of\s+contents)\b', query_lower):
        heading_list = "\n".join([f"- {h}" for h in all_headings])
        return f"The document contains the following sections:\n\n{heading_list}"

    context = assemble_context(retrieved_chunks)

    # Check for table_row chunks (escalation matrices or structured tables)
    has_table_context = any(c.get("chunk_type") in ["table", "table_row"] for c in retrieved_chunks)

    table_instruction = ""
    if has_table_context:
        table_instruction = (
            "\nThe context contains structured table data (possibly individual table rows). "
            "Present table-based answers as clear bullet points or structured prose. "
            "Do NOT output raw markdown table syntax. "
            "Each row represents a distinct entry. DO NOT mix or combine information from different rows. "
            "If multiple rows are provided, list them separately."
        )

    prompt = f"""You are an enterprise document assistant specialising in structured policy documents.

Use ONLY the provided context to answer the question.
{table_instruction}

Rules:
- If the answer is present in the context, provide it fully and accurately.
- Never truncate mid-sentence. If the answer is long, summarise the tail without losing detail.
- If the context contains table row data with the answer, extract and present the specific row(s) accurately.
- DO NOT mix information from different table rows. Each row is a separate entity.
- Do NOT say "Information not found in document" if any part of the answer appears in the context.
- If the answer truly does not appear anywhere, say exactly: "Information not found in document."

Context:
{context}

Question:
{query}

Answer:"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

#st.set_page_config(page_title="Enterprise Hybrid RAG", layout="wide")
st.title("RAG Pipeline")

st.markdown("""
**Architecture**
- `section` chunks — heading + text paragraphs
- `table_row` chunks — EACH TABLE ROW gets its own chunk with semantic extraction (escalation matrices automatically detected)
- `image` chunks — OCR + Groq description
- Hybrid retrieval: Dense (Ollama) + BM25 (`re.findall`) + RRF + type-aware boost

**Row-Level Table Chunking:**
Each table row is converted to natural language before embedding:
- Escalation matrices: "Severity: High | Escalation Level: Third | Trigger: If not resolved in 48 hours | Actions: ..."
- Generic tables: "Column1: Value1 | Column2: Value2"

This prevents the LLM from mixing rows in structured tables.
""")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "all_headings" not in st.session_state:
    st.session_state.all_headings = []

col_main, col_reset = st.columns([5, 1])
with col_reset:
    if st.button("Reset"):
        st.session_state.chunks = None
        st.session_state.all_headings = []
        st.rerun()

with col_main:
    pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf and st.session_state.chunks is None:
    with open("temp_rag.pdf", "wb") as f:
        f.write(pdf.read())

    groq_client = Groq(api_key=GROQ_API_KEY)

    with st.status("Processing PDF…", expanded=True) as status_box:

        st.write("Stage 1 — Extracting text, tables, images…")
        text_items = extract_text_tables("temp_rag.pdf")
        page_images = extract_images("temp_rag.pdf", groq_client)

        st.write("Stage 2 — Building granular chunks…")
        raw_chunks = build_chunks(text_items, page_images)

        n_section = sum(1 for c in raw_chunks if c["chunk_type"] == "section")
        n_table   = sum(1 for c in raw_chunks if c.get("chunk_type") in ["table", "table_row"])
        n_image   = sum(1 for c in raw_chunks if c["chunk_type"] == "image")
        st.write(f"Chunks built: {n_section} section | {n_table} table | {n_image} image")

        st.write("Stage 3 — Generating embeddings…")
        progress_bar = st.progress(0.0)
        embedded_chunks = []

        for idx, chunk in enumerate(raw_chunks):
            emb = embed_text(chunk["content"])
            ec = dict(chunk)
            ec["embedding"] = emb
            embedded_chunks.append(ec)
            progress_bar.progress((idx + 1) / len(raw_chunks))

        progress_bar.empty()
        st.session_state.chunks = embedded_chunks
        st.session_state.all_headings = [
            c["heading"] for c in raw_chunks if c["chunk_type"] == "section"
        ]
        status_box.update(label="Ingestion complete!", state="complete")

    try:
        os.remove("temp_rag.pdf")
    except Exception:
        pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Chunks", len(embedded_chunks))
    c2.metric("Section Chunks", n_section)
    c3.metric("Table Chunks", n_table)
    c4.metric("Image Chunks", n_image)

    with st.expander("All Detected Headings"):
        for h in st.session_state.all_headings:
            st.write(f"- {h}")

    with st.expander("Preview First 5 Chunks"):
        for chunk in embedded_chunks[:5]:
            type_icon = {"section": "📄", "table": "📊", "image": "🖼️"}.get(chunk["chunk_type"], "📄")
            st.markdown(f"{type_icon} **{chunk['chunk_id']}** — {chunk['heading']} (Page {chunk.get('page', '?')})")
            st.text_area("", chunk["content"][:600], height=140, key=f"prev_{chunk['chunk_id']}")

# ── Query interface ────────────────────────────────────────────────────────────

if st.session_state.chunks:
    st.divider()
    st.subheader("Ask Questions")

    query = st.text_input(
        "Enter your question:",
        placeholder="e.g. What are the roles and responsibilities? / What passcode types are supported?"
    )

    if query:
        with st.spinner("Retrieving relevant chunks…"):
            retrieved = hybrid_retrieve(query, st.session_state.chunks, top_k=5)

        st.subheader("Retrieved Chunks + Scores")

        type_icon_map = {"section": "📄", "table": "📊", "table_row": "📊", "image": "🖼️"}

        for r in retrieved:
            icon = type_icon_map.get(r.get("chunk_type", "section"), "📄")
            label = (
                f"{icon} [{r['chunk_type']}] {r['heading']} "
                f"| Final: {r['final_score']} | Page: {r.get('page', 'N/A')}"
            )
            with st.expander(label):
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Dense",        r["dense_score"])
                s2.metric("BM25",         r["bm25_score"])
                s3.metric("RRF Hybrid",   r["hybrid_score"])
                s4.metric("Boost",        r["heading_boost"])
                st.divider()
                st.text_area(
                    "Content",
                    r["content"],
                    height=220,
                    key=f"ret_{r['chunk_id']}"
                )

        with st.spinner("Generating answer…"):
            answer = generate_answer(
                query,
                retrieved,
                st.session_state.all_headings
            )

        st.subheader("Final Answer")
        st.success(answer)