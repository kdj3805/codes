"""
Updated table chunking logic for row-level semantic extraction.

This module replaces the table handling in the RAG pipeline to:
1. Parse each table row individually
2. Extract semantic structure (severity, escalation level, triggers, actions)
3. Create natural-language chunks per row for better embedding and retrieval
4. Handle multi-page table continuations correctly
"""

import re
import math
from typing import List, Dict, Any, Optional


def _parse_markdown_table_rows(md: str) -> List[str]:
    """
    Extract raw rows from a markdown table, skipping the separator line.
    
    Returns:
        List of row strings (including header)
    """
    lines = [l for l in md.splitlines() if l.strip()]
    result = []
    for line in lines:
        # Skip the |---|---| separator
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
        return {"headers": [], "data_rows": []}
    
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


def build_chunks(
    items: List[Dict[str, Any]],
    page_images: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Updated chunking function with row-level table processing.
    
    Changes from original:
    1. Tables are now processed row-by-row using split_table_by_rows()
    2. Each table row becomes a semantic chunk with natural language description
    3. Escalation matrices are detected and structured accordingly
    4. Multi-page table continuations work because each row is independent
    5. Section chunks, image chunks remain unchanged
    
    Chunk types produced:
    - "section": heading + text paragraphs (unchanged)
    - "table_row": individual table row with semantic extraction (NEW)
    - "image": OCR + AI description (unchanged)
    - "orphan": pre-heading text (unchanged)
    """
    all_chunks: List[Dict[str, Any]] = []
    
    section_heading: Optional[str] = None
    section_page: Optional[int] = None
    section_text_blocks: List[str] = []
    
    section_id = 0
    table_id = 0
    image_chunk_id = 0
    orphan_id = 0
    
    def flush_section() -> None:
        nonlocal section_id
        if not section_heading or not section_text_blocks:
            return
        body = "\n\n".join(section_text_blocks)
        all_chunks.append({
            "chunk_id": f"section_{section_id}",
            "chunk_type": "section",
            "heading": section_heading.strip(),
            "parent_heading": "",
            "content": f"{section_heading.strip()}\n\n{body}",
            "page": (section_page or 0) + 1
        })
        section_id += 1
    
    items_sorted = sorted(items, key=lambda x: (x["page"], x["y0"]))
    
    for item in items_sorted:
        content = item["content"].strip()
        page = item["page"]
        
        # Skip TOC
        if item["type"] == "text" and _is_toc_block(content):
            continue
        
        # Heading detection
        if item["type"] == "text" and _is_heading(content):
            flush_section()
            section_heading = content
            section_page = page
            section_text_blocks = []
            continue
        
        # TABLE HANDLING - NOW ROW-LEVEL
        if item["type"] == "table":
            parent = section_heading.strip() if section_heading else f"Page {page + 1}"
            prefix = f"table_{table_id}"
            
            # Use row-level splitting instead of chunk-size splitting
            row_chunks = split_table_by_rows(item, parent, prefix)
            all_chunks.extend(row_chunks)
            table_id += len(row_chunks)
            continue
        
        # Normal text blocks
        if item["type"] == "text":
            if section_heading:
                section_text_blocks.append(content)
            else:
                # Orphan chunk
                all_chunks.append({
                    "chunk_id": f"orphan_{orphan_id}",
                    "chunk_type": "section",
                    "heading": f"Preamble — Page {page + 1}",
                    "parent_heading": "",
                    "content": content,
                    "page": page + 1
                })
                orphan_id += 1
    
    flush_section()
    
    # Image chunks (unchanged)
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


# Helper functions (need to be defined if not already in the main code)

def _is_toc_block(text: str) -> bool:
    """Detect if a text block is part of the Table of Contents."""
    lines = text.splitlines()
    dotted_lines = sum(1 for line in lines if re.search(r'\.{4,}', line))
    if dotted_lines >= 2:
        return True
    if "table of contents" in text.lower():
        return True
    return False


def _is_heading(text: str) -> bool:
    """Detect if a text block is a section heading."""
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
