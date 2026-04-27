# ============================================================
# Adaptive Chunking Pipeline with Docling HybridChunker
# - Preserves Table of Contents
# - Chunks text, tables, and <imagedesc> separately
# - RAG-ready output
# ============================================================

import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

# -----------------------------
# CONFIG
# -----------------------------
PDF_PATH = "D:\\trial\\data\\IBMMaaS360_Best_Practices_for_Policies.pdf"
OUTPUT_DIR = "chunked_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()

# -----------------------------
# DOC LING SETUP
# -----------------------------
pipeline_opts = PdfPipelineOptions()
pipeline_opts.do_table_structure = True
pipeline_opts.do_ocr = False
pipeline_opts.generate_picture_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
    }
)

# -----------------------------
# LOAD DOCUMENT
# -----------------------------
result = converter.convert(PDF_PATH)
doc = result.document

# -----------------------------
# EXTRACT RAW BLOCKS
# -----------------------------
text_blocks: List[Dict[str, Any]] = []
table_blocks: List[Dict[str, Any]] = []
image_blocks: List[Dict[str, Any]] = []

for item, level in doc.iterate_items():
    page = 0
    if item.prov and hasattr(item.prov[0], "page_no"):
        page = item.prov[0].page_no

    # ---------- TEXT ----------
    if hasattr(item, "text") and item.text:
        label = str(item.label)

        if any(k in label for k in ["TITLE", "SECTION_HEADER", "PARAGRAPH", "TEXT", "LIST_ITEM", "CAPTION"]):
            text_blocks.append({
                "type": "text",
                "page": page,
                "content": item.text.strip()
            })

    # ---------- TABLE ----------
    if hasattr(item, "label") and "TABLE" in str(item.label):
        try:
            table_md = item.export_to_markdown(doc=doc)
            if table_md.strip():
                table_blocks.append({
                    "type": "table",
                    "page": page,
                    "content": table_md.strip()
                })
        except Exception:
            pass

# -----------------------------
# LOAD IMAGE DESCRIPTIONS
# (from enriched_output.md)
# -----------------------------
IMAGE_DESC_PATH = "enriched_output.md"

if os.path.exists(IMAGE_DESC_PATH):
    with open(IMAGE_DESC_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    image_descs = re.findall(
        r"<imagedesc>(.*?)</imagedesc>",
        content,
        flags=re.DOTALL
    )

    for block in image_descs:
        image_blocks.append({
            "type": "image",
            "page": None,
            "content": block.strip()
        })

# -----------------------------
# TABLE OF CONTENTS PRESERVATION
# -----------------------------
toc_blocks = [
    b for b in text_blocks
    if "Table of Contents" in b["content"]
]

# -----------------------------
# HYBRID CHUNKING (TEXT)
# -----------------------------
hybrid_chunker = HybridChunker(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",  # Required tokenizer
    max_tokens=500
)

# HybridChunker expects a DoclingDocument, not a list of strings
text_chunks = list(hybrid_chunker.chunk(dl_doc=doc))

# -----------------------------
# FINAL CHUNKS ASSEMBLY
# -----------------------------
final_chunks = []

# --- TOC FIRST ---
for b in toc_blocks:
    final_chunks.append({
        "chunk_type": "toc",
        "content": b["content"]
    })

# --- TEXT CHUNKS ---
for idx, chunk in enumerate(text_chunks):
    # HybridChunker returns Chunk objects with .text attribute
    chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
    final_chunks.append({
        "chunk_type": "text",
        "chunk_id": f"text_{idx}",
        "content": chunk_text
    })

# --- TABLE CHUNKS ---
for idx, table in enumerate(table_blocks):
    final_chunks.append({
        "chunk_type": "table",
        "chunk_id": f"table_{idx}",
        "content": table["content"]
    })

# --- IMAGE CHUNKS (ISOLATED) ---
for idx, image in enumerate(image_blocks):
    final_chunks.append({
        "chunk_type": "image",
        "chunk_id": f"image_{idx}",
        "content": image["content"]
    })

# -----------------------------
# SAVE OUTPUT
# -----------------------------
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "adaptive_chunks.jsonl")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for chunk in final_chunks:
        f.write(f"{chunk}\n")

print("✅ Adaptive chunking completed successfully")
print(f"📦 Total chunks: {len(final_chunks)}")
print(f"📁 Output saved to: {OUTPUT_PATH}")
