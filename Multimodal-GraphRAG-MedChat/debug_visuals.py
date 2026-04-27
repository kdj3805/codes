# =================== diagnose_visuals_v2.py ===================
# Focused diagnostic — runs after the first diagnose_visuals.py
# specifically targets the image_filename fallback and checks
# whether caption chunks have image_filename in their payload.
#
# Run from your project root:
#   python diagnose_visuals_v2.py
# ==============================================================

from __future__ import annotations
from pathlib import Path
import json

BASE_DIR   = Path(__file__).parent
QDRANT_DIR = BASE_DIR / "vector_db" / "qdrant_store"
IMAGE_DIR  = BASE_DIR / "output" / "images"
CAP_DIR    = BASE_DIR / "output" / "caption_chunks"
COLLECTION = "medical_reviews"
SEP        = "─" * 62

def header(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def ok(m):     print(f"  ✅  {m}")
def fail(m):   print(f"  ❌  {m}")
def warn(m):   print(f"  ⚠️   {m}")
def info(m):   print(f"  ℹ️   {m}")

# ==============================================================
# CHECK 1 — What keys do caption chunks actually have in Qdrant?
# Print FULL payload of first 3 caption chunks, no filtering
# ==============================================================
header("CHECK 1 — Full payload of caption chunks in Qdrant")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    client = QdrantClient(path=str(QDRANT_DIR))

    cap_filter = Filter(must=[FieldCondition(
        key="content_type",
        match=MatchAny(any=["figure_caption", "table_caption"]),
    )])

    scroll_result = client.scroll(
        collection_name = COLLECTION,
        scroll_filter   = cap_filter,
        limit           = 5,
        with_payload    = True,
        with_vectors    = False,
    )

    points = scroll_result[0] if scroll_result else []
    info(f"Found {len(points)} caption points (showing all keys)")

    for pt in points:
        p = pt.payload or {}
        print(f"\n  ── chunk_id: {p.get('chunk_id','?')} ──")
        for k, v in p.items():
            val = str(v)[:80] if v else "(empty)"
            print(f"     {k:20s} : {val}")

except Exception as e:
    fail(f"Qdrant error: {e}")

# ==============================================================
# CHECK 2 — What is inside caption_chunks JSON files?
# These are the source-of-truth for what ingestion saved
# Compare JSON payload vs Qdrant payload
# ==============================================================
header("CHECK 2 — Caption chunk JSON files on disk")

cap_files = list(CAP_DIR.glob("*_caption_chunks.json")) if CAP_DIR.exists() else []

if not cap_files:
    fail(f"No caption chunk JSON files found in {CAP_DIR}")
    warn("This means ingestion either:")
    warn("  a) Did not save caption chunks to disk")
    warn("  b) CAP_DIR path is different — check cancer_ingestion.py")
else:
    ok(f"Found {len(cap_files)} caption chunk files:")
    total_caps = 0
    has_image_path   = 0
    has_image_fname  = 0
    missing_both     = 0

    for jf in cap_files:
        with open(jf, encoding="utf-8") as f:
            chunks = json.load(f)
        total_caps += len(chunks)
        info(f"  {jf.name}: {len(chunks)} chunks")

        # Show first chunk's full keys
        if chunks:
            c = chunks[0]
            print(f"    First chunk keys: {list(c.keys())}")
            print(f"    image_path  : {c.get('image_path', 'MISSING')}")
            print(f"    image_filename: {c.get('image_filename','MISSING')}")

        for c in chunks:
            if c.get("image_path"):
                has_image_path += 1
            elif c.get("image_filename"):
                has_image_fname += 1
            else:
                missing_both += 1

    print()
    info(f"Total caption chunks in JSON: {total_caps}")
    info(f"  Have image_path     : {has_image_path}")
    info(f"  Have image_filename : {has_image_fname}")
    info(f"  Missing BOTH        : {missing_both}")

    if missing_both == total_caps:
        fail("ALL caption chunks are missing both image_path and image_filename")
        fail("Root cause: cancer_ingestion.py is not storing image metadata")
        fail("in caption chunks. The _make_caption_chunk() function needs")
        fail("to be checked — image_path must be included in metadata dict.")
    elif has_image_path > 0:
        ok(f"{has_image_path} chunks have image_path in JSON")
        warn("But Qdrant payload is missing it — ingestion pushed chunks")
        warn("before image_path was added to metadata, or a different")
        warn("version of ingestion was used to build Qdrant.")
        warn("FIX: Re-run cancer_ingestion.py with force_rebuild=True")

# ==============================================================
# CHECK 3 — Do the PNG files exist and are they readable?
# ==============================================================
header("CHECK 3 — PNG files in output/images/")

if not IMAGE_DIR.exists():
    fail(f"IMAGE_DIR does not exist: {IMAGE_DIR}")
else:
    pngs = list(IMAGE_DIR.glob("*.png"))
    ok(f"{len(pngs)} PNG files in {IMAGE_DIR}")

    # Try to open one to confirm it is a valid image
    if pngs:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(pngs[0])
            ok(f"Sample PNG readable: {pngs[0].name} ({img.size[0]}x{img.size[1]}px)")
        except Exception as e:
            warn(f"Could not open PNG with PIL: {e}")

        # List all PNG filenames
        info("All PNG files:")
        for p in sorted(pngs):
            info(f"  {p.name}")

# ==============================================================
# CHECK 4 — Simulate Strategy 2 fallback: image_filename → path
# Does IMAGE_DIR / image_filename resolve for a real filename?
# ==============================================================
header("CHECK 4 — Test image_filename → full path reconstruction")

# Try with the filename we KNOW exists from the first diagnostic
test_fname = "breast-cancer-review_picture_1.png"
test_full  = IMAGE_DIR / test_fname

if test_full.exists():
    ok(f"Strategy 2 reconstruction WORKS:")
    ok(f"  IMAGE_DIR / '{test_fname}'")
    ok(f"  → {test_full}")
else:
    fail(f"Strategy 2 reconstruction FAILS:")
    fail(f"  IMAGE_DIR = {IMAGE_DIR}")
    fail(f"  File: {test_fname}")
    fail(f"  Full path tried: {test_full}")
    warn("Check that IMAGE_DIR is correct in cancer_retrieval_v2_visual.py")
    warn(f"  Defined as: BASE_DIR / 'output' / 'images'")
    warn(f"  BASE_DIR   = {BASE_DIR}")

# ==============================================================
# CHECK 5 — Does any caption chunk in Qdrant have image_filename?
# ==============================================================
header("CHECK 5 — image_filename field in Qdrant caption payloads")

try:
    # Scroll more chunks to get a fuller picture
    scroll_all = client.scroll(
        collection_name = COLLECTION,
        scroll_filter   = cap_filter,
        limit           = 20,
        with_payload    = True,
        with_vectors    = False,
    )
    all_pts = scroll_all[0] if scroll_all else []

    has_img_path  = sum(1 for pt in all_pts if (pt.payload or {}).get("image_path"))
    has_img_fname = sum(1 for pt in all_pts if (pt.payload or {}).get("image_filename"))
    has_neither   = sum(1 for pt in all_pts
                        if not (pt.payload or {}).get("image_path")
                        and not (pt.payload or {}).get("image_filename"))

    info(f"Checked {len(all_pts)} caption chunks in Qdrant:")
    info(f"  Have image_path     : {has_img_path}")
    info(f"  Have image_filename : {has_img_fname}")
    info(f"  Have neither        : {has_neither}")

    if has_neither == len(all_pts):
        fail("CONFIRMED: No caption chunk in Qdrant has image_path OR image_filename")
        fail("")
        fail("THIS IS THE ROOT CAUSE OF MISSING IMAGES.")
        fail("")
        fail("The Strategy 2 fallback in _collect_figures() cannot work")
        fail("because image_filename is also absent from the Qdrant payload.")
        fail("")
        fail("═══════════════════════════════════════════════════════")
        fail("THE FIX: Re-run cancer_ingestion.py")
        fail("═══════════════════════════════════════════════════════")
        fail("")
        fail("Why: The 75 'caption' chunks in Qdrant are TEXT chunks")
        fail("that happen to contain figure caption sentences.")
        fail("They were stored with content_type=figure_caption by the")
        fail("text chunker but were NEVER built by CaptionChunkBuilder.")
        fail("CaptionChunkBuilder is the only code that extracts PNGs")
        fail("and stores image_path/image_filename in the metadata.")
        fail("")
        fail("After re-running ingestion you should see chunk_ids like:")
        fail("  breast-cancer-review_cap_0001  ← cap_ prefix")
        fail("Instead of:")
        fail("  melanoma-skin-cancer-review_0035  ← no cap_ = text chunk")
    elif has_img_fname > 0:
        ok(f"{has_img_fname} chunks have image_filename — Strategy 2 will work")
    elif has_img_path > 0:
        ok(f"{has_img_path} chunks have image_path — Strategy 1 will work")
        warn("But backslash normalisation may still be needed on Linux/Mac")

except Exception as e:
    fail(f"Qdrant scroll error: {e}")

# ==============================================================
# FINAL VERDICT
# ==============================================================
header("FINAL VERDICT")
print("""
  Run this checklist top to bottom:

  [ ] Check 1 shows caption chunk payload keys
      → Does any chunk have 'image_path' or 'image_filename'?
      → If NO → proceed to Fix below

  [ ] Check 2 shows caption_chunks JSON files
      → Do the JSON files have image_path or image_filename?
      → If YES in JSON but NO in Qdrant → re-run ingestion
      → If NO in JSON either → bug in cancer_ingestion.py

  [ ] Check 4 confirms Strategy 2 path reconstruction works
      → If FAILS → IMAGE_DIR path mismatch

  [ ] Check 5 gives the final root cause verdict

  ─────────────────────────────────────────────────────────────
  MOST LIKELY FIX BASED ON DIAGNOSTIC OUTPUT:

  python cancer_ingestion.py --force-rebuild
  (or however your ingestion accepts force_rebuild=True)

  After re-running, the caption chunks in Qdrant will have:
    chunk_id      : breast-cancer-review_cap_0001
    image_path    : output/images/breast-cancer-review_picture_1.png
    image_filename: breast-cancer-review_picture_1.png
    content_type  : figure_caption
  And images will start appearing in the app immediately.
""")