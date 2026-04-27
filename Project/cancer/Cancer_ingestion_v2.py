from __future__ import annotations

import re
import io
import json
import math
import uuid
import hashlib
import unicodedata
from pathlib import Path
from typing import Optional

import fitz                                 # PyMuPDF
from PIL import Image
from tqdm import tqdm

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document as LCDoc
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Optional: Docling ─────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import PictureItem, TableItem
    _DOCLING = True
except Exception:
    _DOCLING = False
    print("ℹ️  Docling not installed — PyMuPDF fallback + regex captions")

# ── Optional: EasyOCR ─────────────────────────────────────
try:
    import easyocr
    _EASYOCR    = True
    _ocr_reader = None
except Exception:
    _EASYOCR = False

# ── Optional: NumPy ───────────────────────────────────────
try:
    import numpy as np
    _NUMPY = True
except Exception:
    _NUMPY = False

# ==========================================================
# ======================== CONFIG ==========================
# ==========================================================

BASE_DIR = Path(__file__).parent

PDF_DIR  = BASE_DIR / "pdfs"

MD_DIR      = BASE_DIR / "output" / "markdown"
CHUNK_DIR   = BASE_DIR / "output" / "chunks"
CAP_DIR     = BASE_DIR / "output" / "caption_chunks"
IMAGE_DIR   = BASE_DIR / "output" / "images"
EMBED_DIR   = BASE_DIR / "output" / "embedding_export"
LOG_DIR     = BASE_DIR / "output" / "logs"

QDRANT_DIR  = BASE_DIR / "vector_db" / "qdrantt_store"   # double-t

COLLECTION_REVIEWS  = "medical_reviews"
COLLECTION_PATIENTS = "patient_records"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM   = 768

MIN_TEXT_CHARS   = 80
MIN_IMAGE_PX     = 100
MAX_ASPECT_RATIO = 6.0
DPI_VECTOR       = 150
DPI_OCR          = 200

CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 150
MIN_CHUNK_CHARS = 120

HEADERS_TO_SPLIT  = [("#", "H1"), ("##", "H2"), ("###", "H3")]
DOCLING_IMG_SCALE = 2.0
EMBED_PREVIEW_DIMS = 8
UPSERT_BATCH = 256

# ── Source URLs ──────────────────────────────────────────
SOURCE_URLS: dict[str, str] = {
    "osteosarcoma-review":          "https://pubmed.ncbi.nlm.nih.gov/33795081/",
    "acute-leukemia-review":        "https://pubmed.ncbi.nlm.nih.gov/32093433/",
    "breast-cancer-review":         "https://pubmed.ncbi.nlm.nih.gov/31735550/",
    "lung-cancer-review":           "https://pubmed.ncbi.nlm.nih.gov/33207404/",
    "melanoma-skin-cancer-review":  "https://pubmed.ncbi.nlm.nih.gov/32887954/",
    "skin-cancer-types-review":     "https://pubmed.ncbi.nlm.nih.gov/30609218/",
}

def _get_source_url(source_name: str) -> str:
    return SOURCE_URLS.get(source_name, "")

# ==========================================================
# ================== DIRECTORY SETUP =======================
# ==========================================================

def ensure_dirs() -> None:
    for d in [MD_DIR, CHUNK_DIR, CAP_DIR, IMAGE_DIR, EMBED_DIR, LOG_DIR, QDRANT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ================== PAGE CLASSIFIER =======================
# ==========================================================

class PageProfile:
    """
    Classifies each PDF page for optimal extraction strategy.
    Strategies: 'text' | 'mixed' | 'vector' | 'scanned' | 'skip'
    """
    def __init__(self, page: fitz.Page) -> None:
        raw                = page.get_text().strip()
        self.text_chars    = len(raw)
        self.image_count   = len(page.get_images(full=True))
        self.drawing_count = len(page.get_drawings())
        self.is_text_rich    = self.text_chars >= MIN_TEXT_CHARS
        self.has_images      = self.image_count > 0
        self.is_vector_heavy = self.drawing_count > 25
        self.is_scanned      = not self.is_text_rich and not self.has_images
        self.is_mixed        = self.is_text_rich and (
            self.has_images or self.is_vector_heavy
        )

    def strategy(self) -> str:
        if self.text_chars < 10:  return "skip"
        if self.is_scanned:       return "scanned"
        if self.is_mixed:         return "mixed"
        if self.is_vector_heavy:  return "vector"
        return "text"

# ==========================================================
# ================== TEXT CLEANING =========================
# ==========================================================

def clean_text(text: str) -> str:
    """
    12-step medical PDF cleaner.
    Steps 1-5:  artifact repair (ligatures, math glyphs, broken words)
    Steps 6-12: noise removal (headers, footers, DOIs, page numbers)
    """
    if not text or not text.strip():
        return ""

    # 1. Mathematical bold digit repair
    math_digits = {
        '\U0001D7CE': '0', '\U0001D7CF': '1', '\U0001D7D0': '2',
        '\U0001D7D1': '3', '\U0001D7D2': '4', '\U0001D7D3': '5',
        '\U0001D7D4': '6', '\U0001D7D5': '7', '\U0001D7D6': '8',
        '\U0001D7D7': '9',
        '\U0001D7EC': '0', '\U0001D7ED': '1', '\U0001D7EE': '2',
        '\U0001D7EF': '3', '\U0001D7F0': '4', '\U0001D7F1': '5',
        '\U0001D7F2': '6', '\U0001D7F3': '7', '\U0001D7F4': '8',
        '\U0001D7F5': '9',
    }
    for weird, normal in math_digits.items():
        text = text.replace(weird, normal)

    # 2. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 3. Raw glyph codes from Docling font-decoding failures
    glyph_codes = {
        r'\s*/uniFB01\s*': 'fi',  r'\s*/uniFB02\s*': 'fl',
        r'\s*/uniFB00\s*': 'ff',  r'\s*/uniFB03\s*': 'ffi',
        r'\s*/uniFB04\s*': 'ffl', r'\s*/uniF642\s*': '%',
        r'\s*/C15\s*': '-',       r'\s*/C19\s*': 'e',
        r'\s*/C20\s*': 'c',       r'\s*/C211\s*': ' ',
    }
    for pattern, fixed in glyph_codes.items():
        text = re.sub(pattern, fixed, text)

    # 4. Unicode ligature characters
    for bad, good in {
        '\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff',
        '\ufb03': 'ffi', '\ufb04': 'ffl', '\ufb05': 'st',
    }.items():
        text = text.replace(bad, good)

    # 5. Broken word stitching — "ef fi cacy" → "efficacy"
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2\3', text)
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)(?=[a-zA-Z])',    r'\1\2',   text)
    text = re.sub(r'(?<=[a-zA-Z])(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)',   r'\1\2',   text)

    # 6. Remove Docling image placeholder tokens
    text = re.sub(r'', '', text, flags=re.IGNORECASE)

    # 7. Remove running journal headers/footers
    text = re.sub(
        r'(?m)^.{0,80}(?:guidelines?\s+version|©\s*\d{4}|page\s+\d+\s+of\s+\d+).{0,80}$\n?',
        '', text, flags=re.IGNORECASE
    )

    # 8. Remove "Author et al. / Journal" running headers
    text = re.sub(r'(?m)^[A-Z][a-z]+\s+et al\..{0,120}$\n?', '', text)

    # 9. Remove bare DOI / URL lines
    text = re.sub(r'(?m)^(?:https?://|doi:\s*|www\.)\S+\s*$\n?', '', text)

    # 10. Cut references section
    ref_match = re.compile(
        r'^#{0,3}\s*\*{0,2}'
        r'(references?|bibliography|works cited|literature cited)'
        r'\*{0,2}\s*$',
        re.IGNORECASE | re.MULTILINE,
    ).search(text)
    if ref_match:
        text = text[: ref_match.start()]

    # 11. Collapse excess whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 12. Remove isolated page-number lines
    text = re.sub(r'(?m)^\s*\d{1,3}\s*$\n?', '', text)

    return text.strip()

# ==========================================================
# ================== CONTENT TYPE TAGGING ==================
# ==========================================================

def detect_content_type(text: str) -> str:
    """5 semantic tags per chunk — enables content-type filtering in retrieval."""
    t = text.lower()
    if re.search(r'\bfig(?:ure)?\.?\s*\d+', t):  return "figure_caption"
    if re.search(r'\btable\s*\d+', t):            return "table_caption"
    if any(w in t for w in [
        "p-value", "p < 0.", "confidence interval", "mann-whitney",
        "chi-square", "statistical analysis", "multivariate", "hazard ratio",
    ]):                                            return "statistical_methods"
    if any(w in t for w in [
        "recommend", "guideline", "should be", "must be", "standard of care",
        "first-line", "indicated", "contraindicated", "treatment plan",
        "protocol", "regimen", "adjuvant", "neo-adjuvant",
    ]):                                            return "clinical_recommendation"
    if any(w in t for w in [
        "survival", "prognosis", "outcome", "recurrence", "5-year",
        "overall survival", "disease-free", "mortality", "remission", "relapse",
    ]):                                            return "prognosis_data"
    return "clinical_text"

# ==========================================================
# ================== CANCER TYPE DETECTION =================
# ==========================================================

def _detect_cancer_type(filename: str) -> str:
    fn = filename.lower()
    for keyword, label in [
        ("osteosarcoma", "osteosarcoma"), ("leukemia", "leukemia"),
        ("melanoma",     "melanoma"),     ("breast",   "breast"),
        ("lung",         "lung"),         ("skin",     "skin"),
    ]:
        if keyword in fn:
            return label
    return "general"

# ==========================================================
# ============== EMBEDDING EXPORTER ========================
# ==========================================================

class EmbeddingExporter:
    """
    Saves computed embedding vectors in human-readable JSONL format.
    Output: text_embeddings.jsonl, caption_embeddings.jsonl, embedding_summary.json
    """

    def __init__(self) -> None:
        self._text_path    = EMBED_DIR / "text_embeddings.jsonl"
        self._caption_path = EMBED_DIR / "caption_embeddings.jsonl"
        self._summary_path = EMBED_DIR / "embedding_summary.json"
        self._text_norms:    list[float] = []
        self._caption_norms: list[float] = []
        self._text_by_cancer:    dict[str, int] = {}
        self._caption_by_cancer: dict[str, int] = {}
        self._text_by_type:      dict[str, int] = {}
        self._caption_by_type:   dict[str, int] = {}

    def save_text_embeddings(self, docs: list[LCDoc], vectors: list[list[float]]) -> None:
        print(f"    💾 Saving {len(docs)} text embeddings → text_embeddings.jsonl")
        with open(self._text_path, "w", encoding="utf-8") as f:
            for doc, vec in zip(docs, vectors):
                m     = doc.metadata
                norm  = _vec_norm(vec)
                ct    = m.get("cancer_type", "?")
                ctype = m.get("content_type", "?")
                self._text_norms.append(norm)
                self._text_by_cancer[ct]    = self._text_by_cancer.get(ct, 0) + 1
                self._text_by_type[ctype]   = self._text_by_type.get(ctype, 0) + 1
                row = {
                    "chunk_id":          m.get("chunk_id", ""),
                    "source_file":       m.get("source_file", ""),
                    "cancer_type":       ct,
                    "content_type":      ctype,
                    "section":           m.get("section_hierarchy", "Body"),
                    "char_length":       m.get("char_length", len(doc.page_content)),
                    "content_preview":   doc.page_content[:200].replace("\n", " "),
                    "embedding_dim":     len(vec),
                    "embedding_norm":    round(norm, 6),
                    "embedding_preview": [round(x, 6) for x in vec[:EMBED_PREVIEW_DIMS]],
                    "full_vector":       [round(x, 8) for x in vec],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"    ✅ text_embeddings.jsonl written ({len(docs)} vectors)")

    def save_caption_embeddings(self, docs: list[LCDoc], vectors: list[list[float]]) -> None:
        print(f"    💾 Saving {len(docs)} caption embeddings → caption_embeddings.jsonl")
        with open(self._caption_path, "w", encoding="utf-8") as f:
            for doc, vec in zip(docs, vectors):
                m     = doc.metadata
                norm  = _vec_norm(vec)
                ct    = m.get("cancer_type", "?")
                ctype = m.get("content_type", "?")
                self._caption_norms.append(norm)
                self._caption_by_cancer[ct]    = self._caption_by_cancer.get(ct, 0) + 1
                self._caption_by_type[ctype]   = self._caption_by_type.get(ctype, 0) + 1
                row = {
                    "chunk_id":          m.get("chunk_id", ""),
                    "source_file":       m.get("source_file", ""),
                    "cancer_type":       ct,
                    "content_type":      ctype,
                    "asset_type":        m.get("asset_type", ""),
                    "image_filename":    m.get("image_filename", ""),
                    "image_path":        m.get("image_path", ""),
                    "extraction_tier":   m.get("extraction_tier", ""),
                    "char_length":       m.get("char_length", len(doc.page_content)),
                    "content_preview":   doc.page_content[:200].replace("\n", " "),
                    "embedding_dim":     len(vec),
                    "embedding_norm":    round(norm, 6),
                    "embedding_preview": [round(x, 6) for x in vec[:EMBED_PREVIEW_DIMS]],
                    "full_vector":       [round(x, 8) for x in vec],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"    ✅ caption_embeddings.jsonl written ({len(docs)} vectors)")

    def save_summary(self) -> None:
        def _stats(norms: list[float]) -> dict:
            if not norms:
                return {"count": 0}
            n    = len(norms)
            mean = sum(norms) / n
            var  = sum((x - mean) ** 2 for x in norms) / n
            return {
                "count":          n,
                "dim":            EMBEDDING_DIM,
                "norm_mean":      round(mean, 6),
                "norm_min":       round(min(norms), 6),
                "norm_max":       round(max(norms), 6),
                "norm_stddev":    round(math.sqrt(var), 6),
                "all_normalized": all(abs(x - 1.0) < 0.01 for x in norms),
            }

        summary = {
            "model":              EMBEDDING_MODEL,
            "embedding_dim":      EMBEDDING_DIM,
            "collection":         COLLECTION_REVIEWS,
            "text_embeddings": {
                **_stats(self._text_norms),
                "by_cancer_type":  self._text_by_cancer,
                "by_content_type": self._text_by_type,
            },
            "caption_embeddings": {
                **_stats(self._caption_norms),
                "by_cancer_type":  self._caption_by_cancer,
                "by_content_type": self._caption_by_type,
            },
            "output_files": {
                "text_embeddings":    str(EMBED_DIR / "text_embeddings.jsonl"),
                "caption_embeddings": str(EMBED_DIR / "caption_embeddings.jsonl"),
                "image_metadata":     str(IMAGE_DIR / "image_metadata.json"),
            },
        }
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"    ✅ embedding_summary.json saved")

        t_ok = all(abs(x - 1.0) < 0.01 for x in self._text_norms)    if self._text_norms    else True
        c_ok = all(abs(x - 1.0) < 0.01 for x in self._caption_norms) if self._caption_norms else True
        print(f"\n    📊 Embedding validation:")
        print(f"       Text    : {len(self._text_norms):4d} vectors | "
              f"norm [{min(self._text_norms, default=0):.4f}–"
              f"{max(self._text_norms, default=0):.4f}] | "
              f"normalized: {'✅' if t_ok else '❌ CHECK CONFIG'}")
        print(f"       Captions: {len(self._caption_norms):4d} vectors | "
              f"norm [{min(self._caption_norms, default=0):.4f}–"
              f"{max(self._caption_norms, default=0):.4f}] | "
              f"normalized: {'✅' if c_ok else '❌ CHECK CONFIG'}")


def _vec_norm(vec: list[float]) -> float:
    if _NUMPY:
        return float(np.linalg.norm(vec))
    return math.sqrt(sum(x * x for x in vec))

# ==========================================================
# ============== IMAGE DEDUPLICATION =======================
# ==========================================================

_img_hashes: set = set()

def _is_duplicate_image(b: bytes) -> bool:
    h = hashlib.md5(b).hexdigest()
    if h in _img_hashes:
        return True
    _img_hashes.add(h)
    return False


# ── Layer 1: Geometry ─────────────────────────────────────
NOISE_MIN_PX          = 100
NOISE_MAX_ASPECT      = 6.0
NOISE_SMALL_BADGE_PX  = 200    # small square badge threshold

# ── Layer 2: Color profile ────────────────────────────────
NOISE_BW_RATIO        = 0.70   # QR code / MDPI BW hexagon
NOISE_LOGO_BW_RATIO   = 0.55   # CC badge / MDPI outline logo
NOISE_GREEN_RATIO     = 0.45   # ORCID green icon
NOISE_TEAL_RATIO      = 0.35   # 4iD / ResearchGate teal
NOISE_ORANGE_RATIO    = 0.30   # CrossMark "check for updates"
NOISE_SEPIA_RATIO     = 0.35   # Elsevier brown tree logo
NOISE_SINGLE_HUE_SAT  = 0.70
NOISE_SINGLE_HUE_FRAC = 0.55

# ── Layer 3: Edge density ─────────────────────────────────
NOISE_QR_EDGE_RATIO   = 0.30
NOISE_FLAT_EDGE_RATIO = 0.05

# ── Layer 4: Perceptual hash ──────────────────────────────
PHASH_HAMMING_THRESH  = 10

# ── Layer 5 + 6: Context keyword patterns ────────────────
NOISE_PUBLISHER_PATTERNS = [
    # Publisher / journal branding
    r'\belsevier\b', r'\bmdpi\b', r'\bspringer\b', r'\bwiley\b',
    r'\btaylor\s*&\s*francis\b', r'\blippincott\b', r'\bwolters\b',
    r'\bcheck\s+for\s+updates\b', r'\bcrossmark\b', r'\bpublished\s+by\b',
    r'\ball\s+rights\s+reserved\b', r'\bopen\s+access\b',
    r'\bcc\s+by\b', r'\bcreative\s+commons\b',
    r'\borcid\b', r'\bissn\b', r'\beissn\b',
    r'©\s*\d{4}', r'\bdoi:\s*10\.',
    r'\bvolume\s+\d+\b', r'\bissue\s+\d+\b',
    r'\bjournal\s+of\b', r'\bcopyright\b',
    # Journal cover / magazine cover signals (NEW — catches CPT, SICOT-J)
    r'\bavailable\s+online\b', r'\bwww\.\w+\.\w+\b',
    r'\bpathogenesis\s+and\s+therapy\b',
    r'\bofficial\s+journal\b', r'\bsociety\s+of\b',
    r'\binternational\s+orthopaedics\b', r'\bsicot\b',
    r'\bcancer\s+pathogenesis\b', r'\boncology\s+journal\b',
]

NOISE_FIGURE_OVERRIDE_PATTERNS = [
    # Scientific content — KEEP even if publisher keywords present nearby
    r'\bfig(?:ure)?\.?\s*\d+', r'\btable\s*\d+',
    r'\bpanel\s*[a-d]\b', r'\bsupplementary\b',
    r'\bp\s*[<>=]\s*0\.\d+', r'\bhazard\s+ratio\b',
    r'\bkaplan.meier\b', r'\boverall\s+survival\b',
    r'\bflow\s+cytometry\b', r'\bimmunohistochem\b',
    r'\bscale\s+bar\b', r'\bmagnification\b',
    r'\bstaining\b', r'\bmicroscopy\b',
    r'\bcell\s+line\b', r'\bwestern\s+blot\b',
    r'\bhistology\b', r'\bpathology\b',
]

# Internal state
_phash_blocklist: list = []
_rejected_log:    list = []


# ── Perceptual hash helpers ───────────────────────────────

def _compute_phash(img: Image.Image, hash_size: int = 8) -> int:
    """dHash — difference hash. Returns 64-bit int."""
    grey   = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(grey.getdata())
    bits   = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left  = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits  = (bits << 1) | (1 if left > right else 0)
    return bits

def _phash_hamming(h1: int, h2: int) -> int:
    x = h1 ^ h2; count = 0
    while x: count += x & 1; x >>= 1
    return count

def _is_in_phash_blocklist(ph: int) -> bool:
    return any(_phash_hamming(ph, k) <= PHASH_HAMMING_THRESH for k in _phash_blocklist)

def _log_rejection(filename: str, reason: str, detail: str) -> None:
    _rejected_log.append({"filename": filename, "reason": reason, "detail": detail})

def save_rejection_log() -> None:
    log_path = LOG_DIR / "rejected_images.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(_rejected_log, f, indent=2, ensure_ascii=False)
    if _rejected_log:
        reasons: dict = {}
        for e in _rejected_log:
            r = e["reason"]; reasons[r] = reasons.get(r, 0) + 1
        print(f"\n    🚫 Rejected {len(_rejected_log)} noise images:")
        for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"       {r:<34} {count:3d}")
        print(f"    📋 Rejection log: {log_path}")


# ── Color analysis (Layer 2 + 3) ─────────────────────────

def _color_analysis(img: Image.Image) -> dict:
    """Pixel-level colour profile — no OpenCV required."""
    rgb   = img.convert("RGB")
    w, h  = rgb.size
    total = w * h
    if total == 0:
        return {"bw_ratio": 0, "green_ratio": 0, "teal_ratio": 0,
                "orange_ratio": 0, "sepia_ratio": 0,
                "dominant_hue_frac": 0, "edge_ratio": 0}

    pixels = list(rgb.getdata())
    BW_THRESH   = 30
    bw_count = green_count = 0
    hue_buckets = [0] * 36

    for r, g, b in pixels:
        lo, hi = min(r, g, b), max(r, g, b)
        if (hi - lo) < BW_THRESH and (hi < 50 or lo > 205):
            bw_count += 1
        if r < 120 and 160 <= g <= 230 and b < 120:
            green_count += 1
        delta = max(r, g, b) - min(r, g, b)
        if delta > 40 and max(r, g, b) > 0:
            mc = max(r, g, b)
            if mc == r:   hue = (60 * ((g - b) / delta)) % 360
            elif mc == g: hue = 60 * ((b - r) / delta) + 120
            else:         hue = 60 * ((r - g) / delta) + 240
            hue_buckets[int(hue / 10) % 36] += 1

    sat_total = sum(hue_buckets)
    if sat_total > total * 0.10:
        tb  = max(range(36), key=lambda i: hue_buckets[i])
        tc  = sum(hue_buckets[(tb + d) % 36] for d in [-1, 0, 1])
        dhf = tc / sat_total
    else:
        dhf = 0.0

    grey  = img.convert("L").resize((64, 64), Image.LANCZOS)
    gpix  = list(grey.getdata())
    gw = gh = 64; ec = 0; ET = 30
    for row in range(gh - 1):
        for col in range(gw - 1):
            idx = row * gw + col
            if (abs(int(gpix[idx]) - int(gpix[idx + 1])) > ET or
                    abs(int(gpix[idx]) - int(gpix[idx + gw])) > ET):
                ec += 1
    edge_ratio = ec / (gw * gh)

    teal_count   = sum(1 for r,g,b in pixels if r<100 and g>150 and b>150 and abs(int(g)-int(b))<40)
    orange_count = sum(1 for r,g,b in pixels if r>180 and 80<=g<=160 and b<80)
    sepia_count  = sum(1 for r,g,b in pixels if 100<=r<=210 and 60<=g<=150 and 20<=b<=110 and r>g>b and (r-b)>40)

    return {
        "bw_ratio":          bw_count     / total,
        "green_ratio":       green_count  / total,
        "teal_ratio":        teal_count   / total,
        "orange_ratio":      orange_count / total,
        "sepia_ratio":       sepia_count  / total,
        "dominant_hue_frac": dhf,
        "edge_ratio":        edge_ratio,
    }


# ── Context classifier (Layer 5) ─────────────────────────

def _classify_by_context(context_text: str) -> str:
    """Returns 'noise' | 'figure' | 'unknown'."""
    if not context_text:
        return "unknown"
    t = context_text.lower()
    if any(re.search(p, t) for p in NOISE_FIGURE_OVERRIDE_PATTERNS):
        return "figure"
    if any(re.search(p, t) for p in NOISE_PUBLISHER_PATTERNS):
        return "noise"
    return "unknown"


# ── Main noise gate ───────────────────────────────────────

def _is_noise_image(img: Image.Image, filename: str = "",
                    context_text: str = "") -> bool:
    """
    5-layer + journal-cover gate. Returns True → reject this image.

    Layer order:
      L5 context first (most reliable when Docling gives surrounding text)
      L1 geometry
      L2 colour profile (logos, badges, QR codes)
      L3 edge density (flat icons)
      L4 perceptual hash blocklist (cross-PDF duplicates)
    """
    w, h = img.size

    # ── Layer 5: text context ─────────────────────────────
    if context_text:
        ctx = _classify_by_context(context_text)
        if ctx == "figure":
            return False           # scientific override — always keep
        if ctx == "noise":
            _log_rejection(filename, "layer5_context", context_text[:80])
            try: _phash_blocklist.append(_compute_phash(img))
            except Exception: pass
            return True

    # ── Layer 1: geometry ─────────────────────────────────
    if w < NOISE_MIN_PX or h < NOISE_MIN_PX:
        _log_rejection(filename, "layer1_too_small", f"{w}×{h}"); return True
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > NOISE_MAX_ASPECT:
        _log_rejection(filename, "layer1_banner", f"aspect={aspect:.1f}"); return True
    if w <= NOISE_SMALL_BADGE_PX and h <= NOISE_SMALL_BADGE_PX and aspect < 1.6:
        _log_rejection(filename, "layer1_small_badge", f"{w}×{h}"); return True

    # ── Layer 2 + 3: colour + edges ───────────────────────
    try:
        ca = _color_analysis(img)
    except Exception:
        ca = {k: 0 for k in ["bw_ratio","green_ratio","teal_ratio",
                              "orange_ratio","sepia_ratio","dominant_hue_frac","edge_ratio"]}

    if ca["bw_ratio"] > NOISE_BW_RATIO and ca["edge_ratio"] > NOISE_QR_EDGE_RATIO:
        _log_rejection(filename, "layer2_qr_code", f"bw={ca['bw_ratio']:.2f} edge={ca['edge_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["green_ratio"] > NOISE_GREEN_RATIO:
        _log_rejection(filename, "layer2_orcid_green", f"green={ca['green_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["teal_ratio"] > NOISE_TEAL_RATIO:
        _log_rejection(filename, "layer2_teal_icon", f"teal={ca['teal_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["orange_ratio"] > NOISE_ORANGE_RATIO and w < 350 and h < 350:
        _log_rejection(filename, "layer2_orange_badge", f"orange={ca['orange_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["sepia_ratio"] > NOISE_SEPIA_RATIO and w < 500 and h < 400:
        _log_rejection(filename, "layer2_sepia_logo", f"sepia={ca['sepia_ratio']:.2f} size={w}×{h}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["bw_ratio"] > NOISE_LOGO_BW_RATIO and w < 500 and h < 350:
        _log_rejection(filename, "layer2_bw_logo", f"bw={ca['bw_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["dominant_hue_frac"] > NOISE_SINGLE_HUE_FRAC and w < 350 and h < 350:
        _log_rejection(filename, "layer2_single_hue_logo", f"dom_hue={ca['dominant_hue_frac']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["edge_ratio"] < NOISE_FLAT_EDGE_RATIO and w < 300 and h < 300:
        _log_rejection(filename, "layer3_flat_icon", f"edge={ca['edge_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True

    # ── Layer 6: journal / magazine cover detection ───────
    # CPT journal cover (Image 1) and SICOT-J (Image 3) are large, colourful
    # portrait images that pass all pixel checks. They are identified by:
    #   a) portrait aspect ratio close to A4 (0.65–0.80 w/h ratio)
    #   b) high colour saturation (dominant_hue_frac > 0.25)
    #   c) large size (both sides > 400px — rules out small logos)
    # These criteria together are only met by magazine/journal covers.
    # Real scientific figures are either landscape, square, or have low
    # dominant-hue fractions (they contain multiple colour channels).
    if (w > 400 and h > 400 and                    # large enough to be a cover
            0.55 < (w / h) < 0.82 and              # portrait A4-ish aspect
            ca["dominant_hue_frac"] > 0.28):        # strong single colour brand
        _log_rejection(filename, "layer6_journal_cover",
                       f"portrait_cover size={w}×{h} dom_hue={ca['dominant_hue_frac']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True

    # ── Layer 4: perceptual hash blocklist ────────────────
    try:
        ph = _compute_phash(img)
        if _is_in_phash_blocklist(ph):
            _log_rejection(filename, "layer4_phash_match", "near-duplicate of known noise")
            return True
    except Exception:
        pass

    return False    # passed all layers → keep

# ==========================================================
# ============== EASYOCR (scanned pages only) ==============
# ==========================================================

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        print("    🔍 Loading EasyOCR (one-time ~10s)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader

def _ocr_page(page: fitz.Page) -> str:
    if not _EASYOCR:
        return ""
    try:
        pil     = Image.open(io.BytesIO(page.get_pixmap(dpi=DPI_OCR).tobytes("png")))
        results = _get_ocr_reader().readtext(pil, detail=1, paragraph=True)
        return "\n".join(
            t.strip()
            for _, t, conf in sorted(results, key=lambda r: r[0][0][1])
            if conf >= 0.4 and t.strip()
        )
    except Exception:
        return ""

# ==========================================================
# =========== CAPTION CHUNK BUILDER ========================
# ==========================================================

class CaptionChunkBuilder:
    """
    Extracts image/table captions → embeddable LCDoc chunks.

    Two extraction paths:
      Docling (preferred)  — 4-tier caption logic on element iterator
      PyMuPDF (fallback)   — raster images + regex caption extraction

    Deduplication: captions are normalised and checked against
    self._seen_caps — same caption injected twice produces one chunk.
    """

    _CAPTION_RE = re.compile(
        r'((?:Supplementary\s+)?(?:Fig(?:ure)?\.?|Table)\s*\d+[a-zA-Z]?'
        r'[\.\-\u2013\s].{20,500}?)'
        r'(?=\n\n|\Z)',
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, source_name: str, cancer_type: str) -> None:
        self.source_name   = source_name
        self.cancer_type   = cancer_type
        self._seen_caps:   set        = set()
        self._master_meta: list[dict] = []

    # ── Docling path ───────────────────────────────────────

    def build_from_docling(
        self,
        conv_result,
        local_image_dir: Path,
    ) -> tuple[list[LCDoc], list[dict]]:
        """
        Iterates Docling's element list.
        For each PictureItem / TableItem:
          1. Saves PNG to disk
          2. Extracts caption with 4-tier logic
          3. Builds a LCDoc caption chunk
        Returns (caption_docs, image_metadata_list).
        """
        items        = list(conv_result.document.iterate_items())
        caption_docs = []
        pic_counter  = 0
        tbl_counter  = 0

        for i, (element, _level) in enumerate(items):
            is_picture = isinstance(element, PictureItem)
            is_table   = isinstance(element, TableItem)
            if not (is_picture or is_table):
                continue

            img = element.get_image(conv_result.document)
            if img is None:
                continue

            # Gather context text from surrounding elements (±3 positions)
            # Used by Layer 5 (publisher keywords) and Layer 6 (journal cover)
            context_parts = []
            for offset in [-3, -2, -1, 1, 2, 3]:
                ni = i + offset
                if 0 <= ni < len(items):
                    el = items[ni][0]
                    if hasattr(el, "text") and el.text:
                        context_parts.append(el.text.strip()[:120])
            context_text = " ".join(context_parts)

            # Full 6-layer noise filter — runs BEFORE saving to disk
            try:
                pil_check = img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img))
                if _is_noise_image(
                    pil_check.convert("RGB"),
                    filename    = f"{self.source_name}_element_{i}",
                    context_text = context_text,
                ):
                    continue
            except Exception:
                pass

            if is_picture:
                pic_counter += 1
                img_filename = f"{self.source_name}_picture_{pic_counter}.png"
            else:
                tbl_counter += 1
                img_filename = f"{self.source_name}_table_{tbl_counter}.png"

            img_path = local_image_dir / img_filename
            with open(img_path, "wb") as fp:
                img.save(fp, "PNG")

            caption_text, tier = self._extract_caption_docling(element, items, i)

            doc_chunk, meta = self._make_caption_chunk(
                caption_text, img_filename, img_path,
                "Picture" if is_picture else "Table", tier,
            )
            if doc_chunk is not None:
                caption_docs.append(doc_chunk)
                self._master_meta.append(meta)

        print(f"    📸 Docling: {tbl_counter} tables + {pic_counter} figures "
              f"→ {len(caption_docs)} caption chunks")
        return caption_docs, self._master_meta

    # ── PyMuPDF fallback path ──────────────────────────────

    def build_from_pymupdf(
        self,
        doc: fitz.Document,
        local_image_dir: Path,
    ) -> tuple[list[LCDoc], list[dict]]:
        caption_docs = []
        img_counter  = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            p    = PageProfile(page)

            if p.has_images:
                for ref in page.get_images(full=True):
                    try:
                        b = doc.extract_image(ref[0])["image"]
                        if _is_duplicate_image(b):
                            continue
                        pil = Image.open(io.BytesIO(b)).convert("RGB")
                        if _is_noise_image(pil):
                            continue
                        img_counter += 1
                        fp = local_image_dir / f"{self.source_name}_p{page_idx+1}_img{img_counter}.png"
                        pil.save(fp)
                    except Exception:
                        pass
            elif p.is_vector_heavy:
                try:
                    b = page.get_pixmap(dpi=DPI_VECTOR).tobytes("png")
                    if not _is_duplicate_image(b):
                        pil = Image.open(io.BytesIO(b))
                        if not _is_noise_image(pil):
                            img_counter += 1
                            fp = local_image_dir / f"{self.source_name}_p{page_idx+1}_vector.png"
                            pil.save(fp)
                except Exception:
                    pass

        seen: set = set()
        for page_idx in range(len(doc)):
            for match in self._CAPTION_RE.findall(doc[page_idx].get_text()):
                cleaned = re.sub(r'\s+', ' ', match.strip())
                if len(cleaned) < 30 or cleaned in seen:
                    continue
                seen.add(cleaned)
                asset_type = "Table" if re.match(r'table', cleaned, re.IGNORECASE) else "Picture"
                img_fname  = f"{self.source_name}_regex_cap_{len(caption_docs)+1:04d}.png"
                doc_chunk, meta = self._make_caption_chunk(
                    cleaned, img_fname, local_image_dir / img_fname, asset_type, "pymupdf_regex"
                )
                if doc_chunk is not None:
                    caption_docs.append(doc_chunk)
                    self._master_meta.append(meta)

        print(f"    📸 PyMuPDF fallback: {img_counter} images | {len(caption_docs)} caption chunks")
        return caption_docs, self._master_meta

    # ── 4-tier caption extraction ──────────────────────────

    def _extract_caption_docling(self, element, items, idx) -> tuple[str, str]:
        """
        Tier 1 — element.captions (Docling native structural linking)
        Tier 2 — items[idx+1] starts with "Fig" or "Table"
        Tier 3 — items[idx-1] starts with "Fig" or "Table"
        Tier 4 — Context snippet from next paragraph (300 chars)
        """
        captions = getattr(element, "captions", [])
        parts    = [c.text for c in captions if hasattr(c, "text") and c.text]
        if parts:
            return " ".join(parts).strip(), "docling_native"

        if idx + 1 < len(items):
            nxt = items[idx + 1][0]
            if hasattr(nxt, "text") and nxt.text:
                t = nxt.text.strip()
                if re.match(r'(?i)(fig|table)', t):
                    return t, "sliding_window_next"

        if idx - 1 >= 0:
            prv = items[idx - 1][0]
            if hasattr(prv, "text") and prv.text:
                t = prv.text.strip()
                if re.match(r'(?i)(fig|table)', t):
                    return t, "sliding_window_prev"

        if idx + 1 < len(items):
            nxt = items[idx + 1][0]
            if hasattr(nxt, "text") and nxt.text:
                return f"Context: {nxt.text.strip()[:300]}...", "context_fallback"

        return "Visual asset extracted from document.", "no_caption"

    # ── LCDoc builder ──────────────────────────────────────

    def _make_caption_chunk(
        self,
        caption_text: str,
        img_filename: str,
        img_path: Path,
        asset_type: str,
        tier: str,
    ) -> tuple[Optional[LCDoc], dict]:
        norm = re.sub(r'\s+', ' ', caption_text.lower().strip())
        if norm in self._seen_caps:
            return None, {}
        self._seen_caps.add(norm)

        try:
            rel_path = str(img_path.relative_to(BASE_DIR))
        except ValueError:
            rel_path = str(img_path)

        cap_idx      = len(self._seen_caps)
        chunk_id     = f"{self.source_name}_cap_{cap_idx:04d}"
        content_type = "table_caption" if asset_type == "Table" else "figure_caption"

        metadata = {
            "chunk_id":        chunk_id,
            "source_file":     self.source_name,
            "source_url":      _get_source_url(self.source_name),
            "collection":      COLLECTION_REVIEWS,
            "cancer_type":     self.cancer_type,
            "content_type":    content_type,
            "asset_type":      asset_type,
            "image_filename":  img_filename,
            "image_path":      rel_path,
            "extraction_tier": tier,
            "char_length":     len(caption_text),
        }
        lc_doc = LCDoc(page_content=caption_text, metadata=metadata)
        return lc_doc, {**metadata, "caption": caption_text}

# ==========================================================
# ================== TABLE DATA EXTRACTOR ==================
# ==========================================================

def extract_table_data(doc: fitz.Document) -> str:
    seen: set = set(); table_blocks: list = []
    for page_num in range(len(doc)):
        for block in doc[page_num].get_text("blocks", sort=True):
            if len(block) < 5: continue
            raw = block[4].strip()
            if len(re.findall(r'\d+\.?\d*', raw)) >= 3 and len(raw) > 40:
                cleaned = re.sub(r'\s+', ' ', raw)
                if cleaned not in seen:
                    seen.add(cleaned); table_blocks.append(cleaned)
    return (
        "\n\n## Key Numerical Data\n\n" + "\n\n".join(table_blocks[:25])
    ) if table_blocks else ""

# ==========================================================
# ================== TEXT EXTRACTION =======================
# ==========================================================

def _pymupdf_extract(doc: fitz.Document) -> str:
    parts: list = []; scanned = 0; ocr_done = 0
    for i in range(len(doc)):
        page  = doc[i]; strat = PageProfile(page).strategy()
        if strat == "skip": continue
        elif strat == "scanned":
            scanned += 1
            if _EASYOCR:
                t = _ocr_page(page)
                parts.append(f"\n[OCR p{i+1}]\n{t}" if t else f"\n[p{i+1}: OCR empty]\n")
                if t: ocr_done += 1
            else:
                parts.append(f"\n[p{i+1}: scanned — pip install easyocr]\n")
        else:
            t = page.get_text()
            if t.strip(): parts.append(t)
    if scanned: print(f"    🔍 Scanned: {scanned} pages, {ocr_done} OCR'd")
    return "\n\n".join(parts)

# ==========================================================
# ========= IMAGE-TAG SECTION BUILDER (v4 NEW) =============
# ==========================================================

def _build_image_tag_section(paper_images: list[dict]) -> str:
    """
    Builds the ## Extracted Visual Assets Database section.

    This section is appended to the markdown BEFORE chunking.
    Each image gets one [IMAGE: filename.png] tag + its caption.
    When retrieved, the chunk carries the tag into the LLM context.
    The LLM then uses [IMAGE: filename.png] in its answer.
    cancer_app_v2.py renders st.image() for each tag found.

    Format (deliberately matches teammate's proven implementation):
      ## Extracted Visual Assets Database

      The following figures, tables and flowcharts were extracted...

      - [IMAGE: breast-cancer-review_picture_1.png]
        Caption: Figure 1. Different topical drug delivery systems...

      - [IMAGE: breast-cancer-review_table_1.png]
        Caption: Table 1. Clinical trial results for...
    """
    if not paper_images:
        return ""

    section  = "\n\n## Extracted Visual Assets Database\n\n"
    section += (
        "The following figures, tables and flowcharts were extracted from "
        "this paper. When answering questions about these visuals, reference "
        "them using their exact tag.\n\n"
    )
    for meta in paper_images:
        section += (
            f"- [IMAGE: {meta['image_filename']}]\n"
            f"  Caption: {meta['caption']}\n\n"
        )
    return section

# ==========================================================
# ============ SINGLE PDF PROCESSOR ========================
# ==========================================================

def process_single_pdf(pdf_path: Path) -> dict:
    """
    Full per-PDF pipeline:

    Phase 1 — EXTRACT TEXT
      a) Try Docling → markdown + element iterator
      b) Fallback: PyMuPDF → raw text
      c) Append table data (PyMuPDF block-mode)

    Phase 2 — CAPTION EXTRACTION + IMAGE SAVING
      a) Docling path: CaptionChunkBuilder.build_from_docling()
         4-tier logic: native → sliding_next → sliding_prev → context
      b) PyMuPDF fallback: CaptionChunkBuilder.build_from_pymupdf()

    Phase 3 — CLEAN MARKDOWN + INJECT [IMAGE:] TAGS (v4 NEW)
      clean_text() → append ## Extracted Visual Assets Database section
      → saved to output/markdown/{name}.md

    Phase 4 — CHUNK TEXT
      MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter
      Metadata: source_file, cancer_type, content_type, section_hierarchy,
                source_url, has_image_tags (v4 NEW)

    Phase 5 — SAVE
      Text chunks    → output/chunks/{name}_chunks.json
      Caption chunks → output/caption_chunks/{name}_caption_chunks.json
      Images         → output/images/ + image_metadata.json (appended)
    """
    name        = pdf_path.stem
    cancer_type = _detect_cancer_type(name)

    print(f"\n{'─'*62}")
    print(f"📄  {name}")
    print(f"    Cancer: {cancer_type}  |  Collection: {COLLECTION_REVIEWS}")
    print(f"    Size  : {pdf_path.stat().st_size // 1024} KB")

    try:
        doc        = fitz.open(pdf_path)
        page_count = len(doc)
        text_rich  = sum(1 for i in range(page_count) if len(doc[i].get_text().strip()) >= MIN_TEXT_CHARS)
        strats     = [PageProfile(doc[i]).strategy() for i in range(page_count)]
        counts     = {s: strats.count(s) for s in set(strats)}
        print(f"    Pages : {page_count}  Text-rich: {text_rich}")
        print("    📊 " + " | ".join(f"{k}={v}" for k, v in sorted(counts.items())))

        caption_builder = CaptionChunkBuilder(name, cancer_type)
        caption_docs: list[LCDoc] = []
        method = "pymupdf"

        if _DOCLING:
            try:
                po = PdfPipelineOptions()
                po.images_scale            = DOCLING_IMG_SCALE
                po.generate_page_images    = True
                po.generate_picture_images = True
                conv_result = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}
                ).convert(pdf_path, max_num_pages=200)
                raw_md = conv_result.document.export_to_markdown()

                if raw_md and len(raw_md.strip()) > 300:
                    print("    ✅ Docling text extraction succeeded")
                    method = "docling"
                    caption_docs, _ = caption_builder.build_from_docling(conv_result, IMAGE_DIR)
                    raw = raw_md
                else:
                    print("    ⚠️  Docling sparse — PyMuPDF fallback")
                    raw = _pymupdf_extract(doc)
                    caption_docs, _ = caption_builder.build_from_pymupdf(doc, IMAGE_DIR)
            except Exception as e:
                print(f"    ⚠️  Docling error ({e}) — fallback")
                raw = _pymupdf_extract(doc)
                caption_docs, _ = caption_builder.build_from_pymupdf(doc, IMAGE_DIR)
        else:
            raw = _pymupdf_extract(doc)
            caption_docs, _ = caption_builder.build_from_pymupdf(doc, IMAGE_DIR)

        tabs = extract_table_data(doc)
        if tabs:
            raw += tabs
            print("    📋 Table data appended")
        doc.close()

        # Phase 3: Clean + inject [IMAGE:] section
        cleaned = clean_text(raw)
        if len(cleaned) < 200:
            print(f"    ⚠️  Low text yield ({len(cleaned)} chars)")

        # ── v4: Build image-tag section from caption metadata ──────
        # caption_builder._master_meta has the definitive list of images
        # extracted from this paper. We build [IMAGE:] tags from it
        # and append to the markdown so tags land in Qdrant as text.
        tag_section = _build_image_tag_section(caption_builder._master_meta)
        if tag_section:
            cleaned += tag_section
            print(f"    🖼️  Injected {len(caption_builder._master_meta)} [IMAGE:] tags")

        (MD_DIR / f"{name}.md").write_text(cleaned, encoding="utf-8")

        # Phase 4: Chunk text
        text_chunks = _chunk_text(cleaned, name, cancer_type)
        type_counts: dict = {}
        for c in text_chunks:
            ct = c["content_type"]; type_counts[ct] = type_counts.get(ct, 0) + 1
        img_chunks = sum(1 for c in text_chunks if c.get("has_image_tags"))

        # Phase 5: Save
        with open(CHUNK_DIR / f"{name}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(text_chunks, f, indent=2, ensure_ascii=False)

        cap_payload = [{**d.metadata, "content": d.page_content} for d in caption_docs]
        with open(CAP_DIR / f"{name}_caption_chunks.json", "w", encoding="utf-8") as f:
            json.dump(cap_payload, f, indent=2, ensure_ascii=False)

        _update_master_image_metadata(caption_builder._master_meta)

        print(f"    ✅ {method} | Text: {len(text_chunks)} chunks "
              f"({img_chunks} with image tags) | Captions: {len(caption_docs)}")
        for ct, n in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"       → {n:3d}  {ct}")

        return {
            "file": name, "cancer_type": cancer_type, "collection": COLLECTION_REVIEWS,
            "pages": page_count, "text_pages": text_rich,
            "text_chunks": len(text_chunks), "image_tag_chunks": img_chunks,
            "caption_chunks": len(caption_docs), "images_extracted": len(caption_builder._master_meta),
            "method": method, "status": "success",
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"file": name, "status": "failed", "error": str(e)}


def _update_master_image_metadata(new_entries: list[dict]) -> None:
    """Appends to image_metadata.json, deduped by image_filename."""
    meta_path = IMAGE_DIR / "image_metadata.json"
    existing: list = []
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f: existing = json.load(f)
        except Exception: existing = []
    seen = {e["image_filename"] for e in existing}
    for entry in new_entries:
        if entry.get("image_filename") not in seen:
            existing.append(entry); seen.add(entry["image_filename"])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

# ==========================================================
# ====================== CHUNKING ==========================
# ==========================================================

def _chunk_text(cleaned: str, source_name: str, cancer_type: str) -> list:
    """
    Two-stage section-aware chunking:
      Stage 1 — MarkdownHeaderTextSplitter: structural boundaries (H1/H2/H3)
      Stage 2 — RecursiveCharacterTextSplitter: size limits

    The ## Extracted Visual Assets Database section is treated as a normal
    H2 section → chunks within it carry [IMAGE:] tags naturally.

    Metadata per chunk:
      chunk_id, source_file, source_url, collection, cancer_type,
      content_type, section_hierarchy, char_length, has_image_tags (v4)
    """
    h_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    t_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        chunks = t_splitter.split_documents(h_splitter.split_text(cleaned))
    except Exception:
        chunks = t_splitter.split_documents([LCDoc(page_content=cleaned)])

    payload: list = []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if len(content) < MIN_CHUNK_CHARS:
            continue
        hierarchy = " > ".join(
            v for k, v in chunk.metadata.items() if k.startswith("H") and v
        ) or "Body"
        payload.append({
            "chunk_id":          f"{source_name}_{i:04d}",
            "source_file":       source_name,
            "source_url":        _get_source_url(source_name),
            "collection":        COLLECTION_REVIEWS,
            "cancer_type":       cancer_type,
            "content_type":      detect_content_type(content),
            "section_hierarchy": hierarchy,
            "content":           content,
            "char_length":       len(content),
            "has_image_tags":    "[IMAGE:" in content,   # v4 flag
        })
    return payload

# ==========================================================
# ============= QDRANT BUILD (PRECOMPUTED VECTORS) =========
# ==========================================================

def _create_empty_patient_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_PATIENTS in existing:
        print(f"    ⏭️   '{COLLECTION_PATIENTS}' already exists"); return
    client.create_collection(
        collection_name=COLLECTION_PATIENTS,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"    ✅ '{COLLECTION_PATIENTS}' created (empty placeholder)")


def build_vector_store(force_rebuild: bool = False) -> None:
    """
    Compute embeddings ONCE → save JSONL export → push to Qdrant.

    1. Load text docs from output/chunks/
    2. Load caption docs from output/caption_chunks/
    3. Load embedding model once
    4. Embed text → save text_embeddings.jsonl
    5. Embed captions → save caption_embeddings.jsonl
    6. save_summary() → validate norms
    7. Upsert pre-computed vectors to Qdrant (batched)
    8. Create empty patient_records collection
    """
    print(f"\n{'─'*62}")
    print(f"🗄️   VECTOR STORE + EMBEDDING EXPORT")
    print(f"    Qdrant : {QDRANT_DIR}")
    print(f"    Export : {EMBED_DIR}")

    qdrant_meta = QDRANT_DIR / "meta.json"
    if qdrant_meta.exists() and not force_rebuild:
        print("    ✅ Collections exist — skipping rebuild")
        print("    ⏭️   To rebuild: main(force_rebuild=True)")
        return

    # 1 — Load text docs
    text_docs: list[LCDoc] = []
    for jp in sorted(CHUNK_DIR.glob("*_chunks.json")):
        with open(jp, encoding="utf-8") as f:
            for c in json.load(f):
                text_docs.append(LCDoc(page_content=c["content"], metadata=c))

    # 2 — Load caption docs
    caption_docs: list[LCDoc] = []
    for jp in sorted(CAP_DIR.glob("*_caption_chunks.json")):
        with open(jp, encoding="utf-8") as f:
            for c in json.load(f):
                content = c.pop("content", "")
                caption_docs.append(LCDoc(page_content=content, metadata=c))

    img_tag_count = sum(1 for d in text_docs if d.metadata.get("has_image_tags"))
    print(f"\n    📦 {len(text_docs)} text ({img_tag_count} with image tags) "
          f"+ {len(caption_docs)} caption = "
          f"{len(text_docs)+len(caption_docs)} total")
    if not text_docs and not caption_docs:
        print("    ❌ No documents found — run PDF processing first"); return

    # 3 — Load model once
    print(f"    🔢 Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    exporter = EmbeddingExporter()

    # 4 — Embed text
    print(f"\n    🔄 Embedding {len(text_docs)} text chunks...")
    text_vecs = embed_model.embed_documents([d.page_content for d in text_docs])
    exporter.save_text_embeddings(text_docs, text_vecs)

    # 5 — Embed captions
    if caption_docs:
        print(f"\n    🔄 Embedding {len(caption_docs)} caption chunks...")
        cap_vecs = embed_model.embed_documents([d.page_content for d in caption_docs])
        exporter.save_caption_embeddings(caption_docs, cap_vecs)
    else:
        cap_vecs = []
        print("    ℹ️  No caption chunks to embed")

    # 6 — Summary
    exporter.save_summary()

    # 7 — Upsert to Qdrant
    print(f"\n    💾 Upserting to '{COLLECTION_REVIEWS}'...")
    client = QdrantClient(path=str(QDRANT_DIR))

    existing_cols = [c.name for c in client.get_collections().collections]
    if COLLECTION_REVIEWS in existing_cols:
        client.delete_collection(COLLECTION_REVIEWS)
    client.create_collection(
        collection_name=COLLECTION_REVIEWS,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    all_docs = text_docs + caption_docs
    all_vecs = text_vecs + list(cap_vecs)
    points: list[PointStruct] = []

    for doc, vec in zip(all_docs, all_vecs):
        payload = {
            "page_content": doc.page_content,
            "metadata": {k: (v if v is not None else "") for k, v in doc.metadata.items()}
        }
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i : i + UPSERT_BATCH]
        client.upsert(collection_name=COLLECTION_REVIEWS, points=batch)
        print(f"    ↳ {min(i+UPSERT_BATCH, len(points)):5d} / {len(points)} points upserted")

    print(f"    ✅ '{COLLECTION_REVIEWS}' complete")

    # 8 — Patient records placeholder
    print(f"\n    🏥 Setting up '{COLLECTION_PATIENTS}'...")
    _create_empty_patient_collection(client)
    client.close()

# ==========================================================
# ============================ MAIN ========================
# ==========================================================

def main(force_rebuild: bool = False) -> None:
    print("=" * 62)
    print("  cancer_ingestion.py — v4 Image-Tag Injection Pipeline")
    print("=" * 62)
    print(f"  Docling  : {'✅' if _DOCLING else '⬜ fallback active'}")
    print(f"  EasyOCR  : {'✅' if _EASYOCR  else '⬜ optional'}")
    print(f"  NumPy    : {'✅' if _NUMPY    else '⬜ pure Python fallback'}")
    print(f"\n  Collections : '{COLLECTION_REVIEWS}'  +  '{COLLECTION_PATIENTS}'")
    print(f"  Image tags  : ON — [IMAGE: filename.png] injected into markdown")
    print(f"  Embed export: {EMBED_DIR}")

    ensure_dirs()
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌  No PDFs in: {PDF_DIR}"); return

    print(f"\n📂  {len(pdf_files)} PDF(s):")
    for p in pdf_files:
        print(f"    • {p.name:<45} [{_detect_cancer_type(p.stem)}]")

    results: list = []
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        results.append(process_single_pdf(pdf))

    with open(LOG_DIR / "pipeline_log.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    build_vector_store(force_rebuild=force_rebuild)

    success        = [r for r in results if r.get("status") == "success"]
    failed         = [r for r in results if r.get("status") != "success"]
    total_text     = sum(r.get("text_chunks", 0)      for r in success)
    total_imgtags  = sum(r.get("image_tag_chunks", 0) for r in success)
    total_captions = sum(r.get("caption_chunks", 0)   for r in success)
    total_images   = sum(r.get("images_extracted", 0) for r in success)

    print(f"\n{'='*62}")
    print(f"  PIPELINE COMPLETE — v4")
    print(f"{'='*62}")
    print(f"  ✅  PDFs processed      : {len(success)} / {len(results)}")
    print(f"  📦  Text chunks         : {total_text}")
    print(f"  🖼️   Image-tag chunks    : {total_imgtags}  ← carry [IMAGE:] tags")
    print(f"  📸  Caption chunks      : {total_captions}  ← embedded in Qdrant")
    print(f"  🗃️   Total in Qdrant     : {total_text + total_captions}")
    print(f"  🖼️   Images saved        : {total_images}")
    print(f"\n  Output directories:")
    print(f"  🖼️   PNG files          : {IMAGE_DIR}")
    print(f"  🗂️   Image registry     : {IMAGE_DIR / 'image_metadata.json'}")
    print(f"  📊  Text embeddings    : {EMBED_DIR / 'text_embeddings.jsonl'}")
    print(f"  📊  Caption embeddings : {EMBED_DIR / 'caption_embeddings.jsonl'}")
    print(f"  📋  Embedding summary  : {EMBED_DIR / 'embedding_summary.json'}")
    print(f"  🗄️   Qdrant store       : {QDRANT_DIR}")

    if failed:
        print(f"\n  ❌  Failed:")
        for r in failed: print(f"      {r['file']}: {r.get('error','?')}")

    # Summary breakdowns
    all_chunks: list = []
    for cf in list(CHUNK_DIR.glob("*_chunks.json")) + list(CAP_DIR.glob("*_caption_chunks.json")):
        with open(cf, encoding="utf-8") as f: all_chunks.extend(json.load(f))

    bar = max(1, len(all_chunks) // 25)
    print(f"\n  Chunks per cancer type:")
    ct_map: dict = {}
    for c in all_chunks:
        k = c.get("cancer_type","?"); ct_map[k] = ct_map.get(k,0)+1
    for cancer, count in sorted(ct_map.items(), key=lambda x: -x[1]):
        print(f"    {cancer:<20} {count:4d}  {'█'*(count//bar)}")

    print(f"\n  Content type breakdown:")
    tt: dict = {}
    for c in all_chunks:
        k = c.get("content_type","?"); tt[k] = tt.get(k,0)+1
    for k, v in sorted(tt.items(), key=lambda x: -x[1]):
        print(f"    {k:<32} {v:4d}  {'█'*(v//bar)}")

    print(f"\n  Caption extraction tiers:")
    tiers: dict = {}
    for cf in CAP_DIR.glob("*_caption_chunks.json"):
        with open(cf, encoding="utf-8") as f:
            for c in json.load(f):
                t = c.get("extraction_tier","?"); tiers[t] = tiers.get(t,0)+1
    for tier, count in sorted(tiers.items(), key=lambda x: -x[1]):
        print(f"    {tier:<28} {count:4d}")

    print(f"\n  ➡️   Next: streamlit run cancer_app_v2.py")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    # First run / after adding new PDFs → force_rebuild=True
    # Normal run (collections exist)   → False (skips Qdrant rebuild)
    main(force_rebuild=True)