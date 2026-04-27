import json
import base64
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import fitz  # PyMuPDF
from PIL import Image

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import TableItem, PictureItem


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "moondream"  # Recommended: "minicpm-v" if your hardware supports it

# IMPROVED PROMPT: Focuses on structure and visual elements, avoiding OCR overlap
VISION_PROMPT = """
You are explaining this page to a human who cannot see the image.

Describe in detail:
- What kind of enterprise software or documentation this page belongs to
- Any visible configuration screens, admin panels, or UI screenshots
- What settings, controls, or options appear to be shown
- How the visual elements are grouped and what they are used for
- What task a user would perform using what is shown on this page

Important rules:
- Do NOT describe layout alone.
- Do NOT stop at high-level structure.
- Focus on what the page is SHOWING, not just how it is arranged.
- Do NOT guess unreadable text.
- If the page is mostly text, explain what the text appears to be about.

Write 5–8 complete sentences.

"""

# OPTIMIZATION SETTINGS
RENDER_DPI = 200          # Increased from 72 to 200 for legibility
REQUEST_TIMEOUT = 120     # Timeout in seconds
MAX_RETRIES = 2           # Retry failed requests
RETRY_DELAY = 2           # Seconds between retries

# IMAGE SETTINGS
MAX_IMAGE_SIZE = (1024, 1024)  # Increased from 800x800 for better detail


# ============================================================
# IMAGE PROCESSING
# ============================================================

def optimize_image(image: Image.Image) -> Image.Image:
    """
    Optimize image for faster processing while maintaining detail.
    """
    # Convert to RGB if needed
    if image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    
    # Resize if too large, using high-quality resampling
    if image.width > MAX_IMAGE_SIZE[0] or image.height > MAX_IMAGE_SIZE[1]:
        image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    return image


def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """
    Convert PIL Image to base64 with compression.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def render_page_as_image(pdf_doc: fitz.Document, page_no: int) -> Optional[Image.Image]:
    """
    Render a single PDF page as HIGH-DPI image for vision analysis.
    """
    try:
        page = pdf_doc[page_no - 1]
        
        # Calculate zoom based on target DPI (Standard PDF is 72 DPI)
        zoom = RENDER_DPI / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # alpha=False forces white background (important for transparent PDFs)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        mode = "RGB" if pix.n == 3 else "L"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        
        return img
        
    except Exception as e:
        print(f"  ✗ Failed to render page {page_no}: {str(e)}")
        return None


# ============================================================
# MOONDREAM VISION ANALYSIS
# ============================================================

def describe_page_with_moondream(
    image: Image.Image, 
    page_no: int,
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES
) -> str:
    """
    Send page image to Moondream with retry logic and optimization.
    """
    # Optimize image first
    optimized_image = optimize_image(image)
    
    for attempt in range(max_retries):
        try:
            # Convert to base64
            image_b64 = image_to_base64(optimized_image)

            payload = {
                "model": VISION_MODEL,
                "prompt": VISION_PROMPT,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "num_predict": 300,   # Increased token limit for full descriptions
                    "temperature": 0.2,   # Slight creativity allowed
                }
            }

            print(f"  → Analyzing page {page_no} (attempt {attempt + 1}/{max_retries})...")
            
            response = requests.post(
                OLLAMA_URL, 
                json=payload, 
                timeout=timeout
            )
            response.raise_for_status()
            
            description = response.json().get("response", "").strip()
            
            if description:
                print(f"  ✓ Page {page_no} complete")
                return description
            else:
                return "No visual description generated"
                
        except requests.exceptions.Timeout:
            print(f"  ⚠ Page {page_no} timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                return f"Vision analysis timed out after {max_retries} attempts"
                
        except requests.exceptions.RequestException as e:
            print(f"  ⚠ Page {page_no} request error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                return f"Vision analysis failed: {str(e)}"
                
        except Exception as e:
            print(f"  ✗ Page {page_no} error: {str(e)}")
            return f"Error processing image: {str(e)}"
    
    return "Vision analysis failed after retries"


def analyze_pages_parallel(
    pdf_path: str, 
    max_workers: int = 2,
    timeout: int = REQUEST_TIMEOUT
) -> Dict[int, str]:
    """
    Render and analyze all pages in parallel.
    """
    pdf_doc = fitz.open(pdf_path)
    page_count = len(pdf_doc)
    
    print(f"\nAnalyzing {page_count} pages with Moondream ({max_workers} workers)...")
    
    visual_descriptions = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {}
        
        # Submit all pages
        for page_no in range(1, page_count + 1):
            img = render_page_as_image(pdf_doc, page_no)
            
            if img is None:
                visual_descriptions[page_no] = "Failed to render page"
                continue
            
            future = executor.submit(
                describe_page_with_moondream, 
                img, 
                page_no,
                timeout
            )
            future_to_page[future] = page_no
        
        # Collect results
        for future in as_completed(future_to_page):
            page_no = future_to_page[future]
            try:
                description = future.result()
                visual_descriptions[page_no] = description
            except Exception as e:
                visual_descriptions[page_no] = f"Error: {str(e)}"
    
    pdf_doc.close()
    
    # Summary
    successful = sum(1 for v in visual_descriptions.values() if not v.startswith(("Error", "Vision analysis failed", "Failed")))
    print(f"\n✓ Completed: {successful}/{page_count} pages analyzed successfully")
    
    return visual_descriptions


def analyze_pages_sequential(
    pdf_path: str,
    timeout: int = REQUEST_TIMEOUT
) -> Dict[int, str]:
    """
    Sequential fallback.
    """
    pdf_doc = fitz.open(pdf_path)
    page_count = len(pdf_doc)
    
    print(f"\nAnalyzing {page_count} pages sequentially (safer mode)...")
    
    visual_descriptions = {}
    
    for page_no in range(1, page_count + 1):
        img = render_page_as_image(pdf_doc, page_no)
        
        if img is None:
            visual_descriptions[page_no] = "Failed to render page"
            continue
        
        description = describe_page_with_moondream(img, page_no, timeout)
        visual_descriptions[page_no] = description
        
        time.sleep(0.5)
    
    pdf_doc.close()
    
    successful = sum(1 for v in visual_descriptions.values() if not v.startswith(("Error", "Vision analysis failed", "Failed")))
    print(f"\n✓ Completed: {successful}/{page_count} pages analyzed successfully")
    
    return visual_descriptions


# ============================================================
# DOCLING STRUCTURED EXTRACTION
# ============================================================

def extract_with_docling(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract structured content using Docling.
    """
    pipeline_opts = PdfPipelineOptions(
        do_table_structure=True,
        generate_picture_images=False,
        do_picture_description=False,
        do_ocr=False,
    )
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )
    
    print("\nExtracting text and tables with Docling...")
    result = converter.convert(pdf_path)
    doc = result.document
    
    pages_data = {}
    
    for item, _ in doc.iterate_items():
        if not hasattr(item, 'prov') or not item.prov:
            continue
        
        page_no = item.prov[0].page_no
        
        if page_no not in pages_data:
            pages_data[page_no] = {
                'page': page_no,
                'text_blocks': [],
                'tables': [],
            }
        
        if isinstance(item, TableItem):
            try:
                table_md = item.export_to_markdown(doc=doc)
                pages_data[page_no]['tables'].append(table_md)
            except:
                try:
                    table_md = item.export_to_markdown()
                    pages_data[page_no]['tables'].append(table_md)
                except:
                    pass
        else:
            try:
                if hasattr(item, 'export_to_markdown'):
                    text = item.export_to_markdown()
                else:
                    text = getattr(item, 'text', str(item))
                
                if text and text.strip():
                    pages_data[page_no]['text_blocks'].append(text.strip())
            except:
                pass
    
    pages_list = [pages_data[page_no] for page_no in sorted(pages_data.keys())]
    print(f"✓ Extracted {len(pages_list)} pages")
    
    return pages_list


# ============================================================
# MERGE: DOCLING + VISION
# ============================================================

def merge_content_and_vision(
    docling_pages: List[Dict[str, Any]],
    visual_descriptions: Dict[int, str]
) -> List[Dict[str, Any]]:
    """
    Merge Docling structured content with Moondream visual descriptions.
    """
    merged = []
    
    for page_data in docling_pages:
        page_no = page_data['page']
        
        full_text = '\n\n'.join(page_data['text_blocks'])
        tables_text = '\n\n'.join(page_data['tables']) if page_data['tables'] else None
        visual_summary = visual_descriptions.get(page_no, "No visual analysis available")
        
        vision_failed = visual_summary.startswith(("Error", "Vision analysis failed", "Failed"))
        
        page_obj = {
            'page_number': page_no,
            'text_content': full_text,
            'tables': page_data['tables'],
            'visual_summary': visual_summary,
            'vision_success': not vision_failed,
            'has_visual_elements': not visual_summary.startswith("Text-only page") and not vision_failed,
        }
        
        # Combined content for RAG
        combined_parts = [full_text]
        if tables_text:
            combined_parts.append(f"\n[TABLES]\n{tables_text}")
        if page_obj['has_visual_elements']:
            combined_parts.append(f"\n[VISUAL ELEMENTS]\n{visual_summary}")
        
        page_obj['combined_content'] = '\n'.join(combined_parts)
        
        merged.append(page_obj)
    
    return merged


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_enterprise_pdf(
    pdf_path: str, 
    max_workers: int = 2,
    sequential: bool = False,
    timeout: int = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """
    Complete pipeline.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*80}")
    
    # Step 1: Docling extraction
    docling_pages = extract_with_docling(pdf_path)
    
    # Step 2: Vision analysis
    if sequential:
        visual_descriptions = analyze_pages_sequential(pdf_path, timeout)
    else:
        visual_descriptions = analyze_pages_parallel(pdf_path, max_workers, timeout)
    
    # Step 3: Merge
    print("\nMerging content and visual summaries...")
    merged_pages = merge_content_and_vision(docling_pages, visual_descriptions)
    
    # Statistics
    total_pages = len(merged_pages)
    successful_vision = sum(1 for p in merged_pages if p['vision_success'])
    pages_with_visuals = sum(1 for p in merged_pages if p['has_visual_elements'])
    
    output = {
        'source_pdf': Path(pdf_path).name,
        'total_pages': total_pages,
        'pages': merged_pages,
        'metadata': {
            'extraction_method': 'Docling + Moondream',
            'vision_model': VISION_MODEL,
            'render_dpi': RENDER_DPI,
            'successful_vision_analysis': successful_vision,
            'pages_with_visual_elements': pages_with_visuals,
        }
    }
    
    print(f"\n{'='*80}")
    print(f"Processing Complete")
    print(f"{'='*80}")
    print(f"  Total pages: {total_pages}")
    print(f"  Vision analysis success: {successful_vision}/{total_pages}")
    print(f"  Pages with visual elements: {pages_with_visuals}")
    print(f"{'='*80}\n")
    
    return output


# ============================================================
# EXPORT UTILITIES
# ============================================================

def export_to_json(output: Dict[str, Any], json_path: str):
    """Export to JSON file."""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"→ Saved JSON: {json_path}")


def export_to_text(output: Dict[str, Any], txt_path: str):
    """Export to plain text for review."""
    lines = [f"Source: {output['source_pdf']}\n", "=" * 80, "\n"]
    
    for page in output['pages']:
        lines.append(f"\n{'='*80}")
        lines.append(f"PAGE {page['page_number']}")
        lines.append(f"{'='*80}\n")
        
        if page['text_content']:
            lines.append(page['text_content'])
        
        if page['tables']:
            lines.append(f"\n[TABLES: {len(page['tables'])} found]")
            for idx, table in enumerate(page['tables'], 1):
                lines.append(f"\nTable {idx}:")
                lines.append(table)
        
        lines.append(f"\n[VISUAL ANALYSIS]")
        if page['vision_success']:
            lines.append(f"{page['visual_summary']}")
        else:
            lines.append(f"✗ {page['visual_summary']}")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"→ Saved text: {txt_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text and visual content from PDFs')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers (default: 2)')
    parser.add_argument('--sequential', action='store_true', help='Use sequential processing (slower but safer)')
    parser.add_argument('--timeout', type=int, default=120, help='Request timeout in seconds (default: 120)')
    
    args = parser.parse_args()
    
    if not Path(args.pdf_path).exists():
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("⚠ Warning: Ollama may not be running properly")
    except:
        print("✗ Error: Cannot connect to Ollama. Make sure it's running:")
        print("  ollama serve")
        sys.exit(1)
    
    # Process
    output = process_enterprise_pdf(
        args.pdf_path, 
        max_workers=args.workers,
        sequential=args.sequential,
        timeout=args.timeout
    )
    
    # Export
    base_name = Path(args.pdf_path).stem
    export_to_json(output, f"{base_name}_rag.json")
    export_to_text(output, f"{base_name}_review.txt")
    
    print("\nPipeline complete!")