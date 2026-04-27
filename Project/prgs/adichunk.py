import os
import time
import logging
import shutil
import json
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import PictureItem, TableItem
from docling.chunking import HybridChunker  

import ollama
from transformers import AutoTokenizer

INPUT_FILE = "PDFs\IBMMaaS360_Best_Practices_for_Policies.pdf"
OUTPUT_DIR = Path("Em_output_data")
OLLAMA_MODEL = "qwen25-vl:3b" 
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2" # The model defining the 512 limit
MAX_TOKENS = 512

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0 
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

def analyze_with_ollama(image_path):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': 'Describe this image details for a technical manual.', 'images': [str(image_path)]}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"(Ollama Error: {e})"

def safety_split(text, tokenizer, max_tokens):
    """
    CPU-efficient fallback: splits chunks that are STILL too big 
    (e.g., giant tables) after HybridChunker is done.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]
    
    logger.warning(f"Safety-splitting a runaway chunk of {len(tokens)} tokens...")
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

def main():
    images_dir = OUTPUT_DIR / "images"
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    images_dir.mkdir(parents=True)

    converter = setup_converter()
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    
    try:
        logger.info(f"Converting {INPUT_FILE}...")
        result = converter.convert(INPUT_FILE)
        doc = result.document 
        
        # --- PART A: VISUAL REPORT ---
        visual_content = ["# VISUAL ANALYSIS\n\n"]
        img_idx = 0
        for item, level in doc.iterate_items():
            if isinstance(item, PictureItem):
                img_idx += 1
                img_path = images_dir / f"image_{img_idx}.png"
                pil_img = item.get_image(doc)
                if pil_img:
                    pil_img.save(img_path)
                    desc = analyze_with_ollama(img_path)
                    visual_content.append(f"### Visual {img_idx}\n![{img_idx}]({img_path})\n**Description:** {desc}\n\n")

        # --- PART B: ROBUST ADAPTIVE CHUNKING ---
        logger.info("Starting Hybrid Chunking with Safety Pass...")
        chunker = HybridChunker(
            tokenizer=EMBED_MODEL_ID, 
            max_tokens=MAX_TOKENS,  
            merge_peers=True 
        )
        
        raw_chunks = chunker.chunk(dl_doc=doc)
        final_chunks = []
        
        for i, chunk in enumerate(raw_chunks):
            # Check if this chunk is still too big (HybridChunker sometimes misses limits on giant tables)
            split_texts = safety_split(chunk.text, tokenizer, MAX_TOKENS)
            
            for sub_idx, text in enumerate(split_texts):
                final_chunks.append({
                    "id": f"{i}_{sub_idx}",
                    "text": text,
                    "metadata": chunk.meta.export_json_dict()
                })

        # Save results
        with open(OUTPUT_DIR / "adaptive_chunks.jsonl", "w", encoding="utf-8") as f:
            for c in final_chunks: f.write(json.dumps(c) + "\n")
                
        logger.info(f"Done! Created {len(final_chunks)} embedding-safe chunks.")

    finally:
        del converter

if __name__ == "__main__":
    main()
