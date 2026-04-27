from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from pathlib import Path
import json
import shutil
import logging

INPUT_FILE = "D:\\trial\\data\\IBMMaaS360_Best_Practices_for_Policies.pdf"
OUTPUT_DIR = Path("Em_output_data")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_converter():
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.generate_picture_images = True
    opts.images_scale = 2.0

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

def safety_split(text, tokenizer, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunks.append(tokenizer.decode(tokens[i:i+max_tokens]))
    return chunks

def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    converter = setup_converter()
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

    logger.info("Converting PDF...")
    doc = converter.convert(INPUT_FILE).document

    logger.info("Adaptive chunking with HybridChunker...")
    chunker = HybridChunker(
        tokenizer=EMBED_MODEL_ID,
        max_tokens=MAX_TOKENS,
        merge_peers=True
    )

    raw_chunks = chunker.chunk(dl_doc=doc)

    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        safe_texts = safety_split(chunk.text, tokenizer, MAX_TOKENS)
        for j, txt in enumerate(safe_texts):
            final_chunks.append({
                "id": f"{i}_{j}",
                "text": txt,
                "metadata": chunk.meta.export_json_dict()
            })

    with open(OUTPUT_DIR / "adaptive_chunks.jsonl", "w", encoding="utf-8") as f:
        for c in final_chunks:
            f.write(json.dumps(c) + "\n")

    logger.info(f"Created {len(final_chunks)} adaptive chunks.")

if __name__ == "__main__":
    main()
