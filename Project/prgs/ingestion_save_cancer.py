import os
import re
import json
import shutil
import unicodedata
from pathlib import Path

# Docling for extraction
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem

# LangChain for chunking & Qdrant
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_qdrant import QdrantVectorStore

# Paths
PDF_DIR = r"D:\trial\pdfs"
MD_DIR = r"D:\trial\markdown_papers"
JSON_DIR = r"D:\trial\rag_chunks"
QDRANT_DIR = r"D:\trial\qdrant_db"
IMG_DIR = r"D:\trial\extracted_images"

def clean_and_polish_markdown(text):
    weird_numbers = {
        '': '0', '': '1', '': '2', '': '3', '': '4', 
        '': '5', '': '6', '': '7', '': '8', '': '9'
    }
    for weird, normal in weird_numbers.items():
        text = text.replace(weird, normal)
        
    text = unicodedata.normalize('NFKC', text)
    
    ref_pattern = re.compile(r'^#*\s*\*?\*?(references?|bibliography|literature cited|works cited)\*?\*?\s*$', re.IGNORECASE | re.MULTILINE)
    match = ref_pattern.search(text)
    if match: text = text[:match.start()].strip()
        
    replacements = {
        r'\s*/uniFB01\s*': 'fi', r'\s*/uniFB02\s*': 'fl', r'\s*/uniFB00\s*': 'ff',
        r'\s*/uniFB03\s*': 'ffi', r'\s*/uniFB04\s*': 'ffl', r'\s*/uniF642\s*': '%',
        r'\s*/C15\s*': '-', r'\s*/C19\s*': 'e', r'\s*/C20\s*': 'c', r'\s*/C211\s*': ' ',
        r'ﬁ': 'fi', r'ﬂ': 'fl'
    }
    for pattern, fixed in replacements.items(): text = re.sub(pattern, fixed, text)
        
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2\3', text)
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)(?=[a-zA-Z])', r'\1\2', text)
    text = re.sub(r'(?<=[a-zA-Z])(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def process_pdfs():
    print("--- 1. EXTRACTING PDFs, IMAGES, TABLES & METADATA ---")
    os.makedirs(MD_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0 
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    pdf_paths = list(Path(PDF_DIR).glob("*.pdf"))
    all_images_metadata = []
    
    for pdf_path in pdf_paths:
        print(f"Extracting: {pdf_path.name}")
        conv_result = converter.convert(pdf_path)
        
        table_counter = 0
        picture_counter = 0
        local_metadata_injection = "\n\n## Extracted Visual Assets Database\n"
        
        # Convert the iterator to a list so we can use a sliding window to look ahead/behind
        items = list(conv_result.document.iterate_items())
        
        for i, (element, _level) in enumerate(items):
            is_picture = isinstance(element, PictureItem)
            is_table = isinstance(element, TableItem)
            
            if is_picture or is_table:
                img = element.get_image(conv_result.document)
                if img:
                    if is_picture:
                        picture_counter += 1
                        img_filename = f"{pdf_path.stem}_picture_{picture_counter}.png"
                    else:
                        table_counter += 1
                        img_filename = f"{pdf_path.stem}_table_{table_counter}.png"
                        
                    with open(Path(IMG_DIR) / img_filename, "wb") as fp:
                        img.save(fp, "PNG")
                    
                    # --- ROBUST CAPTION EXTRACTION LOGIC ---
                    caption_text = ""
                    
                    # 1. Try Docling's native caption linking first
                    captions = getattr(element, "captions", [])
                    caption_parts = [c.text for c in captions if hasattr(c, 'text') and c.text]
                    if caption_parts:
                        caption_text = " ".join(caption_parts).strip()
                        
                    # 2. Sliding Window Heuristic: Look at the NEXT element
                    if not caption_text and (i + 1 < len(items)):
                        next_el = items[i+1][0]
                        if hasattr(next_el, 'text') and next_el.text:
                            next_text = next_el.text.strip()
                            if next_text.lower().startswith(("fig", "table")):
                                caption_text = next_text
                                
                    # 3. Sliding Window Heuristic: Look at the PREVIOUS element
                    if not caption_text and (i - 1 >= 0):
                        prev_el = items[i-1][0]
                        if hasattr(prev_el, 'text') and prev_el.text:
                            prev_text = prev_el.text.strip()
                            if prev_text.lower().startswith(("fig", "table")):
                                caption_text = prev_text
                                
                    # 4. Ultimate Fallback: Grab surrounding context
                    if not caption_text and (i + 1 < len(items)):
                        next_el = items[i+1][0]
                        if hasattr(next_el, 'text') and next_el.text:
                            # Save just the first 250 characters of the next paragraph to give the LLM clues
                            caption_text = f"Context: {next_el.text.strip()[:250]}..."
                            
                    if not caption_text:
                        caption_text = "Visual asset extracted from document without clear caption."
                        
                    # Save to JSON registry
                    meta_dict = {
                        "source_file": pdf_path.name,
                        "image_filename": img_filename,
                        "type": "Picture" if is_picture else "Table",
                        "caption": caption_text
                    }
                    all_images_metadata.append(meta_dict)
                    
                    # Inject into Markdown
                    local_metadata_injection += f"- **File**: {img_filename}\n  **Caption**: {caption_text}\n\n"
                        
        print(f"  ✅ Saved {table_counter} tables and {picture_counter} pictures.")
        
        final_markdown = clean_and_polish_markdown(conv_result.document.export_to_markdown())
        
        if picture_counter > 0 or table_counter > 0:
            final_markdown += local_metadata_injection
            
        with open(Path(MD_DIR) / f"{pdf_path.stem}.md", "w", encoding="utf-8") as f:
            f.write(final_markdown)

    with open(Path(IMG_DIR) / "image_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_images_metadata, f, indent=4, ensure_ascii=False)
    print(f"  ✅ Master Image Metadata saved to {IMG_DIR}/image_metadata.json")

def chunk_markdown():
    print("\n--- 2. SEMANTIC CHUNKING ---")
    os.makedirs(JSON_DIR, exist_ok=True)
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for md_path in Path(MD_DIR).glob("*.md"):
        print(f"Chunking: {md_path.name}")
        with open(md_path, "r", encoding="utf-8") as f:
            splits = text_splitter.split_documents(markdown_splitter.split_text(f.read()))
            
        payload = [{
            "chunk_id": f"{md_path.stem}_{i}",
            "source_file": md_path.name,
            "section_hierarchy": " > ".join([v for k, v in chunk.metadata.items() if k.startswith("H")]) or "Body",
            "content": chunk.page_content
        } for i, chunk in enumerate(splits)]
        
        with open(Path(JSON_DIR) / f"{md_path.stem}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

def build_vector_db():
    print("\n--- 3. BUILDING QDRANT DB ---")
    documents = []
    for json_path in Path(JSON_DIR).glob("*_chunks.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                documents.append(Document(page_content=chunk["content"], metadata=chunk))
                
    if os.path.exists(QDRANT_DIR):
        shutil.rmtree(QDRANT_DIR)
        
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    QdrantVectorStore.from_documents(
        documents=documents, embedding=embeddings, path=QDRANT_DIR, 
        collection_name="medical_papers", force_recreate=True
    )
    print("✅ Ingestion Complete! Database is ready.")

if __name__ == "__main__":
    if not os.path.exists(PDF_DIR): os.makedirs(PDF_DIR)
    process_pdfs()
    chunk_markdown()
    build_vector_db()