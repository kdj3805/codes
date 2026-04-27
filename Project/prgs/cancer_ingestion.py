import os
import re
import json
import shutil
import unicodedata
from pathlib import Path

# Docling for extraction
from docling.document_converter import DocumentConverter

# LangChain for chunking & Qdrant
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Paths
PDF_DIR = r"D:\trial\pdfs"
MD_DIR = r"D:\trial\markdown_papers"
JSON_DIR = r"D:\trial\rag_chunks"
QDRANT_DIR = r"D:\trial\qdrant_db"

def clean_and_polish_markdown(text):
    # 1. FORCE FIX PDF NUMBERS BEFORE ANYTHING ELSE
    # Manually map the weird mathematical unicode digits from the PDF to standard numbers
    weird_numbers = {
        '': '0', '': '1', '': '2', '': '3', '': '4', 
        '': '5', '': '6', '': '7', '': '8', '': '9'
    }
    for weird, normal in weird_numbers.items():
        text = text.replace(weird, normal)
        
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Chop off References section aggressively
    ref_pattern = re.compile(r'^#*\s*\*?\*?(references?|bibliography|literature cited|works cited)\*?\*?\s*$', re.IGNORECASE | re.MULTILINE)
    match = ref_pattern.search(text)
    if match: text = text[:match.start()].strip()
        
    # 3. Clean PDF ligature error codes
    replacements = {
        r'\s*/uniFB01\s*': 'fi', r'\s*/uniFB02\s*': 'fl', r'\s*/uniFB00\s*': 'ff',
        r'\s*/uniFB03\s*': 'ffi', r'\s*/uniFB04\s*': 'ffl', r'\s*/uniF642\s*': '%',
        r'\s*/C15\s*': '-', r'\s*/C19\s*': 'e', r'\s*/C20\s*': 'c', r'\s*/C211\s*': ' ',
        r'ﬁ': 'fi', r'ﬂ': 'fl'
    }
    for pattern, fixed in replacements.items(): text = re.sub(pattern, fixed, text)
        
    # 4. Stitch broken words back together
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2\3', text)
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)(?=[a-zA-Z])', r'\1\2', text)
    text = re.sub(r'(?<=[a-zA-Z])(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2', text)
    
    # 5. Fix double spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def process_pdfs():
    print("--- 1. EXTRACTING PDFs ---")
    os.makedirs(MD_DIR, exist_ok=True)
    converter = DocumentConverter()
    pdf_paths = list(Path(PDF_DIR).glob("*.pdf"))
    
    for pdf_path in pdf_paths:
        print(f"Extracting: {pdf_path.name}")
        conv_result = converter.convert(pdf_path)
        final_markdown = clean_and_polish_markdown(conv_result.document.export_to_markdown())
        with open(Path(MD_DIR) / f"{pdf_path.stem}.md", "w", encoding="utf-8") as f:
            f.write(final_markdown)

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
        
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device': 'cpu'})
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