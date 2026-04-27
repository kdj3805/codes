import sys
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# CONFIGURATION
PDF_FILE = "D:\\trial\\data\\Metamorphosis.pdf" # Make sure this file exists

def print_chunk_sample(name, chunks):
    """Helper to visualize the first 2 chunks of a strategy"""
    print(f"\n{'='*20} STRATEGY: {name} {'='*20}")
    print(f" Total Chunks Created: {len(chunks)}")
    print(f" Average Length: {int(sum(len(c.page_content) for c in chunks) / len(chunks))} characters\n")

    # Print first 2 chunks only to keep it readable
    for i, chunk in enumerate(chunks[:2]):
        print(f"--- [CHUNK {i+1}] ---")
        # Replace newlines with spaces for cleaner visualization
        content_preview = chunk.page_content.replace('\n', ' ')
        print(f"{content_preview}\n")
    print("-" * 60)

def main():
    # 1. LOAD PDF
    if not os.path.exists(PDF_FILE):
        print(f" Error: '{PDF_FILE}' not found.")
        return

    print(f" Loading {PDF_FILE}...")
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()

    # 2. DEFINE STRATEGIES
    
    # Strategy A: Character Splitter (Hard cut)
    # Cuts strictly at 500 chars. Often breaks mid-sentence.
    char_splitter = CharacterTextSplitter(
        separator="", 
        chunk_size=500, 
        chunk_overlap=0
    )

    # Strategy B: Recursive Splitter (Smart cut)
    # Tries to cut at paragraphs (\n\n), then sentences (\n), then spaces.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Strategy C: Token Splitter (LLM Math)
    # Cuts by tokens. 120 tokens is approx 400-500 characters.
    token_splitter = TokenTextSplitter(
        chunk_size=120,
        chunk_overlap=20
    )

    # 3. EXECUTE & COMPARE
    
    # Run Strategy A
    chunks_char = char_splitter.split_documents(docs)
    print_chunk_sample("1. Character Splitter (Brute Force)", chunks_char)

    # Run Strategy B
    chunks_rec = recursive_splitter.split_documents(docs)
    print_chunk_sample("2. Recursive Splitter (Recommended)", chunks_rec)

    # Run Strategy C
    chunks_tok = token_splitter.split_documents(docs)
    print_chunk_sample("3. Token Splitter (LLM Native)", chunks_tok)

if __name__ == "__main__":
    main()