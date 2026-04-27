import os
import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def chunk_markdown_files(input_folder="markdown_papers", output_folder="rag_chunks"):
    """
    Reads markdown files, semantically splits them by headers, 
    and saves them as structured JSON ready for Vector Database ingestion.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Define the Markdown headers we want to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # Initialize the semantic markdown splitter
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 2. Initialize a fallback text splitter for oversized sections
    # Chunk size of 1000 characters with 150 characters of overlap is standard for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150 
    )

    # Find all polished markdown files
    md_files = list(Path(input_folder).glob("*.md"))
    
    if not md_files:
        print(f"No Markdown files found in {input_folder}")
        return

    print(f"Starting semantic chunking for {len(md_files)} files...")
    print("-" * 50)
    
    total_chunks_created = 0
    
    for md_path in md_files:
        print(f"✂️  Chunking: {md_path.name}")
        
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
            
        # Split semantically by Headers first
        header_splits = markdown_splitter.split_text(markdown_content)
        
        # Apply the fallback character splitting to any chunks that are still too large
        final_chunks = text_splitter.split_documents(header_splits)
        
        # Format into a clean JSON structure
        structured_payload = []
        for i, chunk in enumerate(final_chunks):
            # Extract the headers that LangChain attached to this chunk
            metadata = chunk.metadata
            
            # Build a hierarchical context string (e.g., "Header 1 > Header 2")
            # We filter out non-header metadata just in case
            hierarchy_list = [v for k, v in metadata.items() if k.startswith("Header")]
            hierarchy_str = " > ".join(hierarchy_list) if hierarchy_list else "Body / Introduction"
            
            structured_payload.append({
                "chunk_id": f"{md_path.stem}_chunk_{i+1}",
                "source_file": md_path.name,
                "section_hierarchy": hierarchy_str,
                "content": chunk.page_content,
                "char_count": len(chunk.page_content)
            })
            
        # Save the chunked payload for this specific paper
        output_filepath = Path(output_folder) / f"{md_path.stem}_chunks.json"
        
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(structured_payload, f, indent=2, ensure_ascii=False)
            
        print(f"   ✅ Created {len(final_chunks)} chunks. Saved to {output_filepath.name}")
        total_chunks_created += len(final_chunks)

    print("-" * 50)
    print(f"🎉 Process complete! Created a total of {total_chunks_created} chunks.")

if __name__ == "__main__":
    # Ensure these paths match where your files are actually located
    INPUT_DIR = r"D:\trial\markdown_papers"
    OUTPUT_DIR = r"D:\trial\rag_chunks"
    
    chunk_markdown_files(input_folder=INPUT_DIR, output_folder=OUTPUT_DIR)