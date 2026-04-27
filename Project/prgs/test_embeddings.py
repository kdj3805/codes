import sys
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# CONFIGURATION
PDF_FILE = "D:\\trial\\data\\Lecture_Notes_Unit_2.pdf"
MODEL_NAME = "nomic-embed-text" # Ensure 'ollama run llama3' is active

def main():
    # 1. LOAD & SPLIT (The Cleanest Method)
    print("PAGE 1: Processing Text...")
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"   -> Created {len(chunks)} text chunks.")

    # 2. EMBED (The "Magic" Step)
    print(f"PAGE 2: Generating Embeddings with {MODEL_NAME}...")
    print("   (This might take a minute on your laptop CPU...)")
    
    start_time = time.time()
    
    # Initialize the Embedding Model
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    
    # Create the Vector Store (This does the heavy lifting)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    end_time = time.time()
    print(f"   -> Finished embedding in {end_time - start_time:.2f} seconds.")

    # 3. TEST THE SEARCH (No LLM yet, just Math)
    query = "What happened to Gregor?"
    print(f"\nPAGE 3: Testing Retrieval for query: '{query}'")
    
    # Ask the database for the 2 most similar chunks
    results = vector_store.similarity_search(query, k=2)
    
    print(f"   -> Found {len(results)} matches.\n")
    
    for i, res in enumerate(results):
        print(f"--- MATCH {i+1} ---")
        print(f"Content: {res.page_content.replace(chr(10), ' ')[:200]}...") # Preview first 200 chars
        print("----------------")

if __name__ == "__main__":
    main()