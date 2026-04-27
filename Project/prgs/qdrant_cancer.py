import os
import json
from pathlib import Path
from dotenv import load_dotenv  # <-- Added this import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from groq import Groq

# Automatically search for a .env file and load the variables into the system
load_dotenv()

def load_chunks_from_json(json_folder="rag_chunks"):
    documents = []
    json_paths = list(Path(json_folder).glob("*_chunks.json"))
    
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        for chunk in chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "source_file": chunk["source_file"],
                    "section_hierarchy": chunk["section_hierarchy"],
                    "chunk_id": chunk["chunk_id"]
                }
            )
            documents.append(doc)
    return documents

def answer_with_llm(query, retrieved_chunks):
    """Passes the retrieved chunks to the LLM to generate a final answer."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing! Please make sure your .env file is set up correctly.")
    
    client = Groq(api_key=key)
    
    # Combine the text from the top chunks into a single context string
    context_text = "\n\n".join([
        f"Source: {doc.metadata['source_file']} | Section: {doc.metadata['section_hierarchy']}\n{doc.page_content}" 
        for doc in retrieved_chunks
    ])
    
    prompt = f"""
    You are a medical AI assistant. Answer the user's question using ONLY the provided context. 
    If the answer is not in the context, say "I don't know based on the provided documents."
    
    Context from Medical Papers:
    {context_text}
    
    User Question: {query}
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def build_and_test_qdrant():
    print("Loading semantic chunks...")
    documents = load_chunks_from_json(json_folder=r"D:\trial\rag_chunks") 
    
    if not documents:
        return
        
    print(f"Loaded {len(documents)} chunks. Initializing BAAI BGE-Base Model...")
    
    # --- UPGRADED EMBEDDING MODEL ---
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'} 
    )
    
    qdrant_storage_path = r"D:\trial\qdrant_db"
    
    print(f"Generating embeddings and building Qdrant DB... Please wait.")

    # Remove previous database files on disk to avoid dimension mismatches.
    if os.path.exists(qdrant_storage_path):
        try:
            import shutil
            shutil.rmtree(qdrant_storage_path)
            print("Removed existing Qdrant storage path to ensure clean rebuild.")
        except Exception as e:
            print(f"Warning: could not delete old Qdrant path: {e}")

    qdrant = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=qdrant_storage_path,
        collection_name="medical_papers",
        force_recreate=True 
    )
    print("✅ Qdrant Database successfully built with BGE-Base embeddings!")

    # --- TESTING THE NEW RAG PIPELINE ---
    query = "What is the most common type of melanoma skin cancer?"
    print(f"\n🔍 Question: {query}")
    
    # 1. Retrieve the top 4 chunks
    results = qdrant.similarity_search(query, k=4)
    
    # 2. Let the LLM read the chunks and answer
    print("\n🧠 Let's see what the LLM says based on the retrieved chunks:")
    print("-" * 50)
    answer = answer_with_llm(query, results)
    print(answer)
    print("-" * 50)

if __name__ == "__main__":
    build_and_test_qdrant()