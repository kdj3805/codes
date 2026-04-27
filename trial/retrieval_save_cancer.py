import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.tools import DuckDuckGoSearchRun # <-- NEW: Web Search Tool
from typing import List, Any
from groq import Groq

load_dotenv()

QDRANT_DIR = r"D:\trial\qdrant_db"
JSON_DIR = r"D:\trial\rag_chunks"

def get_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_qdrant_retriever():
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        path=QDRANT_DIR,
        collection_name="medical_papers",
    )
    return qdrant.as_retriever(search_kwargs={"k": 10}) 

def get_bm25_retriever():
    documents = []
    for json_path in Path(JSON_DIR).glob("*_chunks.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                documents.append(Document(page_content=chunk["content"], metadata=chunk))
                
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10 
    return bm25_retriever

def get_hybrid_retriever():
    """Combines Dense and Sparse retrievers using custom Reciprocal Rank Fusion (RRF)."""
    dense_retriever = get_qdrant_retriever()
    sparse_retriever = get_bm25_retriever()
    
    class HybridRetriever(BaseRetriever):
        dense_ret: Any = None
        sparse_ret: Any = None
        
        def _get_relevant_documents(self, query: str) -> List[Document]:
            dense_results = self.dense_ret.invoke(query)
            sparse_results = self.sparse_ret.invoke(query)
            
            scores = {}
            k = 60
            
            for rank, doc in enumerate(dense_results):
                doc_id = doc.metadata.get('chunk_id', str(id(doc)))
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            
            for rank, doc in enumerate(sparse_results):
                doc_id = doc.metadata.get('chunk_id', str(id(doc)))
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            
            all_docs = {doc.metadata.get('chunk_id', str(id(doc))): doc 
                        for doc in dense_results + sparse_results}
            
            sorted_docs = sorted(
                [(doc_id, all_docs[doc_id]) for doc_id in sorted(scores.keys(), key=lambda x: scores[x], reverse=True)],
                key=lambda x: scores[x[0]],
                reverse=True
            )
            
            return [doc for _, doc in sorted_docs]
    
    return HybridRetriever(dense_ret=dense_retriever, sparse_ret=sparse_retriever)

# ==============================================================================
# FALLBACK WEB SEARCH LOGIC
# ==============================================================================
def perform_web_search_fallback(query: str, patient_report: str) -> tuple[str, list]:
    """Triggered only when the local database lacks the answer."""
    search = DuckDuckGoSearchRun()
    
    try:
        # Fetch live data from the internet
        web_context = search.invoke(query)
    except Exception as e:
        web_context = "Web search failed to return results."

    prompt = f"""
    You are an advanced medical AI assistant. 
    The user asked a question that was NOT found in our local medical database.
    We performed a live web search to find the answer.
    
    PATIENT'S MEDICAL REPORT:
    {patient_report if patient_report else "No patient report provided for this session."}
    
    LIVE WEB SEARCH RESULTS:
    {web_context}
    
    USER QUESTION: 
    {query}
    
    INSTRUCTIONS:
    - Answer the user's question using the Live Web Search Results.
    - CRITICAL: You MUST explicitly state at the very beginning of your response that this information was retrieved from a live web search because it was not available in the local clinical database.
    - Answer in plain, easy-to-understand language.
    - Always include a brief medical disclaimer at the end.
    """
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Return the answer and label the source as the web!
    return response.choices[0].message.content, ["Live Web Search (DuckDuckGo)"]

# ==============================================================================
# MAIN GENERATION PIPELINE
# ==============================================================================
def generate_answer(query: str, patient_report: str = "") -> tuple[str, list]:
    retriever = get_hybrid_retriever()
    retrieved_docs = retriever.invoke(query)
    
    top_docs = retrieved_docs[:10] 
    
    context_text = "\n\n".join([
        f"Source: {doc.metadata['source_file']} | Section: {doc.metadata['section_hierarchy']}\n{doc.page_content}" 
        for doc in top_docs
    ])
    
    sources = list(set([doc.metadata['source_file'] for doc in top_docs]))
    
    prompt = f"""
    You are an advanced medical AI assistant integrated into a multi-modal user interface.
    
    PATIENT'S MEDICAL REPORT:
    {patient_report if patient_report else "No patient report provided for this session."}
    
    CLINICAL CONTEXT (From Medical Papers):
    {context_text}
    
    USER QUESTION: 
    {query}
    
    INSTRUCTIONS:
    - Answer the user's question based ONLY on the Clinical Context.
    - If the context contains a "Visual Assets Database" section with relevant images, you MUST include them using the exact tag format: [IMAGE: filename.png]
    - CRITICAL: If the answer to the user's question is NOT found in the Clinical Context, you MUST reply EXACTLY with this secret phrase and absolutely nothing else: [NOT_FOUND]
    - NEVER apologize or say you cannot display images.
    
    EXAMPLE OF CORRECT BEHAVIOR FOR IMAGES:
    User: Can you show me the ABCDE criteria?
    AI: Here is the visual reference for the ABCDE criteria: [IMAGE: melanoma-skin-cancer-review_picture_20.png]
    """
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    
    initial_answer = response.choices[0].message.content
    
    # --- THE ROUTER ---
    # If the LLM spits out our secret code, route the query to the Web Search function!
    if "[NOT_FOUND]" in initial_answer:
        return perform_web_search_fallback(query, patient_report)
    
    # Otherwise, return the normal local answer
    return initial_answer, sources