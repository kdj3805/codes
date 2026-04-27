import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Any
from groq import Groq

load_dotenv()

QDRANT_DIR = r"D:\trial\qdrant_db"
JSON_DIR = r"D:\trial\rag_chunks"

def get_embeddings():
    """Initializes BGE model with normalization and query instructions."""
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
        # Note: HuggingFaceBgeEmbeddings automatically injects the required 
        # "Represent this sentence..." prefix into your queries under the hood!
    )

def get_qdrant_retriever():
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        path=QDRANT_DIR,
        collection_name="medical_papers",
    )
    # Increased k from 4 to 10 to cast a wider net
    return qdrant.as_retriever(search_kwargs={"k": 10}) 

def get_bm25_retriever():
    documents = []
    for json_path in Path(JSON_DIR).glob("*_chunks.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                documents.append(Document(page_content=chunk["content"], metadata=chunk))
                
    bm25_retriever = BM25Retriever.from_documents(documents)
    # Increased k from 4 to 10
    bm25_retriever.k = 10 
    return bm25_retriever

def get_hybrid_retriever():
    """Combines Dense and Sparse retrievers using Reciprocal Rank Fusion (RRF)."""
    dense_retriever = get_qdrant_retriever()
    sparse_retriever = get_bm25_retriever()
    
    # Custom hybrid retriever using RRF
    class HybridRetriever(BaseRetriever):
        dense_ret: Any = None
        sparse_ret: Any = None
        
        def _get_relevant_documents(self, query: str) -> List[Document]:
            # Get results from both retrievers
            dense_results = self.dense_ret.invoke(query)
            sparse_results = self.sparse_ret.invoke(query)
            
            # RRF scoring: 1/(k + rank)
            scores = {}
            k = 60
            
            for rank, doc in enumerate(dense_results):
                doc_id = doc.metadata.get('chunk_id', str(id(doc)))
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            
            for rank, doc in enumerate(sparse_results):
                doc_id = doc.metadata.get('chunk_id', str(id(doc)))
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            
            # Combine results and sort by RRF score
            all_docs = {doc.metadata.get('chunk_id', str(id(doc))): doc 
                        for doc in dense_results + sparse_results}
            
            sorted_docs = sorted(
                [(doc_id, all_docs[doc_id]) for doc_id in sorted(scores.keys(), key=lambda x: scores[x], reverse=True)],
                key=lambda x: scores[x[0]],
                reverse=True
            )
            
            return [doc for _, doc in sorted_docs]
    
    return HybridRetriever(dense_ret=dense_retriever, sparse_ret=sparse_retriever)

def generate_answer(query: str, patient_report: str = "") -> tuple[str, list]:
    """
    Takes a user query and an optional patient report, retrieves clinical context, 
    calls Groq, and returns the translated, empathetic answer.
    """
    retriever = get_hybrid_retriever()
    
    # We retrieve documents based on the user's question
    # (Optional enhancement: you could also embed the patient_report to search the DB, 
    # but for now, searching via the user's question is safer and more precise).
    retrieved_docs = retriever.invoke(query)
    
    # Send the top 10 fused chunks to the LLM
    top_docs = retrieved_docs[:10] 
    
    context_text = "\n\n".join([
        f"Source: {doc.metadata['source_file']} | Section: {doc.metadata['section_hierarchy']}\n{doc.page_content}" 
        for doc in top_docs
    ])
    
    sources = list(set([doc.metadata['source_file'] for doc in top_docs]))
    
    # --- UPGRADED CLINICAL PROMPT ---
    prompt = f"""
    You are an empathetic, highly intelligent medical AI assistant. Your goal is to help the user understand their medical condition.
    
    You have access to two things:
    1. The patient's actual medical report/notes (if provided).
    2. Clinical context retrieved from highly vetted oncology and dermatology review papers.
    
    PATIENT'S MEDICAL REPORT:
    {patient_report if patient_report else "No patient report provided for this session."}
    
    CLINICAL CONTEXT (From Medical Papers):
    {context_text}
    
    USER QUESTION: 
    {query}
    
    INSTRUCTIONS:
    - Answer the user's question in plain, easy-to-understand language. Avoid overly dense medical jargon unless you immediately explain what it means.
    - If the user provides a medical report, use the Clinical Context to explain what the findings in their report actually mean.
    - If the answer is not in the Clinical Context, say "I don't have enough information in the provided literature to answer that."
    - IMPORTANT: Always include a brief, gentle medical disclaimer at the end of your response stating that you are an AI and they should consult their oncologist or primary care physician.
    """
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1, # Keep it low for factual accuracy
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content, sources