import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Using a text-optimized model for generating the final answer
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile" 

class HybridRetriever:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 1. Load ChromaDB
        self.vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=self.embeddings
        )
        
        # 2. Load BM25
        # We fetch all documents currently stored in Chroma to initialize BM25
        db_data = self.vectorstore.get()
        docs = [Document(page_content=txt, metadata=meta) 
                for txt, meta in zip(db_data['documents'], db_data['metadatas'])]
        
        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = 5 # Retrieve top 5 keyword matches
        else:
            self.bm25_retriever = None

    def reciprocal_rank_fusion(self, bm25_results: list, chroma_results: list, k=60):
        """Merges results using Reciprocal Rank Fusion."""
        fused_scores = {}
        
        # Helper to process rankings
        def add_to_fusion(results):
            for rank, doc in enumerate(results):
                doc_hash = doc.page_content # Use content as unique ID
                if doc_hash not in fused_scores:
                    fused_scores[doc_hash] = {"score": 0, "doc": doc}
                fused_scores[doc_hash]["score"] += 1 / (rank + 1 + k)

        add_to_fusion(bm25_results)
        add_to_fusion(chroma_results)
        
        # Sort by highest RRF score
        reranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # Return top 5 fused documents
        return [item["doc"] for item in reranked[:5]]

    def retrieve(self, query: str):
        if not self.bm25_retriever:
            return []

        # 1. Get Vector Similarity Results
        chroma_results = self.vectorstore.similarity_search(query, k=5)
        
        # 2. Get BM25 Keyword Results
        bm25_results = self.bm25_retriever.invoke(query)
        
        # 3. Merge with RRF
        fused_docs = self.reciprocal_rank_fusion(bm25_results, chroma_results)
        return fused_docs

    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        # Construct the context string from the retrieved Markdown chunks
        context = "\n\n---\n\n".join(
            [f"Source (Page {doc.metadata.get('page', 'Unknown')}):\n{doc.page_content}" 
             for doc in retrieved_docs]
        )
        
        prompt = f"""You are an intelligent enterprise assistant. 
Use the following retrieved markdown documentation to answer the user's question. 
If the answer is contained in a table, explain it clearly. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}
Answer:"""

        res = self.client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()