import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_TEXT_MODEL = "llama-3.1-8b-instant"


class HybridRetriever:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

        db_data = self.vectorstore.get()
        docs = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(db_data["documents"], db_data["metadatas"])
        ]

        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = 10
        else:
            self.bm25_retriever = None

    # ── DO NOT MODIFY ──────────────────────────────────────────────────────────
    def reciprocal_rank_fusion(self, bm25_results: list, chroma_results: list, k=60):
        fused_scores = {}

        def add_to_fusion(results):
            for rank, doc in enumerate(results):
                doc_hash = doc.page_content
                if doc_hash not in fused_scores:
                    fused_scores[doc_hash] = {"score": 0, "doc": doc}
                fused_scores[doc_hash]["score"] += 1 / (rank + 1 + k)

        add_to_fusion(bm25_results)
        add_to_fusion(chroma_results)

        reranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in reranked[:10]]
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(self, query: str):
        if not self.bm25_retriever:
            return []

        chroma_results = self.vectorstore.similarity_search(query, k=10)
        bm25_results = self.bm25_retriever.invoke(query)

        fused_docs = self.reciprocal_rank_fusion(bm25_results, chroma_results)
        return fused_docs

    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        # Goal 3: include filename in the context block fed to the LLM
        context_blocks = []
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            context_blocks.append(
                f"Source: {source} (Page {page}):\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        # Goal 3: strict citation instruction in the generation prompt
        prompt = f"""You are an intelligent enterprise assistant.
Use the following retrieved markdown documentation to answer the user's question.
If the answer is contained in a table, explain it clearly.
If the answer is not in the context, say "I don't have enough information."

CRITICAL CITATION RULE: Every factual statement you make MUST be followed immediately
by a citation in this exact format: [Source: <filename>, Page: <number>]
Example: "Devices must be enrolled before they can access corporate email. [Source: IBMMaaS360.pdf, Page: 8]"
Never group all citations at the end. Cite inline, after each individual fact.

Context:
{context}

Question: {query}
Answer:"""

        res = self.client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content.strip()