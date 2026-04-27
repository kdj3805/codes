import os
import time
import base64
import tempfile
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# =============================================================================
# INIT & ENV
# =============================================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Vision-capable model on Groq
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"  # Reliable text model on Groq

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set in .env file.")

# =============================================================================
# VISION PROMPT (Strict Anti-Hallucination & Formatting)
# =============================================================================
VISION_PROMPT = """You are an expert data extraction assistant building a RAG database. 
I am providing you with an image of a document page.

Your task is to convert this page into clean, structural Markdown. 

CRITICAL EXTRACTION RULES:
1. TABLE OF CONTENTS BAN: If this page is a Table of Contents or Index, DO NOT extract the text. Simply output: `<TOC_OMITTED>`. 
2. STRICT HEADER HIERARCHY: 
   - Use `##` (Header 2) ONLY for main document titles or primary sections (e.g., "## Best Practice #4").
   - Use `###` (Header 3) for sub-sections.
3. CONTEXTUAL SUBHEADERS (MANDATORY): If a sub-section has a generic title like "Our Recommendations", you MUST prepend the parent section name.
   - BAD: `### Our Recommendations`
   - GOOD: `### Best Practice #4 - Our Recommendations`
4. TABLE RECONSTRUCTION: Reconstruct tables perfectly using Markdown table syntax. Use `<br>` for multi-line cells.
5. NO CONVERSATIONAL FILLER: Return ONLY the markdown string.
"""

# =============================================================================
# SYNCHRONOUS INGESTION PIPELINE (Safe for Free-Tier Limits)
# =============================================================================
MAX_RETRIES = 3

def extract_page_with_vision(b64_image: str, client: Groq, page_num: int) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            res = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                        ],
                    }
                ],
            )
            content = res.choices[0].message.content.strip()
            # Clean HTML <br> tags so they don't leak into LLM answers
            content = content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
            return {"page_number": page_num, "content": content}
        
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {"page_number": page_num, "content": f"FAILED_PAGE_{page_num}_ERROR: {str(e)}"}
            time.sleep(2 ** (attempt + 1)) # Backoff before retry
    
    return {"page_number": page_num, "content": "FAILED_UNKNOWN"}

def process_pdf_vision(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    client = Groq(api_key=GROQ_API_KEY)
    results = []

    # Sequential Processing to prevent 429 Rate Limit Errors
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        b64_image = base64.b64encode(img_bytes).decode("utf-8")
        
        # Pacing: Wait 2 seconds between pages to respect Groq's ~30 Requests/Min limit
        if page_idx > 0:
            time.sleep(2)
            
        res = extract_page_with_vision(b64_image, client, page_idx + 1)
        results.append(res)

    doc.close()
    return results

def chunk_extracted_data(document_data: list, source_filename: str) -> list:
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    recursive_splitter = RecursiveCharacterTextSplitter.from_language(
        language="markdown", chunk_size=2500, chunk_overlap=250
    )

    final_chunks = []
    active_headers = {} 

    for page in document_data:
        page_num = page["page_number"]
        content = page["content"]

        if "FAILED_PAGE" in content:
            print(f" Warning: Skipping failed page {page_num}")
            continue
        if "<TOC_OMITTED>" in content:
            continue

        md_header_splits = markdown_splitter.split_text(content)

        for split in md_header_splits:
            new_h1 = split.metadata.get("Header 1")
            new_h2 = split.metadata.get("Header 2")
            new_h3 = split.metadata.get("Header 3")
            
            if new_h1:
                active_headers["Header 1"] = new_h1
                active_headers.pop("Header 2", None)
                active_headers.pop("Header 3", None)
            if new_h2:
                active_headers["Header 2"] = new_h2
                active_headers.pop("Header 3", None)
            if new_h3:
                active_headers["Header 3"] = new_h3

            split.metadata.update(active_headers)
            split.metadata["page"] = page_num
            split.metadata["source"] = source_filename

            smaller_chunks = recursive_splitter.split_documents([split])
            final_chunks.extend(smaller_chunks)

    return final_chunks

def get_ingested_sources(persist_directory: str = "./chroma_db") -> set:
    """Return the set of source filenames already stored in ChromaDB."""
    if not os.path.exists(persist_directory):
        return set()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    db_data = vectorstore.get(include=["metadatas"])
    return {m["source"] for m in (db_data.get("metadatas") or []) if m and m.get("source")}


def ingest_documents(pdf_files: list[dict], persist_directory: str = "./chroma_db") -> tuple[int, list, list]:
    already_ingested = get_ingested_sources(persist_directory)
    all_chunks: list[Document] = []
    all_raw_data = []
    skipped_files: list[str] = []

    for pdf_info in pdf_files:
        pdf_path: str = pdf_info["path"]
        source_name: str = pdf_info["name"]

        # Skip files that are already in ChromaDB
        if source_name in already_ingested:
            skipped_files.append(source_name)
            continue

        extracted_data = process_pdf_vision(pdf_path)
        all_raw_data.extend([(source_name, page) for page in extracted_data])

        chunks = chunk_extracted_data(extracted_data, source_filename=source_name)
        all_chunks.extend(chunks)

    if all_chunks:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectorstore.add_documents(documents=all_chunks)

    return len(all_chunks), all_raw_data, skipped_files

# =============================================================================
# RETRIEVAL (Widened to catch Dense Keywords)
# =============================================================================
class HybridRetriever:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)

        db_data = self.vectorstore.get()
        docs = [Document(page_content=txt, metadata=meta) for txt, meta in zip(db_data["documents"], db_data["metadatas"])]

        if docs:
            # Case-insensitive tokenization so "critical" matches "Critical"
            self.bm25_retriever = BM25Retriever.from_documents(
                docs, preprocess_func=lambda text: text.lower().split()
            )
            self.bm25_retriever.k = 25  # Widened from 10 to 25
        else:
            self.bm25_retriever = None

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
        return [item["doc"] for item in reranked[:20]]  # Returning Top 20 context blocks to the LLM

    def retrieve(self, query: str):
        if not self.bm25_retriever: return []
        chroma_results = self.vectorstore.similarity_search(query, k=25)  # Widened from 10 to 25
        bm25_results = self.bm25_retriever.invoke(query)
        return self.reciprocal_rank_fusion(bm25_results, chroma_results)

    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        context_blocks = []
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            # Extract headers to give the LLM explicit structural context
            h1 = doc.metadata.get("Header 1", "")
            h2 = doc.metadata.get("Header 2", "")
            h3 = doc.metadata.get("Header 3", "")
            headers = " > ".join(filter(None, [h1, h2, h3]))
            header_context = f"Section: {headers}\n" if headers else ""
            
            # Clean <br> tags from content (may exist in already-ingested chunks)
            clean_content = doc.page_content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

            context_blocks.append(
                f"--- Source: {source} (Page {page}) ---\n"
                f"{header_context}{clean_content}"
            )
            
        context = "\n\n".join(context_blocks)

        prompt = f"""You are a highly capable enterprise AI assistant. Your goal is to answer the user's question accurately and comprehensively based ONLY on the provided Context.

Context blocks are provided below. Each block indicates its source file, page number, and document section.

INSTRUCTIONS:
1. Analyze the Context carefully to find the answer.
2. If the answer is found, provide a clear, detailed, and comprehensive response. Extract all relevant steps or recommendations. Make it readable and flow naturally.
3. CITATION STYLE: DO NOT clutter your answer by putting a citation after every single sentence. Instead, write your full response normally. At the very end of your response, add a single line that says "Sources used: [Filename, Page X]". If you are combining facts from multiple different pages, you may use very brief, subtle inline markers like (Page 8) to distinguish them, but keep it clean.
4. ANTI-HALLUCINATION: If the provided Context does NOT contain the answer to the user's question, you must reply EXACTLY with: "I don't have enough information." Do not attempt to guess or fabricate an answer.

Context:
{context}

Question: {query}
Answer:"""

        res = self.client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content.strip()

# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title(" RAG Pipeline:")

if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever() if os.path.exists("./chroma_db") else None
if "raw_data" not in st.session_state:
    st.session_state.raw_data = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

tab_ingest, tab_chat, tab_debug = st.tabs(["Ingest", "Chat", "Debug Extraction"])

with tab_ingest:
    st.header("Ingest Documents")
    pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    # Store uploaded files in session_state immediately so they survive the button re-run
    if pdf_files:
        st.session_state.uploaded_files = pdf_files

    if st.button("Process & Embed"):
        files_to_process = st.session_state.get("uploaded_files", [])
        if files_to_process:
            with st.spinner(f"Processing {len(files_to_process)} file(s) — skipping already-ingested..."):
                pdf_infos, tmp_paths = [], []
                for f in files_to_process:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(f.getvalue())
                    tmp.close()
                    tmp_paths.append(tmp.name)
                    pdf_infos.append({"path": tmp.name, "name": f.name})

                try:
                    num_chunks, raw_data, skipped = ingest_documents(pdf_infos)
                    st.session_state.raw_data.extend(raw_data)
                    st.session_state.retriever = HybridRetriever()
                    if num_chunks:
                        st.success(f"Indexed {num_chunks} new chunks!")
                    if skipped:
                        st.info(f"Skipped (already ingested): {', '.join(skipped)}")
                    if not num_chunks and not skipped:
                        st.warning("No new files to process.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    for p in tmp_paths:
                        try: os.remove(p)
                        except: pass
        else:
            st.warning("Please upload at least one PDF file first.")

with tab_chat:
    if st.session_state.retriever:
        query = st.text_input("Ask a question:")
        if st.button("Generate Answer") and query:
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.retrieve(query)
                ans = st.session_state.retriever.generate_answer(query, docs)
            st.markdown("### Answer")
            st.write(ans)
            with st.expander("View Sources"):
                for d in docs:
                    st.markdown(f"**{d.metadata.get('source')} (Pg {d.metadata.get('page')})**")
                    st.text(d.page_content[:300] + "...")
                    st.divider()
    else:
        st.info("Upload a document in the Ingest tab to start.")

with tab_debug:
    st.write("### Raw Extracted Markdown (Check for Errors)")
    if st.session_state.raw_data:
        for filename, page in st.session_state.raw_data:
            with st.expander(f"{filename} - Page {page['page_number']}"):
                if "FAILED" in page['content']:
                    st.error(page['content'])
                else:
                    st.code(page['content'], language="markdown")
    else:
        st.write("No documents processed yet.")