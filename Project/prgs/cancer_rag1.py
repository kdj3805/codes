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
# Ensure you have GROQ_API_KEY in your .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Using a large context model for medical reasoning
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile" 
# Vision model for extracting tables/charts from PDFs
GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set in .env file.")

# =============================================================================
# MEDICAL VISION PROMPT (Optimized for Regimens & Tables)
# =============================================================================
VISION_PROMPT = """You are a medical data extraction specialist. 
I am providing you with an image of a medical document page (Oncology/Osteosarcoma).

Your task is to convert this page into clean, structural Markdown. 

CRITICAL EXTRACTION RULES:
1. CHEMOTHERAPY REGIMENS: If you see drug acronyms (e.g., MAP, OGS-12) or dosage tables, format them strictly as Markdown Tables.
2. SIDE EFFECTS: Extract side effect lists clearly.
3. HIERARCHY: Use `##` for Section Titles and `###` for Subsections.
4. ATOMICTY: Do not summarize. Extract the exact text, especially for dosage, survival rates, and p-values.
5. CONTEXTUAL HEADERS: If a section is named "Treatment", prepend the parent chapter, e.g., "## Osteosarcoma - Treatment".
6. IGNORE: Headers/Footers with just page numbers.
7. OUPUT: Return ONLY the markdown string.
"""

# =============================================================================
# INGESTION PIPELINE (With Metadata Tagging)
# =============================================================================
MAX_RETRIES = 3

def determine_doc_type(filename):
    """Simple heuristic to classify documents for better retrieval weighting."""
    filename = filename.lower()
    if "booklet" in filename or "guide" in filename or "patient" in filename:
        return "patient_education"
    return "clinical_research"

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
            return {"page_number": page_num, "content": content}
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {"page_number": page_num, "content": f"FAILED_PAGE_{page_num}_ERROR: {str(e)}"}
            time.sleep(2 ** (attempt + 1))
    return {"page_number": page_num, "content": "FAILED_UNKNOWN"}

def process_pdf_vision(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    client = Groq(api_key=GROQ_API_KEY)
    results = []
    # Sequential Processing
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        b64_image = base64.b64encode(img_bytes).decode("utf-8")
        
        if page_idx > 0: time.sleep(1.5) # Rate limit handling
        res = extract_page_with_vision(b64_image, client, page_idx + 1)
        results.append(res)
    doc.close()
    return results

def chunk_extracted_data(document_data: list, source_filename: str) -> list:
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    # Larger chunk size for medical contexts to keep regimens together
    recursive_splitter = RecursiveCharacterTextSplitter.from_language(
        language="markdown", chunk_size=3000, chunk_overlap=300
    )

    final_chunks = []
    doc_type = determine_doc_type(source_filename)

    for page in document_data:
        content = page["content"]
        if "FAILED_" in content: continue

        md_header_splits = markdown_splitter.split_text(content)
        for split in md_header_splits:
            split.metadata["page"] = page["page_number"]
            split.metadata["source"] = source_filename
            split.metadata["doc_type"] = doc_type
            
            smaller_chunks = recursive_splitter.split_documents([split])
            final_chunks.extend(smaller_chunks)
    return final_chunks

def ingest_documents(pdf_files: list[dict], persist_directory: str = "./chroma_db_med") -> tuple[int, list]:
    all_chunks = []
    all_raw = []
    
    for pdf in pdf_files:
        raw_data = process_pdf_vision(pdf["path"])
        all_raw.extend([(pdf["name"], p) for p in raw_data])
        chunks = chunk_extracted_data(raw_data, pdf["name"])
        all_chunks.extend(chunks)

    if all_chunks:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectorstore.add_documents(documents=all_chunks)
        
    return len(all_chunks), all_raw

# =============================================================================
# HYBRID RETRIEVER (With Audience Filtering)
# =============================================================================
class MedicalRetriever:
    def __init__(self, persist_directory: str = "./chroma_db_med"):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        
        # Build BM25
        db_data = self.vectorstore.get()
        docs = [Document(page_content=t, metadata=m) for t, m in zip(db_data["documents"], db_data["metadatas"])]
        if docs:
            self.bm25 = BM25Retriever.from_documents(docs, preprocess_func=lambda x: x.lower().split())
            self.bm25.k = 20
        else:
            self.bm25 = None

    def retrieve(self, query: str, mode: str):
        if not self.bm25: return []
        
        # 1. Semantic Search (Vector)
        chroma_res = self.vectorstore.similarity_search(query, k=20)
        
        # 2. Keyword Search (BM25)
        bm25_res = self.bm25.invoke(query)
        
        # 3. RRF Fusion
        fused = {}
        for rank, doc in enumerate(chroma_res + bm25_res):
            key = doc.page_content
            if key not in fused: fused[key] = {"doc": doc, "score": 0}
            fused[key]["score"] += 1 / (rank + 60)
            
            # BOOOST LOGIC: If mode matches doc_type, boost score
            doc_type = doc.metadata.get("doc_type", "")
            if mode == "Patient" and doc_type == "patient_education":
                fused[key]["score"] += 0.2  # Significant boost
            elif mode == "Clinician" and doc_type == "clinical_research":
                fused[key]["score"] += 0.2

        sorted_docs = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return [x["doc"] for x in sorted_docs[:15]]

    def generate_response(self, query: str, context_docs: list, patient_case: str, mode: str):
        context_text = ""
        for d in context_docs:
            context_text += f"\n--- Source: {d.metadata['source']} (Pg {d.metadata['page']}) ---\n{d.page_content}\n"

        # Specialized System Prompts
        if mode == "Clinician":
            system_role = """You are an expert Oncologist Assistant specializing in Osteosarcoma.
            Target Audience: Oncologists and Residents.
            Tone: Professional, Technical, Evidence-Based.
            
            Specific Directives:
            1. PROTOCOLS: When discussing chemotherapy, mention the MAP regimen (Standard) but explicitly highlight Indian/resource-constrained alternatives (e.g., non-MTX, OGS-12) if relevant context appears.
            2. SYNTHESIS: Use the provided 'Patient Case' to tailor your answer. If the patient has renal issues, warn about Cisplatin/MTX.
            3. JARGON: Use correct medical terminology (e.g., 'necrosis', 'neoadjuvant').
            """
        else:
            system_role = """You are a compassionate Cancer Care Guide.
            Target Audience: Patients and Families.
            Tone: Empathetic, Clear, Non-Alarmist, Simple Language (Grade 8 reading level).
            
            Specific Directives:
            1. EXPLAIN: Deconstruct medical terms. (e.g., explain 'necrosis' as 'tumor cell death - a good sign').
            2. SUPPORT: Focus on practical advice, side-effect management, and survivorship.
            3. DISCLAIMER: Always remind the user to consult their doctor.
            """

        prompt = f"""
        {system_role}

        ### PATIENT CASE / CONTEXT:
        {patient_case if patient_case else "No specific patient case provided. Answer generally."}

        ### KNOWLEDGE BASE CONTEXT:
        {context_text}

        ### USER QUESTION:
        {query}

        ### INSTRUCTIONS:
        Answer the user question based strictly on the Knowledge Base and the Patient Case.
        If the answer is not in the context, state "I cannot find this information in the provided documents."
        Cite your sources at the end.
        """

        res = self.client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return res.choices[0].message.content

# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="OsteoRAG AI", layout="wide")

st.sidebar.title("OsteoRAG Settings")
mode = st.sidebar.radio("User Mode", ["Clinician", "Patient"])
st.sidebar.info("Clinician: Prioritizes Research Papers, Technical Jargon.\n\nPatient: Prioritizes Booklets, Simple Explanations.")

st.title(f"🦴 OsteoRAG Assistant ({mode} Mode)")

if "retriever" not in st.session_state:
    st.session_state.retriever = MedicalRetriever() if os.path.exists("./chroma_db_med") else None

# Input Tabs
tab1, tab2 = st.tabs(["💬 Consultation", "📂 Knowledge Base"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Patient Context")
        patient_case = st.text_area(
            "Paste Synthetic Case Report / Symptoms:",
            height=300,
            placeholder="E.g., 14-year-old male, localized distal femur osteosarcoma. ALP elevated. Post-neoadjuvant chemo pathology shows 90% necrosis. Asking about next steps..."
        )
    
    with col2:
        st.subheader("Query")
        query = st.text_area("Question:", placeholder="E.g., What is the standard protocol vs Indian practice? Or What does 90% necrosis mean?")
        
        if st.button("Analyze & Respond", type="primary"):
            if not st.session_state.retriever:
                st.error("Please ingest documents in the Knowledge Base tab first.")
            elif not query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing case against guidelines..."):
                    docs = st.session_state.retriever.retrieve(query, mode)
                    response = st.session_state.retriever.generate_response(query, docs, patient_case, mode)
                    
                    st.markdown("### AI Analysis")
                    st.write(response)
                    
                    with st.expander("Evidence Sources"):
                        for d in docs:
                            st.caption(f"📄 **{d.metadata['source']}** (Pg {d.metadata['page']})")
                            st.text(d.page_content[:200] + "...")

with tab2:
    st.write("Upload PDF Guidelines (e.g., BCRT Booklet, Research Papers)")
    files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
    if st.button("Process Documents"):
        if files:
            with st.spinner("Reading & indexing medical texts..."):
                file_data = []
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        file_data.append({"path": tmp.name, "name": f.name})
                
                count, _ = ingest_documents(file_data)
                st.session_state.retriever = MedicalRetriever()
                st.success(f"Indexed {count} text blocks.")