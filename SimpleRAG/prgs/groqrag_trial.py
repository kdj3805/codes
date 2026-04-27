import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader

# -----------------------------
# OCR CONFIG
# -----------------------------
USE_EASYOCR = True   # set False to use RapidOCR

if USE_EASYOCR:
    import easyocr
else:
    from rapidocr_onnxruntime import RapidOCR

# -----------------------------
# ENV
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# -----------------------------
# STREAMLIT SETUP
# -----------------------------
st.set_page_config(page_title="Poppler-Free PDF RAG", layout="wide")
st.title("RAG for Complex PDFs")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# -----------------------------
# OCR HELPERS
# -----------------------------
def ocr_with_easyocr(pdf_path):
    reader = easyocr.Reader(["en"])
    results = reader.readtext(pdf_path, detail=0, paragraph=True)
    return "\n".join(results)

def ocr_with_rapidocr(pdf_path):
    ocr = RapidOCR()
    result, _ = ocr(pdf_path)
    return "\n".join([line[1] for line in result])

# -----------------------------
# PDF PROCESSING
# -----------------------------
if uploaded_file and "vectors" not in st.session_state:
    with st.spinner("Processing PDF (Poppler-free)..."):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # 1️⃣ Digital PDF extraction
        loader = UnstructuredPDFLoader(
            "temp.pdf",
            strategy="fast",               # no poppler
            infer_table_structure=True
        )

        docs = loader.load()
        extracted_text = " ".join(d.page_content for d in docs).strip()

        # 2️⃣ OCR fallback for scanned PDFs
        if len(extracted_text) < 100:
            st.warning("Scanned PDF detected — using OCR fallback")

            if USE_EASYOCR:
                extracted_text = ocr_with_easyocr("temp.pdf")
            else:
                extracted_text = ocr_with_rapidocr("temp.pdf")

            docs = [Document(page_content=extracted_text)]

        # ✅ SAVE EXTRACTED CONTENT FOR DISPLAY
        st.session_state.extracted_text = extracted_text

        # 3️⃣ Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # 4️⃣ Embeddings + Vector DB
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vectors = FAISS.from_documents(chunks, embeddings)

        st.success(f"Created {len(chunks)} chunks")

# -----------------------------
# SHOW EXTRACTED CONTENT
# -----------------------------
if "extracted_text" in st.session_state:
    st.subheader("📄 Extracted Content")

    with st.expander("View extracted content"):
        st.text_area(
            label="Extracted text from PDF",
            value=st.session_state.extracted_text,
            height=400
        )

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer strictly using the provided context.
    If the answer is not present, say "Not found in document".

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# -----------------------------
# QUERY
# -----------------------------
query = st.text_input("Ask a question about the document")

if query and "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    start = time.process_time()
    response = rag_chain.invoke(query)
    elapsed = time.process_time() - start

    st.subheader("Answer")
    st.write(response)
    st.caption(f"{elapsed:.2f}s")

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(retriever.invoke(query)):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.page_content)
            st.divider()
