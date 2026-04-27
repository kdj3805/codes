import os
import time
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io
from PIL import Image, ImageEnhance

# -----------------------------
# LangChain
# -----------------------------
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Docling
# -----------------------------
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DocItemLabel

# -----------------------------
# OCR
# -----------------------------
USE_EASYOCR = True
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
# STREAMLIT
# -----------------------------
st.set_page_config(page_title="Correct Multimodal RAG", layout="wide")
st.title("📄 Correct Multimodal RAG (Docling + PyMuPDF)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# -----------------------------
# IMAGE OCR
# -----------------------------
def extract_images_with_ocr(pdf_path):
    images = []
    try:
        reader = easyocr.Reader(["en"], gpu=False) if USE_EASYOCR else RapidOCR()
    except Exception as e:
        st.warning(f"Failed to initialize OCR: {e}")
        return images
    
    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        st.warning(f"Failed to open PDF: {e}")
        return images

    for p in range(len(pdf)):
        page = pdf[p]
        try:
            image_list = page.get_images(full=True)
        except Exception:
            continue
            
        for idx, img in enumerate(image_list):
            try:
                xref = img[0]
                rects = page.get_image_rects(xref)
                y = rects[0].y0 if rects else idx * 100

                img_bytes = pdf.extract_image(xref)["image"]
                pil = Image.open(io.BytesIO(img_bytes)).convert("L")
                pil = ImageEnhance.Contrast(pil).enhance(1.5)

                buf = io.BytesIO()
                pil.save(buf, format="PNG")

                if USE_EASYOCR:
                    text = "\n".join(reader.readtext(buf.getvalue(), detail=0))
                else:
                    result, _ = reader(buf.getvalue())
                    text = " ".join(r[1] for r in result) if result else ""

                if len(text.strip()) < 5:
                    continue

                images.append({
                    "page": p,
                    "y": y,
                    "content": f"""<imagedesc>
Page {p+1}, Image {idx+1}
{text.strip()}
</imagedesc>"""
                })
            except Exception as e:
                # Skip problematic images silently
                continue
    
    pdf.close()
    return images

# -----------------------------
# STRUCTURED SECTION BUILDER
# -----------------------------
def build_structured_sections(docling_doc):
    sections = []
    current = None

    try:
        items = list(docling_doc.iterate_items())
    except Exception as e:
        st.warning(f"Failed to iterate document items: {e}")
        return sections

    for item, _ in items:
        try:
            label = getattr(item, 'label', None)
            if label is None:
                continue
            text = getattr(item, "text", "") or ""
            text = text.strip()

            # Page + position with safe access
            page = 0
            y = 0
            try:
                if item.prov and len(item.prov) > 0:
                    page = getattr(item.prov[0], 'page_no', 1) - 1
                    bbox = getattr(item.prov[0], 'bbox', None)
                    if bbox:
                        y = getattr(bbox, 'y0', None) or getattr(bbox, 't', None) or getattr(bbox, 'top', 0) or 0
            except Exception:
                pass

            # ---- HEADERS (start new section)
            if label in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER} and text:
                if current:
                    sections.append(current)
                current = {
                    "page": page,
                    "y": y,
                    "content": [text]
                }

            # ---- TABLES (atomic)
            elif label == DocItemLabel.TABLE:
                if current:
                    sections.append(current)
                    current = None
                try:
                    table = item.export_to_markdown(doc=docling_doc)
                    if table:
                        sections.append({
                            "page": page,
                            "y": y,
                            "content": [f"<table>\n{table}\n</table>"]
                        })
                except Exception:
                    # Try without doc parameter for older versions
                    try:
                        table = item.export_to_markdown()
                        if table:
                            sections.append({
                                "page": page,
                                "y": y,
                                "content": [f"<table>\n{table}\n</table>"]
                            })
                    except Exception:
                        pass

            # ---- NORMAL TEXT
            elif text:
                if not current:
                    current = {"page": page, "y": y, "content": []}
                current["content"].append(text)
                
        except Exception:
            # Skip problematic items
            continue

    if current:
        sections.append(current)

    return sections

# -----------------------------
# ADAPTIVE CHUNKING
# -----------------------------
def adaptive_chunks(sections):
    docs = []
    for sec in sections:
        try:
            content = sec.get("content", "")
            # Handle both string (images) and list (text sections)
            if isinstance(content, list):
                block = "\n\n".join(str(c) for c in content)
            else:
                block = str(content)
            if block.strip():
                docs.append(Document(page_content=block))
        except Exception:
            continue
    return docs

# -----------------------------
# PIPELINE
# -----------------------------
if uploaded_file and "vectors" not in st.session_state:
    with st.spinner("Processing PDF..."):
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # --- Docling
            converter = DocumentConverter()
            doc = converter.convert("temp.pdf").document

            sections = build_structured_sections(doc)

            # --- Images
            images = extract_images_with_ocr("temp.pdf")
            for img in images:
                sections.append(img)

            sections.sort(key=lambda x: (x.get("page", 0), x.get("y", 0)))

            docs = adaptive_chunks(sections)

            if not docs:
                st.warning("No content extracted from the PDF.")
            else:
                st.session_state.extracted_text = "\n\n".join(d.page_content for d in docs)

                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                st.session_state.vectors = FAISS.from_documents(docs, embeddings)

                st.success(f"✅ {len(docs)} structured chunks created")
                
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# -----------------------------
# SHOW CONTENT
# -----------------------------
if "extracted_text" in st.session_state:
    with st.expander("View extracted content"):
        st.text_area("Extracted", st.session_state.extracted_text, height=450)

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template("""
Answer strictly using the provided context.
If the answer is not present, say "Not found in document".

<context>
{context}
</context>

Question: {input}
""")

# -----------------------------
# QUERY
# -----------------------------
if "vectors" in st.session_state:
    q = st.text_input("Ask a question")

    if q:
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})

        rag = (
            {"context": retriever | (lambda d: "\n\n".join(x.page_content for x in d)),
             "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.write(rag.invoke(q))
