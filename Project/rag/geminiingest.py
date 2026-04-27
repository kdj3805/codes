import os
import base64
import fitz  # PyMuPDF
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Your required Vision model
GROQ_VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct" 

VISION_PROMPT = """You are an expert data extraction assistant.
I am providing you with an image of a document page.
Your task:
1. Extract ALL the text and tables from this page accurately.
2. Format the output entirely in clean, structural Markdown.
3. For tables: Reconstruct them perfectly. If a cell has multiple lines or bullet points, combine them into a single line within the cell using `<br>` or spaces so the Markdown table formatting does not break.
4. Do not include any conversational filler. Just return the Markdown.
"""

def extract_page_with_vision(b64_image: str, client: Groq) -> str:
    try:
        res = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                    ]
                }
            ]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Error extracting page: {str(e)}"

def process_pdf_vision(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    client = Groq(api_key=GROQ_API_KEY)
    document_data = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        b64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        extracted_md = extract_page_with_vision(b64_image, client)
        document_data.append({"page_number": page_idx + 1, "content": extracted_md})

    doc.close()
    return document_data

def chunk_extracted_data(document_data: list) -> list:
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # Corrected text splitter that preserves markdown tables
    recursive_splitter = RecursiveCharacterTextSplitter.from_language(
        language="markdown", chunk_size=2500, chunk_overlap=250
    )

    final_chunks = []
    for page in document_data:
        page_num = page["page_number"]
        md_header_splits = markdown_splitter.split_text(page["content"])
        
        for split in md_header_splits:
            split.metadata["page"] = page_num
            smaller_chunks = recursive_splitter.split_documents([split])
            final_chunks.extend(smaller_chunks)

    return final_chunks

def ingest_pdf(pdf_path: str, persist_directory: str = "./chroma_db"):
    """Main function to run the ingestion pipeline."""
    # 1. Extract
    extracted_data = process_pdf_vision(pdf_path)
    # 2. Chunk
    chunks = chunk_extracted_data(extracted_data)
    
    # 3. Embed and store in Chroma
    # Using a fast, free open-source embedding model that runs locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return len(chunks)