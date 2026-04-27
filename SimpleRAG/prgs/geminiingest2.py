import os
import asyncio
import base64
import fitz  # PyMuPDF
from dotenv import load_dotenv
from groq import AsyncGroq
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# ── DO NOT MODIFY ──────────────────────────────────────────────────────────────
VISION_PROMPT = """You are an expert data extraction assistant.
I am providing you with an image of a document page.

Your task:
1. Extract ALL the text and tables from this page accurately.
2. Format the output entirely in clean, structural Markdown.
3. CRITICAL MANDATE FOR HEADERS: You MUST enforce a strict parent-child hierarchy. 
   - Use `##` (Header 2) ONLY for the main Best Practice titles (e.g., `## Best Practice #4: Restrict Device Features as Necessary`).
   - Use `###` (Header 3) ONLY for the subsections under a Best Practice (e.g., `### Our Recommendations` and `### How MaaS360 Helps`). 
   This strict nesting is required so the downstream text chunker does not lose context.
4. For tables: Reconstruct them perfectly. If a cell has multiple lines or bullet points, combine them into a single line within the cell using `<br>` or spaces so the Markdown table formatting does not break.
5. Do not include any conversational filler. Just return the Markdown.
"""
# ──────────────────────────────────────────────────────────────────────────────

# Goal 1: Semaphore to cap concurrent Groq Vision API calls
CONCURRENCY_LIMIT = 7


async def extract_page_with_vision_async(
    b64_image: str,
    client: AsyncGroq,
    semaphore: asyncio.Semaphore,
    page_num: int,
) -> dict:
    """Async wrapper: acquires the semaphore, fires the Groq Vision call, returns result dict."""
    async with semaphore:
        try:
            res = await client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                            },
                        ],
                    }
                ],
            )
            content = res.choices[0].message.content.strip()
        except Exception as e:
            content = f"Error extracting page: {str(e)}"

        return {"page_number": page_num, "content": content}


async def process_pdf_vision_async(pdf_path: str) -> list:
    """
    Render every PDF page to PNG, then blast all Groq Vision calls
    concurrently (bounded by CONCURRENCY_LIMIT).
    """
    doc = fitz.open(pdf_path)
    client = AsyncGroq(api_key=GROQ_API_KEY)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    tasks = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        b64_image = base64.b64encode(img_bytes).decode("utf-8")

        tasks.append(
            extract_page_with_vision_async(b64_image, client, semaphore, page_idx + 1)
        )

    # Fire all tasks concurrently; results arrive in completion order
    results = await asyncio.gather(*tasks)

    doc.close()
    await client.close()

    # Re-sort by page number so chunks stay in document order
    results.sort(key=lambda r: r["page_number"])
    return results


# ── DO NOT MODIFY (stateful header tracking logic) ────────────────────────────
def chunk_extracted_data(document_data: list, source_filename: str) -> list:
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    recursive_splitter = RecursiveCharacterTextSplitter.from_language(
        language="markdown", chunk_size=2500, chunk_overlap=250
    )

    final_chunks = []
    for page in document_data:
        page_num = page["page_number"]
        md_header_splits = markdown_splitter.split_text(page["content"])

        for split in md_header_splits:
            # Goal 2: inject source filename + page number into every chunk
            split.metadata["page"] = page_num
            split.metadata["source"] = source_filename

            smaller_chunks = recursive_splitter.split_documents([split])
            final_chunks.extend(smaller_chunks)

    return final_chunks
# ──────────────────────────────────────────────────────────────────────────────


def ingest_documents(pdf_files: list[dict], persist_directory: str = "./chroma_db") -> int:
    """
    Goal 2: Accept a list of dicts  {"path": str, "name": str}  representing
    one or more PDFs, process each asynchronously, and embed all chunks together.
    """
    all_chunks: list[Document] = []

    for pdf_info in pdf_files:
        pdf_path: str = pdf_info["path"]
        source_name: str = pdf_info["name"]

        # Goal 1: run the async pipeline in a clean event loop
        extracted_data = asyncio.run(process_pdf_vision_async(pdf_path))

        # Goal 2: pass the filename so every chunk carries source metadata
        chunks = chunk_extracted_data(extracted_data, source_filename=source_name)
        all_chunks.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return len(all_chunks)


# ── Thin backwards-compat shim (single file, used by legacy callers) ──────────
def ingest_pdf(pdf_path: str, persist_directory: str = "./chroma_db") -> int:
    filename = os.path.basename(pdf_path)
    return ingest_documents(
        [{"path": pdf_path, "name": filename}],
        persist_directory=persist_directory,
    )