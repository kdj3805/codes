# ============================================================
# VISION-FIRST PIPELINE: PAGE-AS-IMAGE EXTRACTION
# Uses PyMuPDF (Rendering) + Groq Vision (Llama 3.2 90B)
# ============================================================

import os
import base64
import fitz  # PyMuPDF
import streamlit as st
import json
from dotenv import load_dotenv
from groq import Groq

# =========================
# ENV & INIT
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Use Groq's Vision Model
GROQ_VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct" 

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# =========================
# SYSTEM PROMPT
# =========================
# We ask for Markdown because LLMs are excellent at converting 
# multi-line table cells into clean, valid Markdown tables 
# (e.g., using <br> for line breaks), which makes chunking easy.
VISION_PROMPT = """You are an expert data extraction assistant.
I am providing you with an image of a document page.

Your task:
1. Extract ALL the text and tables from this page accurately.
2. Format the output entirely in clean, structural Markdown.
3. For tables: Reconstruct them perfectly. If a cell has multiple lines or bullet points, combine them into a single line within the cell using `<br>` or spaces so the Markdown table formatting does not break.
4. Do not include any conversational filler (e.g., "Here is the extracted text"). Just return the Markdown.
"""

# =========================
# VISION EXTRACTION
# =========================
def extract_page_with_vision(b64_image: str, client: Groq) -> str:
    """Passes the base64 image to Groq Vision for extraction."""
    try:
        res = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            temperature=0.0, # Zero for maximum factual adherence
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Error extracting page: {str(e)}"

# =========================
# MAIN PIPELINE
# =========================
def process_pdf_vision(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    client = Groq(api_key=GROQ_API_KEY)
    
    document_data = []

    # Iterate through each page
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        
        # 1. Render the page as an image (Pixmap)
        # dpi=150 is a sweet spot: high enough for the LLM to read small text, 
        # but keeps the payload small enough for the API limit.
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        
        # 2. Convert to Base64
        b64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        # 3. Send to Groq Vision
        extracted_md = extract_page_with_vision(b64_image, client)
        
        # 4. Store the structured result
        document_data.append({
            "page_number": page_idx + 1,
            "content": extracted_md
        })

    doc.close()
    
    # Return as a structured JSON array where each node is a page's Markdown
    return json.dumps(document_data, indent=2)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config("Vision PDF Extractor", layout="wide")
st.title("👁️ Vision-Based PDF Extraction")
st.write("Converts pages to images and uses Groq Vision to perfectly reconstruct complex layouts.")

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    with st.spinner("Rendering pages and analyzing with Groq Vision..."):
        json_output = process_pdf_vision("temp.pdf")

    st.success("Vision Extraction completed!")

    st.download_button(
        label="Download Vision-Extracted JSON",
        data=json_output,
        file_name="vision_extracted_document.json",
        mime="application/json"
    )

    st.subheader("Preview")
    
    # Load JSON to display nice markdown tabs
    data = json.loads(json_output)
    tabs = st.tabs([f"Page {d['page_number']}" for d in data])
    
    for i, tab in enumerate(tabs):
        with tab:
            st.markdown(data[i]["content"])

    os.remove("temp.pdf")