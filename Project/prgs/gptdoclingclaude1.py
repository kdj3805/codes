
import os
import streamlit as st
from dotenv import load_dotenv
import fitz
from PIL import Image
import io
from typing import List, Dict, Any, Tuple
from groq import Groq

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

USE_EASYOCR = True

if USE_EASYOCR:
    import easyocr
else:
    from rapidocr_onnxruntime import RapidOCR

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

IMAGE_DESCRIPTION_PROMPT = """You are analyzing OCR text extracted from a screenshot or diagram in an enterprise policy document.

Based ONLY on the OCR text provided below, generate a structured description following these rules:

1. Identify what type of screen or interface this is (e.g., configuration panel, policy settings screen, diagram, table)
2. List ALL visible fields, settings, options, or labels you can identify from the OCR text
3. Explain the purpose of this screen/diagram in the context of enterprise policy management
4. If the OCR text is unclear, incomplete, or illegible, explicitly state: "OCR text is unclear or incomplete"
5. Do NOT make assumptions about content not present in the OCR text
6. Do NOT add information beyond what the OCR text shows

OCR Text:
{ocr_text}

Provide a concise, factual description in 3-5 sentences."""


def extract_images_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    image_items = []
    
    if USE_EASYOCR:
        reader = easyocr.Reader(["en"], gpu=False)
    else:
        ocr_engine = RapidOCR()
    
    pdf_doc = fitz.open(pdf_path)
    
    for page_idx in range(len(pdf_doc)):
        page = pdf_doc[page_idx]
        image_list = page.get_images(full=True)
        
        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                
                img_rects = page.get_image_rects(xref)
                if img_rects and len(img_rects) > 0:
                    y_position = float(img_rects[0].y0)
                else:
                    y_position = float(img_idx * 100)
                
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    if pil_image.mode not in ('RGB', 'L'):
                        pil_image = pil_image.convert('RGB')
                    
                    min_dimension = 300
                    if pil_image.width < min_dimension or pil_image.height < min_dimension:
                        scale = max(min_dimension / pil_image.width, min_dimension / pil_image.height)
                        new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
                        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Contrast(pil_image)
                    pil_image = enhancer.enhance(1.5)
                    
                    pil_image = pil_image.convert('L')
                    
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='PNG')
                    processed_bytes = img_buffer.getvalue()
                    
                except Exception:
                    processed_bytes = image_bytes
                
                ocr_text = ""
                if USE_EASYOCR:
                    try:
                        results = reader.readtext(processed_bytes, detail=0, paragraph=True)
                        ocr_text = "\n".join(results) if results else ""
                    except Exception:
                        pass
                else:
                    try:
                        result, _ = ocr_engine(processed_bytes)
                        if result:
                            ocr_text = " ".join([line[1] for line in result])
                    except Exception:
                        pass
                
                ocr_text = ocr_text.strip()
                
                if len(ocr_text) < 3:
                    continue
                
                image_items.append({
                    'page': page_idx,
                    'y_pos': y_position,
                    'type': 'image',
                    'ocr_text': ocr_text,
                    'image_number': img_idx + 1
                })
                
            except Exception:
                continue
    
    pdf_doc.close()
    return image_items


def get_groq_image_description(ocr_text: str, groq_client: Groq) -> str:
    try:
        prompt = IMAGE_DESCRIPTION_PROMPT.format(ocr_text=ocr_text)
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=500,
        )
        
        description = chat_completion.choices[0].message.content.strip()
        return description
        
    except Exception as e:
        return f"Image description generation failed: {str(e)}"


def get_y_position(item) -> float:
    if not hasattr(item, 'prov') or not item.prov:
        return 0.0
    
    prov = item.prov[0] if isinstance(item.prov, list) else item.prov
    
    if hasattr(prov, 'bbox') and prov.bbox:
        bbox = prov.bbox
        
        if hasattr(bbox, 'y0'):
            return float(bbox.y0)
        elif hasattr(bbox, 't'):
            return float(bbox.t)
        elif hasattr(bbox, 'l'):
            try:
                return float(bbox.t) if hasattr(bbox, 't') else float(bbox[1])
            except:
                pass
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            return float(bbox[1])
    
    return 0.0


def extract_with_docling(pdf_path: str) -> Tuple[List[Dict[str, Any]], int, int]:
    pipeline_opts = PdfPipelineOptions(
        do_table_structure=True,
        generate_picture_images=False,
        do_picture_description=False,
        do_ocr=False,
    )
    
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    
    all_items = []
    text_count = 0
    table_count = 0
    
    seen_hashes = set()
    current_text_group = []
    current_page = -1
    current_y = 0.0
    
    for item, level in doc.iterate_items():
        page_idx = 0
        if hasattr(item, 'prov') and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(prov, 'page_no'):
                page_idx = prov.page_no - 1
        
        y_position = get_y_position(item)
        
        if hasattr(item, 'label') and hasattr(item, 'text'):
            label_str = str(item.label)
            
            if 'PARAGRAPH' in label_str or 'TEXT' in label_str or 'LIST_ITEM' in label_str or 'CAPTION' in label_str:
                if item.text and item.text.strip():
                    text_content = item.text.strip()
                    text_hash = hash(text_content)
                    
                    if text_hash in seen_hashes:
                        continue
                    seen_hashes.add(text_hash)
                    
                    if page_idx == current_page and abs(y_position - current_y) < 50:
                        current_text_group.append(text_content)
                        current_y = y_position
                    else:
                        if current_text_group:
                            grouped_text = "\n\n".join(current_text_group)
                            all_items.append({
                                'page': current_page,
                                'y_pos': current_y,
                                'type': 'text',
                                'content': grouped_text
                            })
                            text_count += 1
                        
                        current_text_group = [text_content]
                        current_page = page_idx
                        current_y = y_position
            
            elif 'TITLE' in label_str or 'SECTION_HEADER' in label_str:
                if item.text and item.text.strip():
                    text_content = item.text.strip()
                    text_hash = hash(text_content)
                    
                    if text_hash in seen_hashes:
                        continue
                    seen_hashes.add(text_hash)
                    
                    if current_text_group:
                        grouped_text = "\n\n".join(current_text_group)
                        all_items.append({
                            'page': current_page,
                            'y_pos': current_y,
                            'type': 'text',
                            'content': grouped_text
                        })
                        text_count += 1
                        current_text_group = []
                    
                    all_items.append({
                        'page': page_idx,
                        'y_pos': y_position,
                        'type': 'text',
                        'content': text_content
                    })
                    text_count += 1
        
        if hasattr(item, 'label') and 'TABLE' in str(item.label):
            if current_text_group:
                grouped_text = "\n\n".join(current_text_group)
                all_items.append({
                    'page': current_page,
                    'y_pos': current_y,
                    'type': 'text',
                    'content': grouped_text
                })
                text_count += 1
                current_text_group = []
            
            table_text = ""
            try:
                if hasattr(item, 'export_to_markdown'):
                    table_text = item.export_to_markdown(doc=doc)
            except TypeError:
                try:
                    table_text = item.export_to_markdown()
                except:
                    pass
            except Exception:
                pass
            
            if table_text and table_text.strip():
                all_items.append({
                    'page': page_idx,
                    'y_pos': y_position,
                    'type': 'table',
                    'content': table_text.strip()
                })
                table_count += 1
    
    if current_text_group:
        grouped_text = "\n\n".join(current_text_group)
        all_items.append({
            'page': current_page,
            'y_pos': current_y,
            'type': 'text',
            'content': grouped_text
        })
        text_count += 1
    
    return all_items, text_count, table_count


def process_pdf(pdf_path: str, groq_client: Groq) -> Tuple[str, str]:
    docling_items, text_count, table_count = extract_with_docling(pdf_path)
    
    image_items = extract_images_with_ocr(pdf_path)
    
    image_items_with_desc = []
    for img_item in image_items:
        ocr_text = img_item['ocr_text']
        description = get_groq_image_description(ocr_text, groq_client)
        
        image_items_with_desc.append({
            'page': img_item['page'],
            'y_pos': img_item['y_pos'],
            'type': 'image',
            'ocr_text': ocr_text,
            'description': description,
            'image_number': img_item['image_number']
        })
    
    all_items = docling_items + image_items_with_desc
    all_items.sort(key=lambda x: (x['page'], x['y_pos']))
    
    raw_content = []
    enriched_content = []
    
    for item in all_items:
        if item['type'] == 'text':
            raw_content.append(item['content'])
            enriched_content.append(item['content'])
        elif item['type'] == 'table':
            raw_content.append(item['content'])
            enriched_content.append(item['content'])
        elif item['type'] == 'image':
            raw_content.append(f"[IMAGE - Page {item['page'] + 1}, Image {item['image_number']}]")
            raw_content.append(f"OCR Text: {item['ocr_text']}")
            
            image_desc_block = f"""<imagedesc>
Page {item['page'] + 1}, Image {item['image_number']}

OCR Text:
{item['ocr_text']}

Description:
{item['description']}
</imagedesc>"""
            enriched_content.append(image_desc_block)
    
    raw_text = "\n\n".join(raw_content)
    enriched_text = "\n\n".join(enriched_content)
    
    return raw_text, enriched_text


st.set_page_config(page_title="Enterprise PDF Ingestion Pipeline", layout="wide")
st.title("Enterprise PDF Ingestion Pipeline for RAG")

st.markdown("""
This pipeline extracts structured content from enterprise policy documents:
- **Text extraction**: Titles, headers, paragraphs, lists
- **Table extraction**: GitHub-flavored Markdown tables
- **Image processing**: OCR + AI-powered descriptions via Groq
""")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in environment variables")
    else:
        with st.spinner("Processing PDF..."):
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded_file.read())
            
            try:
                groq_client = Groq(api_key=GROQ_API_KEY)
                
                raw_text, enriched_text = process_pdf("temp_uploaded.pdf", groq_client)
                
                with open("raw_extracted.txt", "w", encoding="utf-8") as f:
                    f.write(raw_text)
                
                with open("enriched_output.md", "w", encoding="utf-8") as f:
                    f.write(enriched_text)
                
                st.success("Processing complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="Download raw_extracted.txt",
                        data=raw_text,
                        file_name="raw_extracted.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.download_button(
                        label="Download enriched_output.md",
                        data=enriched_text,
                        file_name="enriched_output.md",
                        mime="text/markdown"
                    )
                
                st.subheader("Preview: Enriched Output (first 5000 chars)")
                st.text_area("", enriched_text[:5000], height=400)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                if os.path.exists("temp_uploaded.pdf"):
                    os.remove("temp_uploaded.pdf")