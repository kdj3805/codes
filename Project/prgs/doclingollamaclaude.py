import streamlit as st
import re
import tempfile
import requests
import base64
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import io
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    PictureDescriptionApiOptions,
    EasyOcrOptions
)
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem


def check_ollama_status() -> Dict[str, Any]:
    """
    Verify Ollama is running and llava:7b is available.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            llava_available = any('llava' in name.lower() for name in model_names)
            return {
                'running': True,
                'llava_available': llava_available,
                'models': model_names
            }
    except Exception as e:
        return {
            'running': False,
            'error': str(e)
        }
    return {'running': False}


def get_ollama_description(image_data: bytes, model: str = "llava:7b") -> str:
    """
    Directly call Ollama API to generate image description.
    This is a fallback method when Docling's integration fails.
    """
    try:
        # Convert image bytes to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Call Ollama API directly
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": "Describe this image in detail.",
                "images": [base64_image],
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def extract_text_from_image_ocr(image_data: bytes) -> str:
    """
    Extract text from image using EasyOCR (built into Docling).
    This is useful for images that contain text.
    """
    try:
        # EasyOCR is already integrated in Docling
        # This is a placeholder - actual OCR happens in Docling pipeline
        return "[OCR text extraction handled by Docling]"
    except Exception as e:
        return f"OCR Error: {str(e)}"


def create_converter(enable_image_descriptions: bool = False, enable_ocr: bool = True) -> DocumentConverter:
    """
    Initialize DocumentConverter with OCR and optional VLM descriptions.
    """
    table_structure_options = TableStructureOptions()
    
    # Configure OCR for text extraction from images
    ocr_options = EasyOcrOptions(
        force_full_page_ocr=False,  # Only OCR images, not full pages
        use_gpu=False  # Set to True if you have GPU
    )
    
    # Build base pipeline options
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        table_structure_options=table_structure_options,
        generate_picture_images=True,
        do_ocr=enable_ocr,  # Enable OCR for text extraction from images
        ocr_options=ocr_options,
    )
    
    # Add picture description only if enabled
    if enable_image_descriptions:
        picture_api_options = PictureDescriptionApiOptions(
            api_url="http://localhost:11434/v1/chat/completions",
            model="llava:7b",
        )
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = picture_api_options
        pipeline_options.enable_remote_services = True
    else:
        pipeline_options.do_picture_description = False
    
    # Create converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    
    return converter


def extract_image_descriptions(markdown: str) -> List[Dict[str, str]]:
    """
    Extract image descriptions from Docling markdown annotations.
    """
    descriptions = []
    
    # Pattern 1: Standard annotation tags
    pattern1 = r'<!--\s*<annotation\s+kind=["\']description["\']\s*>(.*?)</annotation>\s*-->'
    matches1 = re.findall(pattern1, markdown, re.DOTALL | re.IGNORECASE)
    for match in matches1:
        clean_text = match.strip()
        if clean_text:
            descriptions.append({
                'text': clean_text,
                'source': 'annotation'
            })
    
    # Pattern 2: Alternative comment format
    pattern2 = r'<!--\s*[Ii]mage\s+description:\s*(.*?)\s*-->'
    matches2 = re.findall(pattern2, markdown, re.DOTALL)
    for match in matches2:
        clean_text = match.strip()
        if clean_text:
            descriptions.append({
                'text': clean_text,
                'source': 'comment'
            })
    
    return descriptions


def extract_image_data_from_doc(doc) -> List[Dict[str, Any]]:
    """
    Extract actual image data from Docling document.
    This allows us to manually send images to Ollama if Docling's integration fails.
    """
    images = []
    
    for item, level in doc.iterate_items():
        if isinstance(item, PictureItem):
            image_info = {
                'item': item,
                'caption': getattr(item, 'caption', None),
                'page_no': None
            }
            
            # Get page number
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    if hasattr(prov, 'page_no'):
                        image_info['page_no'] = prov.page_no
                        break
            
            # Try to get image data
            try:
                if hasattr(item, 'image') and item.image:
                    # Try to get PIL image
                    if hasattr(item.image, 'pil_image'):
                        pil_img = item.image.pil_image
                        # Convert to bytes
                        img_bytes = io.BytesIO()
                        pil_img.save(img_bytes, format='PNG')
                        image_info['data'] = img_bytes.getvalue()
                    elif hasattr(item.image, 'uri') and item.image.uri:
                        # Image might be stored as URI
                        image_info['uri'] = item.image.uri
            except Exception as e:
                image_info['error'] = str(e)
            
            images.append(image_info)
    
    return images


def extract_tables_from_doc(doc) -> List[Dict[str, Any]]:
    """
    Extract table data from Docling document structure.
    """
    tables = []
    
    for item, level in doc.iterate_items():
        if isinstance(item, TableItem):
            try:
                table_text = item.export_to_markdown(doc=doc)
            except:
                try:
                    table_text = item.export_to_markdown()
                except:
                    continue
            
            tables.append({
                'text': table_text,
                'caption': getattr(item, 'caption', None)
            })
    
    return tables


def process_pdf(pdf_file, converter: DocumentConverter, enable_descriptions: bool, 
                enable_manual_ollama: bool, ollama_model: str) -> Dict[str, Any]:
    """
    Process uploaded PDF through Docling pipeline.
    If enable_manual_ollama=True, manually extract images and send to Ollama.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    
    try:
        result = converter.convert(tmp_path)
        doc = result.document
        
        markdown = doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
        
        # Extract using Docling's method
        image_descriptions = extract_image_descriptions(markdown)
        tables = extract_tables_from_doc(doc)
        
        # Extract image data for manual processing
        image_data_list = extract_image_data_from_doc(doc)
        
        # If manual Ollama is enabled and no descriptions found, try manual approach
        manual_descriptions = []
        if enable_manual_ollama and len(image_descriptions) == 0 and len(image_data_list) > 0:
            st.info(f"Attempting manual image description for {len(image_data_list)} images...")
            for idx, img_info in enumerate(image_data_list):
                if 'data' in img_info:
                    with st.spinner(f"Describing image {idx + 1} of {len(image_data_list)}..."):
                        description = get_ollama_description(img_info['data'], ollama_model)
                        manual_descriptions.append({
                            'text': description,
                            'source': 'manual_ollama',
                            'page_no': img_info.get('page_no'),
                            'caption': img_info.get('caption')
                        })
        
        # Combine descriptions
        all_descriptions = image_descriptions + manual_descriptions
        
        image_placeholder_count = markdown.count('<!-- image -->')
        
        try:
            if hasattr(doc, 'pages') and doc.pages:
                page_numbers = [p.page_no for p in doc.pages if hasattr(p, 'page_no')]
                num_pages = max(page_numbers) if page_numbers else len(doc.pages)
            else:
                num_pages = 1
        except:
            num_pages = 1
        
        pages_data = []
        for page_num in range(1, num_pages + 1):
            page_markdown_parts = []
            page_tables = []
            page_images = []
            
            for item, level in doc.iterate_items():
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no') and prov.page_no == page_num:
                            if isinstance(item, TableItem):
                                try:
                                    table_text = item.export_to_markdown(doc=doc)
                                except:
                                    try:
                                        table_text = item.export_to_markdown()
                                    except:
                                        table_text = None
                                
                                if table_text:
                                    page_tables.append({
                                        'text': table_text,
                                        'caption': getattr(item, 'caption', None)
                                    })
                            elif isinstance(item, PictureItem):
                                page_images.append(item)
                            else:
                                try:
                                    if hasattr(item, 'export_to_markdown'):
                                        text = item.export_to_markdown()
                                    else:
                                        text = str(item)
                                    
                                    if text and text.strip():
                                        page_markdown_parts.append(text)
                                except:
                                    pass
            
            pages_data.append({
                'page_num': page_num,
                'text': '\n\n'.join(page_markdown_parts),
                'tables': page_tables,
                'image_count': len(page_images)
            })
        
        return {
            'pages': pages_data,
            'all_image_descriptions': all_descriptions,
            'full_markdown': markdown,
            'num_pages': num_pages,
            'diagnostics': {
                'total_tables': len(tables),
                'total_image_placeholders': image_placeholder_count,
                'total_descriptions_found': len(all_descriptions),
                'docling_descriptions': len(image_descriptions),
                'manual_descriptions': len(manual_descriptions),
                'descriptions_enabled': enable_descriptions,
                'manual_ollama_enabled': enable_manual_ollama,
                'images_found': len(image_data_list)
            }
        }
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main():
    st.set_page_config(
        page_title="Multimodal RAG PDF Ingestion",
        layout="wide"
    )
    
    st.title("Multimodal RAG PDF Ingestion Pipeline")
    st.markdown("**Docling + Ollama LLaVA** | PyPdfium2 Backend | OCR + VLM")
    
    with st.sidebar:
        st.header("Configuration")
        
        ollama_status = check_ollama_status()
        
        if ollama_status.get('running'):
            st.success("Ollama is running")
            if ollama_status.get('llava_available'):
                st.success("LLaVA model detected")
            else:
                st.warning("LLaVA model not found")
                st.caption("Available models: " + ", ".join(ollama_status.get('models', [])))
        else:
            st.error("Ollama is not running")
            st.caption(f"Error: {ollama_status.get('error', 'Unknown')}")
        
        st.divider()
        
        st.subheader("Image Processing Options")
        
        enable_ocr = st.checkbox(
            "Enable OCR (Extract text from images)",
            value=True,
            help="Use EasyOCR to extract text from images"
        )
        
        enable_vl_descriptions = st.checkbox(
            "Enable Docling VLM Integration",
            value=False,
            help="Let Docling handle image descriptions via Ollama (may not work)"
        )
        
        enable_manual_ollama = st.checkbox(
            "Enable Manual Ollama Fallback",
            value=ollama_status.get('running', False) and ollama_status.get('llava_available', False),
            help="Manually extract images and send to Ollama if Docling integration fails"
        )
        
        ollama_model = st.selectbox(
            "Ollama Model",
            ["llava:7b", "llava:13b", "llava:34b", "bakllava"],
            help="Select which vision model to use"
        )
        
        if enable_manual_ollama:
            st.info("Manual mode: Will extract images and send directly to Ollama API")
            st.warning("This may take longer but is more reliable")
        
        if enable_vl_descriptions:
            st.info("**Endpoint**: http://localhost:11434/v1/chat/completions")
            if not ollama_status.get('running'):
                st.warning("Start Ollama: ollama serve")
    
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="PDFs with text, tables, and images"
    )
    
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        
        if st.button("Process Document", type="primary"):
            if (enable_vl_descriptions or enable_manual_ollama) and not ollama_status.get('running'):
                st.error("Cannot enable image descriptions: Ollama is not running")
                st.stop()
            
            with st.spinner("Initializing Docling converter..."):
                converter = create_converter(
                    enable_image_descriptions=enable_vl_descriptions,
                    enable_ocr=enable_ocr
                )
            
            processing_msg = "Processing PDF"
            if enable_ocr:
                processing_msg += " with OCR"
            if enable_manual_ollama:
                processing_msg += " + Manual VLM"
            processing_msg += "..."
            
            with st.spinner(processing_msg):
                try:
                    result = process_pdf(
                        uploaded_file, 
                        converter, 
                        enable_vl_descriptions,
                        enable_manual_ollama,
                        ollama_model
                    )
                    
                    diag = result['diagnostics']
                    st.success(f"Processed {result['num_pages']} pages successfully")
                    
                    with st.expander("Processing Details"):
                        st.write(f"**Tables found**: {diag['total_tables']}")
                        st.write(f"**Image placeholders**: {diag['total_image_placeholders']}")
                        st.write(f"**Images extracted**: {diag['images_found']}")
                        st.write(f"**Descriptions (Docling)**: {diag['docling_descriptions']}")
                        st.write(f"**Descriptions (Manual)**: {diag['manual_descriptions']}")
                        st.write(f"**Total descriptions**: {diag['total_descriptions_found']}")
                        st.write(f"**OCR enabled**: {enable_ocr}")
                        
                        if diag['images_found'] > 0:
                            if diag['total_descriptions_found'] == 0:
                                st.warning("Images found but no descriptions generated")
                                st.info("Try enabling 'Manual Ollama Fallback' in sidebar")
                            elif diag['manual_descriptions'] > 0:
                                st.success(f"Manual fallback generated {diag['manual_descriptions']} descriptions")
                    
                    tab1, tab2, tab3 = st.tabs(["Page-by-Page", "Image Descriptions", "Full Markdown"])
                    
                    with tab1:
                        st.header("Extracted Content by Page")
                        
                        if not result['pages']:
                            st.warning("No content extracted")
                        
                        for page_data in result['pages']:
                            page_label = f"Page {page_data['page_num']}"
                            if page_data.get('image_count', 0) > 0:
                                page_label += f" ({page_data['image_count']} images)"
                            
                            with st.expander(page_label, expanded=True):
                                if page_data['text']:
                                    st.subheader("Text Content")
                                    st.markdown(page_data['text'])
                                else:
                                    st.caption("No text content on this page")
                                
                                if page_data['tables']:
                                    st.subheader(f"Tables ({len(page_data['tables'])})")
                                    for idx, table in enumerate(page_data['tables'], 1):
                                        st.markdown(f"**Table {idx}**")
                                        if table['caption']:
                                            st.caption(table['caption'])
                                        st.markdown(table['text'])
                                        st.divider()
                    
                    with tab2:
                        st.header("Image Descriptions")
                        
                        if result['all_image_descriptions']:
                            st.success(f"Found {len(result['all_image_descriptions'])} image description(s)")
                            
                            for idx, desc in enumerate(result['all_image_descriptions'], 1):
                                source_label = desc.get('source', 'unknown')
                                if source_label == 'manual_ollama':
                                    source_label = f"Manual Ollama (Page {desc.get('page_no', '?')})"
                                
                                st.markdown(f"**Image {idx}** (Source: {source_label})")
                                
                                if desc.get('caption'):
                                    st.caption(f"Caption: {desc['caption']}")
                                
                                st.info(desc['text'])
                                st.divider()
                        else:
                            st.warning("No image descriptions found")
                            if diag['images_found'] > 0:
                                st.info(f"{diag['images_found']} image(s) detected")
                                st.markdown("""
                                **Try these options:**
                                1. Enable 'Manual Ollama Fallback' in sidebar
                                2. Check Ollama is running: `ollama serve`
                                3. Test model: `ollama run llava:7b "describe this"`
                                4. Try a different model (llava:13b, bakllava)
                                """)
                    
                    with tab3:
                        st.header("Complete Markdown Export")
                        st.markdown("Raw markdown output from Docling")
                        
                        preview_text = result['full_markdown'][:2000]
                        if len(result['full_markdown']) > 2000:
                            preview_text += "\n\n... (truncated, download full file below)"
                        
                        st.code(preview_text, language='markdown')
                        
                        st.download_button(
                            label="Download Full Markdown",
                            data=result['full_markdown'],
                            file_name=f"{uploaded_file.name}.md",
                            mime="text/markdown"
                        )
                
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("Upload a PDF document to begin processing")
        
        with st.expander("How it works"):
            st.markdown("""
            **Three-Tiered Image Processing:**
            
            1. **OCR (Text Extraction)**: EasyOCR extracts text from images containing text
            2. **Docling VLM**: Docling's built-in Ollama integration (may fail)
            3. **Manual Fallback**: Direct Ollama API calls for image descriptions
            
            **Recommended Settings:**
            - Enable OCR: Always on
            - Enable Manual Ollama Fallback: When you need reliable descriptions
            - Docling VLM: Optional (experimental)
            
            **Why Manual Fallback?**
            - More reliable than Docling's integration
            - Direct control over Ollama API
            - Better error handling
            - Works when Docling integration fails
            """)
        
        with st.expander("Quick Start"):
            st.markdown("""
            **Setup:**
            
            1. Install and start Ollama: `ollama serve`
            2. Pull model: `ollama pull llava:7b`
            3. Install packages: `pip install docling streamlit pillow`
            4. Run: `streamlit run app.py`
            
            **For best results:**
            - Enable "Manual Ollama Fallback"
            - Keep OCR enabled for text extraction
            - Use llava:7b for speed, llava:13b for quality
            """)


if __name__ == "__main__":
    main()