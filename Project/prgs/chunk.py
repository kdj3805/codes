import streamlit as st
import tempfile
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import io
from PIL import Image
import json

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling_core.types.doc import ImageRefMode, TableItem, PictureItem

from rapidocr_onnxruntime import RapidOCR

# Chunking imports
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================
# CONFIG
# =========================

ocr_engine = RapidOCR()


# =========================
# DOCLING CONVERTER
# =========================

def create_converter(aggressive_ocr: bool = False) -> DocumentConverter:
    """
    Configure Docling to extract text properly.
    """
    
    ocr_options = EasyOcrOptions(
        force_full_page_ocr=False,
        use_gpu=False
    )
    
    pipeline_opts = PdfPipelineOptions(
        do_table_structure=True,
        generate_picture_images=True,
        do_picture_description=False,
        do_ocr=aggressive_ocr,
        ocr_options=ocr_options,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_opts
            )
        }
    )


# =========================
# IMAGE OCR
# =========================

def extract_all_image_text(pdf_path: str, page_no: int) -> List[Dict[str, Any]]:
    """
    Extract text from images using RapidOCR.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]

    image_texts = []
    
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        
        try:
            pix = fitz.Pixmap(doc, xref)
            
            if pix.n < 5:
                if pix.n == 1:
                    img_pil = Image.frombytes("L", [pix.width, pix.height], pix.samples)
                else:
                    img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                img_pil.save(img_path)

                result, _ = ocr_engine(img_path)
                
                if result:
                    ocr_lines = [line[1] for line in result]
                    ocr_text = "\n".join(ocr_lines)
                    
                    if ocr_text.strip():
                        image_texts.append({
                            'text': ocr_text,
                            'index': img_index,
                            'size': f"{pix.width}x{pix.height}",
                            'lines': len(ocr_lines)
                        })

                Path(img_path).unlink(missing_ok=True)

            pix = None
            
        except Exception as e:
            continue

    doc.close()
    return image_texts


# =========================
# PDF PROCESSOR
# =========================

def process_pdf(pdf_bytes: bytes, max_pages: int | None = None, 
                ocr_mode: str = "auto") -> Dict[str, Any]:
    """
    Process PDF extracting ALL content: text, tables, and images.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        pdf_path = f.name

    try:
        converter = create_converter(aggressive_ocr=(ocr_mode == "aggressive"))
        result = converter.convert(pdf_path)
        doc = result.document

        raw_text_parts = []
        structured_parts = []
        
        current_page = 0
        page_stats = {
            'total_pages': 0,
            'text_items': 0,
            'tables': 0,
            'images': 0,
            'images_with_text': 0,
        }
        
        processed_images_per_page = {}

        for item, level in doc.iterate_items():
            if hasattr(item, "prov") and item.prov:
                page_no = item.prov[0].page_no
                
                if page_no != current_page:
                    current_page = page_no
                    page_stats['total_pages'] = current_page
                    processed_images_per_page[current_page] = False
                    
                    if max_pages and current_page > max_pages:
                        break

                    structured_parts.append(f"\n\n## Page {current_page}\n")
                    raw_text_parts.append(f"\n\n=== PAGE {current_page} ===\n\n")
                    
                    if ocr_mode in ["aggressive", "auto"] and not processed_images_per_page[current_page]:
                        image_texts = extract_all_image_text(pdf_path, current_page)
                        if image_texts:
                            for img_data in image_texts:
                                page_stats['images_with_text'] += 1
                                structured_parts.append(f"\n<imagedesc>\n{img_data['text']}\n</imagedesc>\n")
                                raw_text_parts.append(f"\n[IMAGE TEXT]:\n{img_data['text']}\n")
                        processed_images_per_page[current_page] = True

            if isinstance(item, TableItem):
                page_stats['tables'] += 1
                
                try:
                    table_md = item.export_to_markdown(doc=doc)
                    structured_parts.append(f"\n<tableinfo>\n{table_md.strip()}\n</tableinfo>\n")
                    raw_text_parts.append(f"\n{table_md.strip()}\n")
                except Exception:
                    try:
                        table_md = item.export_to_markdown()
                        structured_parts.append(f"\n<tableinfo>\n{table_md.strip()}\n</tableinfo>\n")
                        raw_text_parts.append(f"\n{table_md.strip()}\n")
                    except Exception:
                        structured_parts.append("<tableinfo>Table detected but could not be parsed</tableinfo>")
                    
            elif isinstance(item, PictureItem):
                page_stats['images'] += 1
                
                try:
                    if hasattr(item, 'text') and item.text and item.text.strip():
                        ocr_text = item.text.strip()
                        page_stats['images_with_text'] += 1
                        structured_parts.append(f"\n<imagedesc>\n{ocr_text}\n</imagedesc>\n")
                        raw_text_parts.append(f"\n[IMAGE TEXT]:\n{ocr_text}\n")
                except Exception:
                    pass
                    
            else:
                page_stats['text_items'] += 1
                
                try:
                    if hasattr(item, 'export_to_markdown'):
                        text = item.export_to_markdown()
                    else:
                        text = getattr(item, 'text', str(item))
                    
                    if text and text.strip():
                        structured_parts.append(text)
                        raw_text_parts.append(text)
                        
                except Exception as e:
                    try:
                        if hasattr(item, 'text'):
                            text = item.text
                            if text and text.strip():
                                structured_parts.append(text)
                                raw_text_parts.append(text)
                    except:
                        pass

        return {
            'raw_text': '\n\n'.join(raw_text_parts),
            'structured_output': '\n\n'.join(structured_parts),
            'diagnostics': page_stats
        }

    finally:
        Path(pdf_path).unlink(missing_ok=True)


# =========================
# CHUNKING STRATEGIES
# =========================

@st.cache_resource
def load_embeddings():
    """Load embeddings model for semantic chunking (cached)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


def chunk_text(text: str, strategy: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Apply different chunking strategies to text.
    
    Returns list of chunks with metadata.
    """
    
    if strategy == "Character-based":
        splitter = CharacterTextSplitter(
            separator=kwargs.get('separator', '\n\n'),
            chunk_size=kwargs.get('chunk_size', 1000),
            chunk_overlap=kwargs.get('chunk_overlap', 200),
            length_function=len,
        )
        
    elif strategy == "Recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=kwargs.get('chunk_size', 1000),
            chunk_overlap=kwargs.get('chunk_overlap', 200),
            length_function=len,
            separators=kwargs.get('separators', ["\n\n", "\n", " ", ""]),
        )
        
    elif strategy == "Token-based":
        splitter = TokenTextSplitter(
            chunk_size=kwargs.get('chunk_size', 500),
            chunk_overlap=kwargs.get('chunk_overlap', 50),
        )
        
    elif strategy == "Semantic":
        embeddings = load_embeddings()
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=kwargs.get('breakpoint_type', 'percentile'),
            breakpoint_threshold_amount=kwargs.get('breakpoint_amount', 95),
        )
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Split the text
    chunks = splitter.split_text(text)
    
    # Add metadata
    chunks_with_metadata = []
    for idx, chunk in enumerate(chunks):
        chunks_with_metadata.append({
            'chunk_id': idx + 1,
            'text': chunk,
            'length': len(chunk),
            'tokens': len(chunk.split()),  # Approximate token count
        })
    
    return chunks_with_metadata


def analyze_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze chunk statistics.
    """
    if not chunks:
        return {}
    
    lengths = [c['length'] for c in chunks]
    token_counts = [c['tokens'] for c in chunks]
    
    return {
        'total_chunks': len(chunks),
        'total_characters': sum(lengths),
        'avg_chunk_length': sum(lengths) / len(lengths),
        'min_chunk_length': min(lengths),
        'max_chunk_length': max(lengths),
        'avg_tokens': sum(token_counts) / len(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts),
    }


# =========================
# STREAMLIT UI
# =========================

def main():
    st.set_page_config(
        page_title="RAG Pipeline: Extract + Chunk",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 RAG Pipeline: PDF Extraction + Chunking")
    st.markdown("**Extract text from PDFs → Apply chunking strategies → Prepare for embeddings**")
    
    # Initialize session state
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None
    if 'extraction_stats' not in st.session_state:
        st.session_state.extraction_stats = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("1. Extraction Settings")
        
        max_pages = st.number_input(
            "Limit pages (0 = all)", 
            min_value=0, 
            max_value=100, 
            value=0
        )
        
        ocr_mode = st.selectbox(
            "Image OCR Mode",
            ["aggressive", "auto", "off"],
            index=0,
            help="Aggressive mode recommended for most PDFs"
        )
        
        st.divider()
        
        st.subheader("2. Chunking Strategy")
        
        chunking_strategy = st.selectbox(
            "Strategy",
            ["Recursive", "Character-based", "Token-based", "Semantic"],
            help="""
            - Recursive: Best for general use (recommended)
            - Character-based: Simple split by separator
            - Token-based: Split by token count
            - Semantic: AI-powered semantic boundaries (slower)
            """
        )
        
        st.divider()
        
        st.subheader("3. Chunking Parameters")
        
        if chunking_strategy in ["Recursive", "Character-based"]:
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=100,
                max_value=4000,
                value=1000,
                step=100,
                help="Target size for each chunk"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between consecutive chunks"
            )
            
        elif chunking_strategy == "Token-based":
            chunk_size = st.slider(
                "Chunk Size (tokens)",
                min_value=50,
                max_value=2000,
                value=500,
                step=50
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap (tokens)",
                min_value=0,
                max_value=200,
                value=50,
                step=10
            )
            
        elif chunking_strategy == "Semantic":
            breakpoint_type = st.selectbox(
                "Breakpoint Type",
                ["percentile", "standard_deviation", "interquartile"],
                help="How to determine semantic boundaries"
            )
            
            if breakpoint_type == "percentile":
                breakpoint_amount = st.slider(
                    "Percentile Threshold",
                    min_value=50,
                    max_value=99,
                    value=95,
                    help="Higher = fewer, larger chunks"
                )
            else:
                breakpoint_amount = st.slider(
                    "Threshold Amount",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1
                )
        
        st.divider()
        
        st.markdown("""
        **Pipeline Flow:**
        1. Upload PDF
        2. Extract text + tables + images
        3. Apply chunking strategy
        4. Review chunks
        5. Export for embeddings
        """)
    
    # Main content area
    uploaded = st.file_uploader(
        "📄 Upload PDF Document",
        type=["pdf"],
        help="Upload your PDF to begin extraction and chunking"
    )
    
    if uploaded:
        st.success(f"✓ Uploaded: {uploaded.name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            extract_btn = st.button(
                "🔄 Extract Text from PDF",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            if st.session_state.extracted_text:
                chunk_btn = st.button(
                    "✂️ Apply Chunking Strategy",
                    type="secondary",
                    use_container_width=True
                )
            else:
                st.button(
                    "✂️ Apply Chunking Strategy",
                    disabled=True,
                    use_container_width=True,
                    help="Extract text first"
                )
                chunk_btn = False
        
        # STEP 1: EXTRACTION
        if extract_btn:
            with st.spinner("🔍 Extracting text, tables, and images..."):
                result = process_pdf(
                    uploaded.getvalue(),
                    None if max_pages == 0 else max_pages,
                    ocr_mode
                )
                
                st.session_state.extracted_text = result['raw_text']
                st.session_state.extraction_stats = result['diagnostics']
            
            st.success("✓ Text extraction complete!")
            st.rerun()
        
        # Show extraction results
        if st.session_state.extracted_text:
            st.divider()
            
            st.subheader("📊 Extraction Results")
            
            stats = st.session_state.extraction_stats
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Pages", stats['total_pages'])
            with col2:
                st.metric("Text Items", stats['text_items'])
            with col3:
                st.metric("Tables", stats['tables'])
            with col4:
                st.metric("Images", stats['images'])
            with col5:
                st.metric("Chars", len(st.session_state.extracted_text))
            
            with st.expander("📄 View Extracted Text"):
                preview_len = 2000
                preview = st.session_state.extracted_text[:preview_len]
                if len(st.session_state.extracted_text) > preview_len:
                    preview += "\n\n... (truncated, see full text after chunking)"
                
                st.text_area(
                    "Extracted Text Preview",
                    preview,
                    height=300
                )
        
        # STEP 2: CHUNKING
        if st.session_state.extracted_text and chunk_btn:
            with st.spinner(f"✂️ Applying {chunking_strategy} chunking..."):
                
                # Prepare kwargs based on strategy
                if chunking_strategy in ["Recursive", "Character-based"]:
                    kwargs = {
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap
                    }
                elif chunking_strategy == "Token-based":
                    kwargs = {
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap
                    }
                elif chunking_strategy == "Semantic":
                    kwargs = {
                        'breakpoint_type': breakpoint_type,
                        'breakpoint_amount': breakpoint_amount
                    }
                
                # Chunk the text
                chunks = chunk_text(
                    st.session_state.extracted_text,
                    chunking_strategy,
                    **kwargs
                )
                
                # Analyze chunks
                chunk_stats = analyze_chunks(chunks)
            
            st.success(f"✓ Created {len(chunks)} chunks using {chunking_strategy} strategy")
            
            st.divider()
            
            # Chunk Statistics
            st.subheader("📊 Chunk Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Chunks", chunk_stats['total_chunks'])
            with col2:
                st.metric("Avg Length", f"{chunk_stats['avg_chunk_length']:.0f} chars")
            with col3:
                st.metric("Avg Tokens", f"{chunk_stats['avg_tokens']:.0f}")
            with col4:
                st.metric("Size Range", f"{chunk_stats['min_chunk_length']}-{chunk_stats['max_chunk_length']}")
            
            # Detailed stats
            with st.expander("📈 Detailed Statistics"):
                st.json(chunk_stats)
            
            st.divider()
            
            # Chunk Viewer
            st.subheader("🔍 Chunk Viewer")
            
            # Navigation
            col_nav1, col_nav2 = st.columns([3, 1])
            with col_nav1:
                selected_chunk_idx = st.slider(
                    "Select Chunk to View",
                    min_value=1,
                    max_value=len(chunks),
                    value=1,
                    help="Navigate through chunks"
                )
            with col_nav2:
                st.metric("Current Chunk", f"{selected_chunk_idx}/{len(chunks)}")
            
            # Display selected chunk
            selected_chunk = chunks[selected_chunk_idx - 1]
            
            st.markdown(f"**Chunk #{selected_chunk['chunk_id']}**")
            st.info(f"Length: {selected_chunk['length']} characters | Tokens: ~{selected_chunk['tokens']}")
            
            st.text_area(
                "Chunk Content",
                selected_chunk['text'],
                height=300,
                key=f"chunk_{selected_chunk_idx}"
            )
            
            # Navigation buttons
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
            with col_btn1:
                if st.button("⬅️ Previous", disabled=(selected_chunk_idx == 1)):
                    st.session_state.selected_chunk_idx = selected_chunk_idx - 1
                    st.rerun()
            with col_btn2:
                if st.button("➡️ Next", disabled=(selected_chunk_idx == len(chunks))):
                    st.session_state.selected_chunk_idx = selected_chunk_idx + 1
                    st.rerun()
            
            st.divider()
            
            # Export Options
            st.subheader("💾 Export Chunks")
            
            # Prepare export data
            chunks_json = json.dumps(chunks, indent=2)
            chunks_text = "\n\n---CHUNK SEPARATOR---\n\n".join([
                f"CHUNK {c['chunk_id']}\nLength: {c['length']} | Tokens: {c['tokens']}\n\n{c['text']}"
                for c in chunks
            ])
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.download_button(
                    "⬇️ Download as JSON",
                    chunks_json,
                    file_name=f"{uploaded.name}_chunks.json",
                    mime="application/json",
                    use_container_width=True,
                    help="JSON format with metadata (for RAG pipelines)"
                )
            
            with col_exp2:
                st.download_button(
                    "⬇️ Download as TXT",
                    chunks_text,
                    file_name=f"{uploaded.name}_chunks.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Plain text format (for review)"
                )
            
            st.divider()
            
            # Recommendations
            st.subheader("💡 Recommendations for RAG")
            
            if chunk_stats['avg_chunk_length'] < 500:
                st.warning("⚠️ Chunks are small - consider increasing chunk size for better context")
            elif chunk_stats['avg_chunk_length'] > 2000:
                st.warning("⚠️ Chunks are large - consider decreasing chunk size for better retrieval")
            else:
                st.success("✓ Chunk size looks good for most RAG use cases")
            
            if chunk_overlap < chunk_size * 0.1:
                st.info("ℹ️ Low overlap - may miss context at boundaries")
            elif chunk_overlap > chunk_size * 0.3:
                st.info("ℹ️ High overlap - more redundancy but better context preservation")
            
            st.markdown("""
            **Next Steps:**
            1. Review chunks to ensure quality
            2. Export chunks as JSON
            3. Generate embeddings (OpenAI, Cohere, or local)
            4. Store in vector database (Pinecone, Weaviate, ChromaDB, etc.)
            5. Build RAG retrieval system
            """)
    
    else:
        st.info("👆 Upload a PDF document to begin")
        
        with st.expander("ℹ️ About Chunking Strategies"):
            st.markdown("""
            ## Chunking Strategies Explained
            
            ### 1. Recursive (Recommended for most cases)
            - Splits text using multiple separators in order
            - Default: `["\\n\\n", "\\n", " ", ""]`
            - Best for preserving document structure
            - **Use when:** General purpose, mixed content
            
            ### 2. Character-based
            - Simple split by separator (e.g., paragraph breaks)
            - Fast and predictable
            - **Use when:** Clean, well-structured documents
            
            ### 3. Token-based
            - Splits by token count (useful for LLM limits)
            - More accurate for API token limits
            - **Use when:** Need precise token control (OpenAI, etc.)
            
            ### 4. Semantic (Experimental)
            - AI-powered semantic boundary detection
            - Preserves meaning and context
            - Slower but more intelligent
            - **Use when:** Complex documents, need best semantic coherence
            
            ## Parameters Guide
            
            **Chunk Size:**
            - Small (500-1000): Better precision, more chunks
            - Medium (1000-1500): Balanced (recommended)
            - Large (1500-3000): Better context, fewer chunks
            
            **Chunk Overlap:**
            - 10-20% of chunk size (recommended)
            - Prevents loss of context at boundaries
            - Higher overlap = more redundancy but safer
            
            ## Best Practices
            
            1. **Start with Recursive** - Works well for most cases
            2. **Test different sizes** - Depends on your content
            3. **Monitor retrieval quality** - Adjust based on results
            4. **Consider content type** - Technical docs vs narrative text
            """)
        
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            **Setup:**
```bash
            pip install docling streamlit pymupdf pillow rapidocr-onnxruntime
            pip install langchain langchain-experimental sentence-transformers
```
            
            **Usage:**
            1. Upload your PDF
            2. Click "Extract Text from PDF"
            3. Configure chunking strategy in sidebar
            4. Click "Apply Chunking Strategy"
            5. Review chunks and statistics
            6. Export for your RAG pipeline
            
            **For Production RAG:**
            - Use exported JSON format
            - Generate embeddings with your model
            - Store in vector DB
            - Implement retrieval + generation
            """)


if __name__ == "__main__":
    main()