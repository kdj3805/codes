import os
import re
import unicodedata
from pathlib import Path
from docling.document_converter import DocumentConverter

def clean_and_polish_markdown(text):
    """
    A unified pipeline to clean up PDF extraction errors, fix unicode issues,
    stitch broken words, and surgically remove the References section.
    """
    # 1. Normalize weird unicode numbers and characters (turns  into 2019)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. CHOP OFF REFERENCES SECTION (Aggressive Version)
    # Matches "# References", "**References**", "REFERENCES", or "Reference" 
    # as long as it is the only thing on that line.
    ref_pattern = re.compile(
        r'^#*\s*\*?\*?(references?|bibliography|literature cited|works cited)\*?\*?\s*$', 
        re.IGNORECASE | re.MULTILINE
    )
    match = ref_pattern.search(text)
    if match:
        # Keep everything up to the exact moment the References section starts
        text = text[:match.start()].strip()
        
    # 3. Clean PDF ligature error codes (e.g., /uniFB01) and stray markers
    replacements = {
        r'\s*/uniFB01\s*': 'fi',
        r'\s*/uniFB02\s*': 'fl',
        r'\s*/uniFB00\s*': 'ff',
        r'\s*/uniFB03\s*': 'ffi',
        r'\s*/uniFB04\s*': 'ffl',
        r'\s*/uniF642\s*': '%',
        r'\s*/C15\s*': '-',   # Stray bullet point markers
        r'\s*/C19\s*': 'e',   # Misread accented e
        r'\s*/C20\s*': 'c',
        r'\s*/C211\s*': ' ',  # Stray copyright/header characters
        r'ﬁ': 'fi',
        r'ﬂ': 'fl'
    }
    for pattern, fixed in replacements.items():
        text = re.sub(pattern, fixed, text)
        
    # 4. Stitch broken words back together (e.g., "strati fi cation" -> "stratification")
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2\3', text)
    # Catch edge cases with a space on only one side
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)(?=[a-zA-Z])', r'\1\2', text)
    text = re.sub(r'(?<=[a-zA-Z])(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2', text)
        
    # 5. Fix any double spaces created by the extraction
    text = re.sub(r' {2,}', ' ', text)
        
    return text.strip()

def process_pdfs_pipeline(input_folder="pdfs", output_folder="markdown_papers"):
    """
    Reads PDFs, extracts layout-aware Markdown, cleans it perfectly, 
    removes references, and saves to disk.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print("Initializing Docling AI models (this may take a moment on first run)...")
    converter = DocumentConverter()
    
    pdf_paths = list(Path(input_folder).glob("*.pdf"))
    
    if not pdf_paths:
        print(f"No PDFs found in the '{input_folder}' directory.")
        return

    print(f"Found {len(pdf_paths)} PDFs. Starting extraction & cleaning pipeline...")
    print("-" * 50)
    
    for pdf_path in pdf_paths:
        print(f"⏳ Processing: {pdf_path.name}")
        try:
            # 1. Convert the document to raw Markdown
            conv_result = converter.convert(pdf_path)
            raw_markdown = conv_result.document.export_to_markdown()
            
            # 2. Run the unified cleaning and reference-removal function
            final_markdown = clean_and_polish_markdown(raw_markdown)
            
            # 3. Save the perfect file
            output_filename = f"{pdf_path.stem}.md"
            output_filepath = Path(output_folder) / output_filename
            
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(final_markdown)
                
            print(f"  ✅ Success! Saved to {output_filepath}")
            
        except Exception as e:
            print(f"  ❌ Failed to process {pdf_path.name}")
            print(f"     Error: {e}")

if __name__ == "__main__":
    # Define your folders here
    INPUT_DIR = r"D:\trial\pdfs"
    OUTPUT_DIR = r"D:\trial\markdown_papers"
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created folder '{INPUT_DIR}'. Please put your PDFs in here and run again.")
    else:
        process_pdfs_pipeline(input_folder=INPUT_DIR, output_folder=OUTPUT_DIR)