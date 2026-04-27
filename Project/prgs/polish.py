import os
import re
import unicodedata
from pathlib import Path

def final_polish_markdown(input_folder="markdown_papers"):
    md_files = list(Path(input_folder).glob("*.md"))
    
    for md_path in md_files:
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # 1. Normalize weird unicode numbers (turns  into 2019)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Stitch broken words back together (e.g., "strati fi cation" -> "stratification")
        # Matches: Letters + space(s) + (fi|fl|ff|ffi|ffl) + space(s) + Letters
        text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2\3', text)
        
        # 3. Catch edge cases with a space on only one side
        text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)(?=[a-zA-Z])', r'\1\2', text)
        text = re.sub(r'(?<=[a-zA-Z])(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2', text)
        
        # Save the polished text
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
            
    print(f"✅ Successfully polished {len(md_files)} markdown files!")

if __name__ == "__main__":
    final_polish_markdown(input_folder=r"D:\trial\markdown_papers") # Change to your actual path