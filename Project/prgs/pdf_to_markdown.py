from pymupdf4llm import to_markdown

pdf_file = "D:\\trial\\data\\IBM_SAM_ESSO_PoliciesGuide_pdf.pdf"
md_text = to_markdown(doc=pdf_file)
print(md_text)