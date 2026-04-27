
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

PDF_PATH = "D:\\trial\\data\\IBMMaaS360_Best_Practices_for_Policies.pdf"

pipeline_opts = PdfPipelineOptions(
    do_table_structure=True,
    do_picture_description=False,
    enable_remote_services=False,
    generate_picture_images=False,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_opts
        )
    }
)

result = converter.convert(PDF_PATH)
doc = result.document

print("===== TEXT DUMP =====\n")
print(doc.export_to_markdown())
