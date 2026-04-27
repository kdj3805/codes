import streamlit as st
import pandas as pd
import os
import re

from langchain_community.document_loaders import UnstructuredPDFLoader


st.set_page_config(layout="wide")
st.title("Unstructured PDF Loader")

PDF_PATH = "D:\\trial\\data\\Disaster_Management_Unit_1.pdf"   # change filename only
TARGET_PAGE = 1              # change page number if needed

if not os.path.exists(PDF_PATH):
    st.error("PDF file not found.")
    st.stop()


loader = UnstructuredPDFLoader(
    PDF_PATH,
    mode="elements",
    strategy="fast"
)

docs = loader.load()

page_elements = [
    d for d in docs
    if d.metadata.get("page_number") == TARGET_PAGE
]


st.header("Extracted Text")

raw_text = "\n".join(d.page_content for d in page_elements)
st.text_area("Raw extracted text", raw_text, height=300)

lines = [l.strip() for l in raw_text.splitlines() if l.strip()]


st.header("Extracted Tables")

tables = []


def looks_like_grid_header(line):
    words = line.split()
    if len(words) < 3:
        return False
    return all(w[0].isupper() for w in words if w.isalpha())

grid_header_index = None

for i, line in enumerate(lines):
    if looks_like_grid_header(line):
        grid_header_index = i
        break

if grid_header_index is not None:
    headers = lines[grid_header_index].split()
    rows = []

    for line in lines[grid_header_index + 1:]:
        cells = line.split()
        if len(cells) >= len(headers):
            rows.append(cells[:len(headers)])
        else:
            break

    if rows:
        df_grid = pd.DataFrame(rows, columns=headers)
        tables.append(("Grid Table", df_grid))


if not tables:

    def looks_like_key(line):
        return (
            len(line.split()) <= 4
            and re.match(r"^[A-Za-z ]+$", line)
            and line[0].isupper()
            and not re.fullmatch(r"[•\-vV]+", line)
        )

    schema = []
    rows = []
    current_key = None
    buffer = []
    inside_table = False

    for i, line in enumerate(lines):
        # detect header row like "Attribute Description"
        if not inside_table and looks_like_key(line) and i + 1 < len(lines):
            next_line = lines[i + 1]
            if looks_like_key(next_line):
                schema = [line, next_line]
                inside_table = True
                continue

        if not inside_table:
            continue

        if line in schema:
            continue

        if looks_like_key(line):
            if current_key:
                rows.append([current_key, " ".join(buffer).strip()])
            current_key = line
            buffer = []
        else:
            if current_key:
                buffer.append(line)

    if current_key:
        rows.append([current_key, " ".join(buffer).strip()])

    if rows and len(schema) == 2:
        df_kv = pd.DataFrame(rows, columns=schema)
        tables.append(("Key–Value Table", df_kv))


if not tables:
    st.warning("Could not infer any table structure from this page.")
else:
    for title, df in tables:
        st.subheader(title)
        st.dataframe(df)
