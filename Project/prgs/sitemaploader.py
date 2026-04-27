import re
import asyncio
from bs4 import BeautifulSoup
from langchain_community.document_loaders.sitemap import SitemapLoader
import streamlit as st
#import nest_asyncio

# --- CRITICAL FIX FOR STREAMLIT ---
# Streamlit already runs an event loop. This patches it to allow 
# SitemapLoader (which uses asyncio) to run inside it.
#nest_asyncio.apply()

# --- BS4 EXTRACTOR FUNCTION ---
def bs4_extractor(html: str) -> str:
    """Extract clean text from HTML using BeautifulSoup"""
    soup = BeautifulSoup(html, "lxml")
    # Remove script and style elements to clean up output
    for script in soup(["script", "style"]):
        script.extract()
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

# --- APP INTERFACE ---
st.title("Web Page Loader Example")

# --- LOADER CONFIGURATION ---
# We use FastAPI docs because they are static and do not block scrapers.
# OpenAI requires a headless browser (Selenium/Playwright), which SitemapLoader does not support natively.
sitemap_loader = SitemapLoader(
    web_path="https://fastapi.tiangolo.com/sitemap.xml",
    filter_urls=["https://fastapi.tiangolo.com/tutorial/"], # Filter to load only the "Tutorial" section
    parsing_function=bs4_extractor # Using your custom cleaner
)

# Optional: Slow down requests to be a polite scraper
sitemap_loader.requests_per_second = 2

# Loading with a spinner for better UX
with st.spinner("Scraping sitemap..."):
    try:
        docs = sitemap_loader.load()
        st.success(f"Successfully loaded {len(docs)} documents.")
    except Exception as e:
        st.error(f"Error loading sitemap: {e}")
        docs = []

# --- DISPLAY OUTPUT ---
if docs:
    for i, doc in enumerate(docs): # Limiting to first 5 for performance
        st.subheader(f"Section {i+1}: {doc.metadata.get('title', 'Untitled')}")
        st.write(doc.page_content)
        st.caption(f"Source: {doc.metadata.get('source')}")
        st.divider()

    st.subheader("Metadata (First Doc)")
    st.json(docs[0].metadata)
else:
    st.warning("No documents loaded")