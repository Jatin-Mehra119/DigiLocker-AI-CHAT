import streamlit as st
import asyncio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_db.vector_database import VectorDatabase
from src.data_collection.web_crawler import fetch_using_LLM
from src.data_collection.pdf_processor import extract_text_from_pdf

# Set consistent storage path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VECTOR_DB_PATH = os.path.join(DATA_DIR, 'vector_db')
os.makedirs(DATA_DIR, exist_ok=True)
vector_db = VectorDatabase(storage_path=VECTOR_DB_PATH)

def main():
    st.title("Async Web Crawler & Storage")
    
    url = st.text_input("Enter URL", "https://www.digilocker.gov.in/web/about/faq")
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Crawl"):
        with st.spinner("Fetching data..."):
            result = asyncio.run(fetch_using_LLM(url))
            st.markdown(result)
            vector_db.add_data([result])
            st.success("Data successfully stored in FAISS!")
    
    if pdf_file and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(pdf_file)
            st.markdown(text)
            vector_db.add_data([text])
            st.success("PDF successfully stored in FAISS!")
    
    if st.button("Show FAISS Index"):
        index_data = vector_db.view_index()
        st.json(index_data)

if __name__ == "__main__":
    main()