import streamlit as st
import asyncio
from vector_database import VectorDatabase
from web_crawler import fetch_data, fetch_using_LLM
import PyPDF2

# Initialize Vector Database
vector_db = VectorDatabase()

def main():
    st.title("Async Web Crawler & Storage")
    
    # URL input for web crawling
    url = st.text_input("Enter URL", "https://www.digilocker.gov.in/web/about/faq")
    
    # PDF file uploader
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Crawl"):
        with st.spinner("Fetching data..."):
            result = asyncio.run(fetch_using_LLM(url))
            st.markdown(result)
            vector_db.add_data([result])
            st.success("Data successfully stored in FAISS!")
    
    if pdf_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                # Read PDF and extract text
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                st.markdown(text)
                vector_db.add_data([text])
                st.success("PDF successfully stored in FAISS!")
    
    if st.button("Show FAISS Index"):
        index_data = vector_db.view_index()
        st.json(index_data)

if __name__ == "__main__":
    main()
