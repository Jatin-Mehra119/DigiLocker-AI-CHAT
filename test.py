import streamlit as st
import asyncio
from vector_database import VectorDatabase
from web_crawler import fetch_data


# Initialize Vector Database
vector_db = VectorDatabase()

def main():
    st.title("Async Web Crawler & Storage")
    
    url = st.text_input("Enter URL", "https://www.digilocker.gov.in/web/about/faq")
    
    if st.button("Crawl"):
        with st.spinner("Fetching data..."):
            result = asyncio.run(fetch_data(url))
            st.markdown(result)
            vector_db.add_data([result])
            st.success("Data successfully stored in FAISS!")

    if st.button("Show FAISS Index"):
        index_data = vector_db.view_index()
        st.json(index_data)


if __name__ == "__main__":
    main()
