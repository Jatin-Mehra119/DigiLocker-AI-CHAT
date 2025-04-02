# DigiLocker RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot that answers questions about DigiLocker by crawling websites and processing PDFs to build a comprehensive knowledge base.

[Live Demo](https://digilocker-ai-chat.streamlit.app/)

![DigiLocker Logo](https://img1.digitallocker.gov.in/digilocker-landing-page/assets/img/DigilockerLogo.svg)

## Overview

This project implements a conversational AI assistant specialized in providing information about DigiLocker, India's digital documentation platform. The system:

1. **Collects Data**: Crawls official DigiLocker websites and processes PDF documentation
2. **Builds Knowledge Base**: Creates vector embeddings stored in a FAISS database
3. **Retrieves Information**: Searches for relevant context when questions are asked
4. **Generates Responses**: Uses GROQ's Llama 3.1 model to create accurate, contextual answers

## Technologies Used

- **Web Crawling**: crawl4ai for AI-powered website extraction
- **Vector Database**: FAISS by Facebook Research for efficient similarity search
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **LLM Integration**: GROQ API with Llama 3.1 model
- **PDF Processing**: PyPDF2 for document text extraction
- **Frontend**: Streamlit UI framework

## Installation & Usage

1. **Clone the repository**
   ```
   git clone https://github.com/Jatin-Mehra119/DigiLocker-AI-CHAT.git
   cd DigiLocker-AI-CHAT
   ```
2. **Install dependencies**

```
pip install -r requirements.txt
```

3. **Environment setup Create a .env file with your API keys:**
```
GROQ_API_KEY=your_groq_api_key_here
```
4. Run the application

- Data Collection App: For Adding more knowledge base/Data from other sources if you want
```
streamlit run apps/rag_chat_app.py
```
- Chatbot App: To chat with the model.

```
streamlit run apps/rag_chat_app.py
```
## **Features**
- **Intelligent Web Crawling**: Extracts relevant information from DigiLocker websites
- **PDF Knowledge Extraction**: Processes official documentation PDFs
- **Vector-based Retrieval**: Finds most relevant information for user queries
- **Contextual Response Generation**: Provides accurate answers based on retrieved information
- **User-friendly Interface**: Easy-to-use chat interface built with Streamlit

## **Acknowledgments**
- DigiLocker for providing the official documentation
- Facebook Research for FAISS vector database
- GROQ for the LLM API
- Sentence-Transformers team for embedding models
