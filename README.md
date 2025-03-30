# DigiLocker RAG Chatbot Project
- [AI CHAT BOT](https://digilocker-ai-chat.streamlit.app/)
## Overview
This project crawls websites and processes PDFs to build a knowledge base in a FAISS vector database, then powers a RAG chat app for DigiLocker Q&A.

## Technology
- Web Scraping powered by AI (Crawl4ai)
- Knowledge Base (Faiss by Facebook)
- AI insights from GROQ APIs.
- Frontend Streamlit UI
- Pdf processing using PyPDF2

```
GOV/
├── apps/
│   ├── data_collector_app.py/# Website crawler/scraper using crawl4ai
│   └── rag_chat_app.py       # Main Streamlit application
├── data/
│   ├── crawled/              # Crawled website data
│   ├── pdfs/                 # Source PDF documents
│   └── processed/            # Processed text chunks
├── models/
│   └── faiss/                # FAISS vector database files
├── scripts/
│   ├── crawler.py            # Web crawling functionality
│   ├── pdf_processor.py      # PDF extraction and processing
│   └── build_knowledge_base.py # Knowledge base construction
├── utils/
│   ├── embedding.py          # Text embedding utilities
│   └── search_engine.py      # Vector search implementation
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

### Acknowledgments
- DigiLocker for providing the official documentation.
