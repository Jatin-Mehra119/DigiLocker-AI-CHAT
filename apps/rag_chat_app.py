import streamlit as st
import asyncio
import sys
import os
import dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search.search_engine import SearchEngine
from src.rag.llm import LLM

dotenv.load_dotenv()

# Set consistent storage path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VECTOR_DB_PATH = os.path.join(DATA_DIR, 'vector_db')
os.makedirs(DATA_DIR, exist_ok=True)
search_engine = SearchEngine(storage_path=VECTOR_DB_PATH)

async def main():
    st.title("Digi Locker AI Chatbot Expert")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("https://img1.digitallocker.gov.in/digilocker-landing-page/assets/img/DigilockerLogo.svg", width=200)
    with col2:
        st.image("https://www.digilocker.gov.in/assets/img/chat-bot.svg", width=50)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask something about DigiLocker..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                search_results = search_engine.search(prompt)
                context = search_results if search_results else [{"text": ""}]
                llm = LLM()
                response = await llm.generate_response(context, prompt)
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = "I apologize, but I encountered an issue processing your request."
                if "token" in str(e).lower():
                    error_message = "Something went wrong! The query used too many tokens."
                elif "api" in str(e).lower():
                    error_message = "I'm having trouble connecting to my knowledge base. Please try again later."
                elif "timeout" in str(e).lower():
                    error_message = "The request timed out. Please try a shorter or simpler question."
                print(f"Error generating response: {str(e)}")
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    asyncio.run(main())