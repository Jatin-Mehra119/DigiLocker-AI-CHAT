import groq
from search import SearchEngine
import asyncio
import os
import dotenv
import tiktoken
import streamlit as st

dotenv.load_dotenv()

# Initialize the search engine
search_engine = SearchEngine()

def trim_text_to_token_limit(text, token_limit=6000):
    """Custom function to trim text using cl100k_base tokenizer"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's base tokenizer
        tokens = encoding.encode(text)
        if len(tokens) > token_limit:
            tokens = tokens[:token_limit]
            return encoding.decode(tokens)
        return text
    except Exception as e:
        print(f"Warning: Token counting failed, using character-based fallback: {e}")
        # Fallback to character-based trimming (rough approximation)
        char_limit = token_limit * 4  # Rough estimate of chars per token
        return text[:char_limit]

class LLM:
    def __init__(self):
        self.model_name = "llama-3.1-8b-instant"
        self.groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        self.system_prompt = (
            "You are DIGILOCKER AI CHATBOT EXPERT.\n"
            "You will answer the questions based on the provided context.\n"
            "If you don't know the answer, type 'I don't know'."
        )

    async def generate_response(self, context: list, query: str) -> str:
        # Combine only the top two retrieved texts to form a concise context
        context_text = "\n".join([item["text"] for item in context][:2])
        # Trim the combined context
        context_text = trim_text_to_token_limit(context_text, token_limit=6000)
        
        # Generate response
        response = await asyncio.to_thread(
            self._generate_response_sync, context_text, query
        )
        return response

    def _generate_response_sync(self, context: str, query: str) -> str:
        prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}"
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.2,
            max_tokens=6000
        )
        return response.choices[0].message.content

async def main():
    st.title("Digi Locker AI Chatbot Expert")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("https://img1.digitallocker.gov.in/digilocker-landing-page/assets/img/DigilockerLogo.svg", width=200)
    with col2:
        st.image("https://www.digilocker.gov.in/assets/img/chat-bot.svg", width=50)
    
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask something about DigiLocker..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Retrieve context using the search engine
                search_results = search_engine.search(prompt)
                context = search_results if search_results else [{"text": ""}]
                
                # Generate response
                llm = LLM()
                response = await llm.generate_response(context, prompt)
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            except Exception as e:
                error_message = "I apologize, but I encountered an issue processing your request."
                
                # Handle specific exceptions with friendly messages
                if "token" in str(e).lower():
                    error_message = "Something went wrong! The query used too many tokens."
                elif "api" in str(e).lower():
                    error_message = "I'm having trouble connecting to my knowledge base. Please try again later."
                elif "timeout" in str(e).lower():
                    error_message = "The request timed out. Please try a shorter or simpler question."
                
                # Log the actual error for debugging (won't be visible to users)
                print(f"Error generating response: {str(e)}")
                
                # Display the user-friendly error message
                message_placeholder.markdown(error_message)
                
                # Add error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    asyncio.run(main())
