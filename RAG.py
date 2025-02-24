import groq
from search import SearchEngine
import asyncio
import os
import dotenv
import tiktoken

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
    query = input("Please enter your query: ")
    # Retrieve context using the search engine (default returns 5 results)
    search_results = search_engine.search(query) 
    # If no results found, use an empty string for context.
    context = search_results if search_results else [{"text": ""}]
    llm = LLM()
    response = await llm.generate_response(context, query)
    print("Response:\n", response)

if __name__ == "__main__":
    asyncio.run(main())
