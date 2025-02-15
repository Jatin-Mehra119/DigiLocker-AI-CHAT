import groq
from search import SearchEngine
import asyncio
import os
import dotenv

dotenv.load_dotenv()

# Initialize the search engine
search_engine = SearchEngine()

class LLM:
    def __init__(self):
        self.model = groq.LLAMA()
        self.model_name = "llama-3.1-8b-instant"
        self.groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        self.system_prompt = (
            "You are DIGILOCKER AI CHATBOT EXPERT.\n"
            "You will answer the questions based on the provided context.\n"
            "If you don't know the answer, type 'I don't know'."
        )

    async def generate_response(self, context, query):
        # Combine the system prompt, the retrieved context, and the query to form the complete prompt.
        prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}"
        # Call the groq chat completion API asynchronously in a thread.
        response = await asyncio.to_thread(
            self.groq_client.chat.completions.create,
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.2,
            max_tokens=4000
        )
        return response

async def main():
    query = input("Please enter your query: ")
    # Retrieve context using the search engine
    context = search_engine.search(query)
    llm = LLM()
    response = await llm.generate_response(context, query)
    print("Response:\n", response)

if __name__ == "__main__":
    asyncio.run(main())

