import groq
import asyncio
import os
from src.utils.common import trim_text_to_token_limit

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
        context_text = "\n".join([item["text"] for item in context][:2])
        context_text = trim_text_to_token_limit(context_text, token_limit=6000)
        
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