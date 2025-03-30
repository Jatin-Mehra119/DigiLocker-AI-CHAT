import asyncio
from crawl4ai import AsyncWebCrawler, LLMExtractionStrategy, CrawlerRunConfig, CacheMode, BrowserConfig
import os


async def fetch_data(url: str, config=None) -> str:
    """
    Asynchronously fetches data from a given URL using AsyncWebCrawler.
    
    Args:
        url (str): The URL to fetch data from.
    
    Returns:
        str: Extracted content in markdown format.
    """
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(url=url, config=config)
            return result.markdown if result else "No data found"
        except Exception as e:
            return f"Error fetching data: {e}"

async def fetch_using_LLM(url:str) -> str:
    """
    Asynchronously fetches data from a given URL using AsyncWebCrawler and LLMExtractionStrategy.
    
    Args:
        url (str): The URL to fetch data from.
    
    Returns:
        str: Extracted content in markdown format.
    """
    extraction_strategy = LLMExtractionStrategy(
        provider="groq",
        model_name="llama-3.1-8b-instant",
        api_token=os.getenv("GROQ_API_KEY"),
        extraction_type="schema",
        schema={"type" : "object", "properties": {"content": {"type": "string"}}},
        instruction=f"Extract the main content avoid navigation, ads, and other noise from {url}",
        chunk_token_threshold=1200,
        apply_chunking=True,
        extra_args={"temperature": 0.1, "max_tokens": 2500}
    )

    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, extraction_strategy=extraction_strategy)
    return await fetch_data(url=url, config=config)