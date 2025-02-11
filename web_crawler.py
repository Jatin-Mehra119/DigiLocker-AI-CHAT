import asyncio
from crawl4ai import AsyncWebCrawler


async def fetch_data(url: str) -> str:
    """
    Asynchronously fetches data from a given URL using AsyncWebCrawler.
    
    Args:
        url (str): The URL to fetch data from.
    
    Returns:
        str: Extracted content in markdown format.
    """
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(url=url)
            return result.markdown if result else "No data found"
        except Exception as e:
            return f"Error fetching data: {e}"
