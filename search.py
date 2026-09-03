import aiohttp
import asyncio

async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.text()
    except Exception:
        pass
    return ""

async def fetch_all_urls(urls: list[str]):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)