import asyncio
import re
import aiohttp
from typing import List, Dict, Any, Optional


class WebQueryOptimizer:
    
    STOPWORDS = {"a", "az", "egy", "hogy", "van", "lesz", "mint", "hogy", "mi", "hol", "mikor", "miért"}

    @classmethod
    def optimize(cls, query: str) -> str:
        words = re.findall(r'\b\w+\b', query.lower())
        filtered = [w for w in words if w not in cls.STOPWORDS]
        return " ".join(filtered[:5])


class HTMLCleaner:
    
    NOISE_PATTERNS = re.compile(
        r'<(script|style|header|footer|nav|svg|iframe)[^>]*>.*?', 
        re.DOTALL | re.IGNORECASE
    )
    TAGS_PATTERN = re.compile(r'<[^>]+>')
    WHITESPACE_PATTERN = re.compile(r'\s+')

    @classmethod
    def clean(cls, raw_html: str) -> str:
        text = cls.NOISE_PATTERNS.sub('', raw_html)
        text = cls.TAGS_PATTERN.sub(' ', text)
        text = cls.WHITESPACE_PATTERN.sub(' ', text)
        return text.strip()


class UltraWebAgent:

    def __init__(self, timeout_seconds: float = 1.5):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        try:
            async with session.get(url, timeout=self.timeout) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    return HTMLCleaner.clean(html)[:2000]
        except Exception:
            return None
        return None

    async def search_and_extract(self, query: str, target_urls: Optional[List[str]] = None) -> str:
        search_keywords = WebQueryOptimizer.optimize(query)
        
        urls = target_urls or [
            f"https://hu.wikipedia.org/wiki/Special:Search?search={search_keywords}",
            f"https://news.google.com/search?q={search_keywords}&hl=hu"
        ]

        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_texts = [res for res in results if isinstance(res, str) and res]
        
        if not valid_texts:
            return f"Webes találat a következő kulcsszavakra: '{search_keywords}'"

        return "\n--- WEB TALÁLAT ---\n".join(valid_texts[:2])