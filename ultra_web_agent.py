import asyncio
import re
from urllib.parse import parse_qs, urljoin, urlparse
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

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; ZoliGPT/1.0)"}

    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        try:
            async with session.get(url, timeout=self.timeout, headers=self.headers) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    return HTMLCleaner.clean(html)[:2000]
        except Exception:
            return None
        return None

    @staticmethod
    def parse_search_results(raw_html: str, limit: int = 5) -> List[Dict[str, str]]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_html, "html.parser")
        results = []
        for result in soup.select(".result"):
            title_link = result.select_one("a.result__a")
            snippet_node = result.select_one(".result__snippet")
            if not title_link:
                continue

            title = title_link.get_text(" ", strip=True)
            url = urljoin("https://duckduckgo.com", title_link.get("href", ""))
            parsed_url = urlparse(url)
            if parsed_url.hostname and parsed_url.hostname.endswith("duckduckgo.com"):
                url = parse_qs(parsed_url.query).get("uddg", [""])[0]
            if urlparse(url).scheme not in {"http", "https"}:
                continue

            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            if title and snippet:
                results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= limit:
                break
        return results

    async def search_and_extract(self, query: str, target_urls: Optional[List[str]] = None) -> str:
        async with aiohttp.ClientSession() as session:
            if target_urls:
                pages = await asyncio.gather(
                    *(self._fetch_url(session, url) for url in target_urls[:5]),
                    return_exceptions=True,
                )
                return "\n\n".join(
                    f"[W{index}] Forrás: {url}\nKivonat: {page}"
                    for index, (url, page) in enumerate(zip(target_urls, pages), start=1)
                    if isinstance(page, str) and page
                )

            try:
                async with session.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    timeout=self.timeout,
                    headers=self.headers,
                ) as response:
                    if response.status != 200:
                        return ""
                    raw_html = await response.text()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return ""

        results = self.parse_search_results(raw_html)
        return "\n\n".join(
            f"[W{index}] {item['title']}\nForrás: {item['url']}\nKivonat: {item['snippet']}"
            for index, item in enumerate(results, start=1)
        )