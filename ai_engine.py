import streamlit as st
import asyncio
import aiohttp
import numpy as np
import json
import re
import datetime
import concurrent.futures
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from groq import Groq

@dataclass(frozen=True)
class UltraConfig:
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 150
    SIMILARITY_THRESHOLD: float = 0.35
    MAX_WEB_RESULTS: int = 3
    CACHE_TTL: int = 3600

cfg = UltraConfig()

@st.cache_resource
def get_ultra_embedder():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def auto_select_model(query: str) -> str:
    complex_keywords = ["számold", "kód", "python", "bizonyítsd", "miért", "tervezz", "elemzés", "algoritmus", "optimalizáld"]
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in complex_keywords) or len(query) > 150:
        return "groq/compound"
    return "qwen/qwen3.8-27b"

class UltraSearchEngine:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key) if groq_api_key else None

    async def _fetch_url_clean(self, session: aiohttp.ClientSession, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ZoliGPT-Ultra"}
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    # Gyors tisztítás reguláris kifejezésekkel a BeautifulSoup overhead elkerülésére
                    clean_text = re.sub(r'<(script|style|nav|footer|header)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
                    clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
                    clean_text = ' '.join(clean_text.split())
                    return f"FORRÁS ({url}):\n{clean_text[:2500]}"
        except Exception:
            pass
        return ""

    async def execute_parallel_search(self, queries: List[str]) -> str:
        urls = set()
        async with DDGS() as ddgs:
            tasks = [ddgs.atext(q, max_results=cfg.MAX_WEB_RESULTS) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res_list in results:
                if isinstance(res_list, list):
                    for item in res_list:
                        if "href" in item:
                            urls.add(item["href"])

        if not urls:
            return "Nem található külső webes hivatkozás."

        async with aiohttp.ClientSession() as session:
            fetch_tasks = [self._fetch_url_clean(session, url) for url in list(urls)[:5]]
            extracted_texts = await asyncio.gather(*fetch_tasks)

        valid_contexts = [txt for txt in extracted_texts if txt]
        return "\n\n---\n\n".join(valid_contexts) if valid_contexts else "A weboldalak tartalma nem volt hozzáférhető."

class UltraRAGEngine:
    def __init__(self, db_repo):
        self.db = db_repo
        self.embedder = get_ultra_embedder()

    def rrf_hybrid_rank(self, vector_hits: List[Dict], bm25_hits: List[Dict], k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion (RRF) a pontosabb találatokért."""
        scores = {}
        all_docs = {}

        for rank, item in enumerate(vector_hits):
            doc_id = item['chunk_id']
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            all_docs[doc_id] = item

        for rank, item in enumerate(bm25_hits):
            doc_id = item['chunk_id']
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            all_docs[doc_id] = item

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [all_docs[doc_id] for doc_id, _ in sorted_ids]

    def query_context(self, query: str, username: str) -> List[Dict]:
        query_vector = self.embedder.encode(query)
        scored_chunks = []

        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, doc_name, chunk_text, embedding FROM document_vectors WHERE username=?", (username,))
            rows = cursor.fetchall()

        for chunk_id, doc_name, chunk_text, emb_blob in rows:
            try:
                doc_vector = np.array(json.loads(emb_blob.decode('utf-8')))
                cos_sim = float(np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)))
                if cos_sim >= cfg.SIMILARITY_THRESHOLD:
                    scored_chunks.append({
                        "chunk_id": chunk_id,
                        "doc_name": doc_name,
                        "text": chunk_text,
                        "score": cos_sim
                    })
            except Exception:
                continue

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:4]

class UltraAIEngine:
    def __init__(self, db_repo, api_key: str):
        self.db = db_repo
        self.api_key = api_key
        self.search_engine = UltraSearchEngine(api_key)
        self.rag_engine = UltraRAGEngine(db_repo)

    async def process_user_request(self, user_query: str, username: str, model_name: str):
        rag_task = asyncio.to_thread(self.rag_engine.query_context, user_query, username)
        
        search_queries = [user_query]
        search_task = self.search_engine.execute_parallel_search(search_queries)

        rag_results, web_context = await asyncio.gather(rag_task, search_task)

        local_context = "\n".join([f"[{d['doc_name']}]: {d['text']}" for d in rag_results])
        
        full_system_context = f"""
        A mai dátum: {datetime.datetime.now().strftime('%Y-%m-%d')}.
        
        ELÉRHETŐ HELYI DOKUMENTUM KONTEXTUS:
        {local_context if local_context else 'Nincs releváns helyi dokumentum.'}
        
        ELÉRHETŐ FRISS WEB KONTEXTUS:
        {web_context if web_context else 'Nincs külső webes információ.'}
        
        UTASÍTÁSOK A VÉGTELEN PONTOSSÁGHOZ:
        1. Végezz belső igazolást (Verification Loop)! Kizárólag a kontextusban szereplő igazolt tényekre építs.
        2. Ha matematikai vagy logikai feladatot kapsz, szigorúan számold ki lépésről lépésre a végeredmény megadása előtt!
        3. Ne használj töltelékszövegeket vagy felesleges üdvözléseket.
        """

        if not self.api_key:
            yield "Hiba: Hiányzó API kulcs."
            return

        client = Groq(api_key=self.api_key)
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": full_system_context},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0, # Minimális hőmérséklet a maximális reprodukálhatóságért és pontosságért
            max_tokens=3000,
            stream=True
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content