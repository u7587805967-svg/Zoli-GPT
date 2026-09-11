import asyncio
import time
import re
import json
import numpy as np
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from groq import Groq

@dataclass(frozen=True)
class EngineConfig:
    CACHE_SIMILARITY_THRESHOLD: float = 0.92  # Gyorsítótár egyezés küszöbértéke
    DEFAULT_EMBEDDING_MODEL: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    FAST_MODEL: str = "qwen/qwen3.8-27b"
    REASONING_MODEL: str = "groq/compound"
    MAX_CACHE_SIZE: int = 1000

cfg = EngineConfig()

class FastSemanticCache:
    """RAM-alapú szemantikus gyorsítótár vektoros hasonlóságméréssel."""
    
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.vectors: List[np.ndarray] = []
        self.responses: List[str] = []
        self.queries: List[str] = []

    def get(self, query: str) -> Optional[str]:
        if not self.vectors:
            return None
        
        q_vec = self.embedder.encode(query, normalize_embeddings=True)
        scores = np.dot(np.array(self.vectors), q_vec)
        best_idx = int(np.argmax(scores))
        
        if scores[best_idx] >= cfg.CACHE_SIMILARITY_THRESHOLD:
            return self.responses[best_idx]
        return None

    def set(self, query: str, response: str) -> None:
        if len(self.responses) >= cfg.MAX_CACHE_SIZE:
            self.vectors.pop(0)
            self.responses.pop(0)
            self.queries.pop(0)

        q_vec = self.embedder.encode(query, normalize_embeddings=True)
        self.vectors.append(q_vec)
        self.responses.append(response)
        self.queries.append(query)

class IntentType:
    DIRECT = "direct"               # Egyszerű tények, köszöntések -> ultra-gyors modell
    MATHEMATICAL = "math"           # Számítások -> determinisztikus Python korrekció
    RAG_SEARCH = "rag"              # Helyi doksi / webes keresés -> Hibrid RAG
    DEEP_REASONING = "reasoning"   # Összetett elemzés -> Multi-stage reasoning loop

class AdaptiveIntentRouter:
    """Minimális overhead-del rendelkező dinamikus útválasztó."""
    
    MATH_PATTERNS = re.compile(r'(\d+[\+\-\*/\^\(\)]+\d+|számold|mennyi|kiszámítani|egyenlet)')
    DEEP_PATTERNS = re.compile(r'(bizonyítsd|elemzés|tervezz|optimalizáld|hasonlítsd össze|miért|kód)')

    @classmethod
    def classify(cls, query: str) -> str:
        q_lower = query.lower()
        
        if cls.MATH_PATTERNS.search(q_lower):
            return IntentType.MATHEMATICAL
        elif cls.DEEP_PATTERNS.search(q_lower) or len(query) > 200:
            return IntentType.DEEP_REASONING
        elif len(query) < 25 and not any(kw in q_lower for kw in ["hol", "ki", "mi", "mikor"]):
            return IntentType.DIRECT
        return IntentType.RAG_SEARCH

class ZoliUltraEngine:
    def __init__(self, groq_api_key: str, embedder: SentenceTransformer):
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.embedder = embedder
        self.cache = FastSemanticCache(self.embedder)
        self.router = AdaptiveIntentRouter()

    async def _execute_math_shortcut(self, query: str) -> Optional[str]:
        """Azonnali determinisztikus matek kiértékelés (LLM kihagyásával, ha lehetséges)."""
        expr_match = re.search(r'(\d+[\d\s\+\-\*/\^\(\)\.,]+\d+)', query)
        if expr_match:
            clean_expr = expr_match.group(1).replace(',', '.').strip()
            try:
                allowed_chars = set("0123456789+-*/(). ")
                if all(c in allowed_chars for c in clean_expr):
                    result = eval(clean_expr, {"__builtins__": {}}, {})
                    return f"**Determinisztikus Eredmény:** {clean_expr} = `{result}`"
            except Exception:
                pass
        return None

    async def stream_response(
        self, 
        user_query: str, 
        rag_context_provider=None, 
        web_search_provider=None
    ) -> AsyncGenerator[str, None]:
        
        start_time = time.time()

        cached_resp = self.cache.get(user_query)
        if cached_resp:
            yield f"[Szemantikus Gyorsítótár - Latencia: {round((time.time() - start_time)*1000, 1)}ms]\n\n"
            yield cached_resp
            return

        intent = self.router.classify(user_query)
        
        if intent == IntentType.MATHEMATICAL:
            math_result = await self._execute_math_shortcut(user_query)
            if math_result:
                self.cache.set(user_query, math_result)
                yield math_result
                return

        rag_task = asyncio.create_task(rag_context_provider(user_query)) if rag_context_provider else asyncio.sleep(0)
        web_task = asyncio.create_task(web_search_provider(user_query)) if web_search_provider else asyncio.sleep(0)

        results = await asyncio.gather(rag_task, web_task, return_exceptions=True)
        
        local_docs = results[0] if isinstance(results[0], str) else ""
        web_docs = results[1] if isinstance(results[1], str) else ""

        combined_context = ""
        if local_docs:
            combined_context += f"\nHELYI DOKUMENTUMOK:\n{local_docs}"
        if web_docs:
            combined_context += f"\nWEB KERESÉSI TALÁLATOK:\n{web_docs}"

        selected_model = cfg.REASONING_MODEL if intent == IntentType.DEEP_REASONING else cfg.FAST_MODEL

        system_prompt = f"""
System: ZoliGPT-Ultra Engine v3.0.
Dátum: {time.strftime('%Y-%m-%d')}
Kontextus:
{combined_context if combined_context else "Nincs külső kontextus."}

UTASÍTÁSOK:
1. Közvetlenül a lényegre térj.
2. Kizárólag a kontextusra vagy ellenőrzött tényekre támaszkodj!
"""

        if not self.groq_client:
            yield "Hiba: Hiányzó API kulcs."
            return

        full_response = []
        try:
            stream = self.groq_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1 if intent == IntentType.DEEP_REASONING else 0.0,
                max_tokens=2500,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response.append(content)
                    yield content

            final_text = "".join(full_response)
            if len(final_text) > 20:
                self.cache.set(user_query, final_text)

        except Exception as e:
            yield f"\nHiba történt a generálás során: {str(e)}"