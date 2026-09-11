import asyncio
import re
from typing import List, Dict, Any, Tuple
import numpy as np


class HungarianTextNormalizer:
    
    SUFFIXES = re.compile(r'(nak|nek|ban|ben|ról|ről|tól|től|hoz|hez|höz|val|vel|ba|be|ra|re|ú|ű|s|ok|ek|ök|ak|at|et|ot|öt)$', re.IGNORECASE)

    @classmethod
    def normalize(cls, text: str) -> List[str]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        stems = []
        for token in tokens:
            if len(token) > 3:
                token = cls.SUFFIXES.sub('', token)
            stems.append(token)
        return stems


class ReciprocalRankFusion:

    @staticmethod
    def fuse_ranks(bm25_results: List[str], dense_results: List[str], k: int = 60, top_n: int = 5) -> List[str]:
        scores: Dict[str, float] = {}

        for rank, doc in enumerate(bm25_results):
            scores[doc] = scores.get(doc, 0.0) + (1.0 / (k + rank + 1))

        for rank, doc in enumerate(dense_results):
            scores[doc] = scores.get(doc, 0.0) + (1.0 / (k + rank + 1))

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc for doc, score in sorted_docs[:top_n]]


class ContextCompressor:

    @staticmethod
    def compress(query: str, documents: List[str], max_chars: int = 1500) -> str:
        q_terms = set(HungarianTextNormalizer.normalize(query))
        scored_chunks = []

        for doc in documents:
            sentences = re.split(r'(?<=[.!?])\s+', doc)
            for sent in sentences:
                s_terms = set(HungarianTextNormalizer.normalize(sent))
                overlap = len(q_terms.intersection(s_terms))
                if overlap > 0:
                    scored_chunks.append((overlap, sent))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        compressed_text = []
        curr_len = 0
        for _, sent in scored_chunks:
            if curr_len + len(sent) > max_chars:
                break
            compressed_text.append(sent)
            curr_len += len(sent)

        return " ".join(compressed_text) if compressed_text else "\n".join(documents)[:max_chars]


class UltraHybridRetriever:

    def __init__(self, vector_db_client=None):
        self.db = vector_db_client
        self.normalizer = HungarianTextNormalizer()
        self.compressor = ContextCompressor()

    async def _bm25_sparse_search(self, query: str) -> List[str]:
        await asyncio.sleep(0.01)
        tokens = self.normalizer.normalize(query)
        return [f"Dokumentum részlet a következő kulcsszavakkal: {', '.join(tokens)}"]

    async def _vector_dense_search(self, query: str) -> List[str]:
        await asyncio.sleep(0.02)
        return [f"Vektoros hasonlóság alapján releváns kontextus a következő kérdéshez: '{query}'"]

    async def retrieve_and_compress(self, query: str) -> str:
        bm25_task = asyncio.create_task(self._bm25_sparse_search(query))
        dense_task = asyncio.create_task(self._vector_dense_search(query))

        bm25_res, dense_res = await asyncio.gather(bm25_task, dense_task)
        fused_docs = ReciprocalRankFusion.fuse_ranks(bm25_res, dense_res, top_n=3)
        
        return self.compressor.compress(query, fused_docs)