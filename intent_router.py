import re
from typing import Dict, Any

class IntentRouter:
    
    @staticmethod
    @lru_cache(maxsize=2048)
    def classify_intent(query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        
        if re.search(r'[\d\+\-\*/\^\(\)]', query) and any(op in query for op in ['+', '-', '*', '/', '=', 'számold', 'mennyi']):
            return {"intent": "MATH", "strategy": "deterministic_execution", "use_ensemble": False}
        
        fact_keywords = ['mi a', 'ki az', 'mikor', 'hol', 'forrás', 'szabályzat', 'dokumentum', 'keresd']
        if any(kw in query_lower for kw in fact_keywords) or len(query.split()) > 6:
            return {"intent": "FACTUAL_RAG", "strategy": "hybrid_search_with_rerank", "use_ensemble": True}
        
        if any(kw in query_lower for kw in ['python', 'code', 'hiba', 'függvény', 'script', 'def ', 'class ']):
            return {"intent": "CODING", "strategy": "strict_syntax_generation", "use_ensemble": False}
        
        return {"intent": "CHAT", "strategy": "direct_llm", "use_ensemble": False}