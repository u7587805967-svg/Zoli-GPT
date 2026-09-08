from typing import List
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class ContextReranker:
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.reranker = CrossEncoder(model_name) if CrossEncoder else None

    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[str]:
        if not self.reranker or not documents:
            return documents[:top_k]
        
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs if score > 0.0][:top_k]