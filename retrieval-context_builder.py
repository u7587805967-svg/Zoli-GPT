from sentence_transformers import CrossEncoder

class DynamicMemoryRAG:
    def __init__(self, vector_db, bm25_index, graph_db):
        self.vector_db = vector_db
        self.bm25 = bm25_index
        self.graph_db = graph_db
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query: str, user_id: str) -> str:
        graph_entities = self.graph_db.get_related_nodes(query)

        dense_hits = self.vector_db.search(query, k=10)
        sparse_hits = self.bm25.search(query, k=10)
        candidates = list(set(dense_hits + sparse_hits))

        pairs = [[query, doc.text] for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked_docs = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]

        context = f"Gráf entitások: {graph_entities}\nDokumentumok: " + "\n".join([d.text for d in ranked_docs[:3]])
        return context