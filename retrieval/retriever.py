from vector_store.store import VectorStore

class SemanticRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieves the top_k most similar chunks for the given query using
        the underlying VectorStore FAISS index.
        """
        return self.vector_store.search(query, top_k=top_k)
