"""
retriever.py
------------
Recuperación por similitud: vectoriza la pregunta con el mismo modelo que
indexó el corpus y devuelve los `top_k` fragmentos más cercanos (coseno).
"""

from typing import List

from embeddings import Embedder
from vector_store import RetrievedChunk, VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        return self.store.query(self.embedder.embed_query(query), top_k or self.top_k)
