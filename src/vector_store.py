"""
vector_store.py
---------------
Base de datos vectorial con Chroma (persistente en disco).

* Cada combinación (modelo de embeddings, chunk_size, overlap, contenido del
  corpus) usa su propia colección. Si el corpus cambia, el hash cambia y se
  crea un índice nuevo automáticamente; si no cambia, se reutiliza el
  existente. Esto permite comparar configuraciones en la evaluación sin
  mezclar índices.
* Metadatos por fragmento: fuente (archivo), título, sección, página
  (-1 si no aplica) e índice. Son los que permiten citar la fuente.
* Distancia coseno; el puntaje que se reporta es similitud = 1 - distancia.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chromadb

from chunking import Chunk, chunk_corpus
from config import CHROMA_DIR, KNOWLEDGE_BASE_DIR
from embeddings import Embedder
from ingest import corpus_fingerprint, load_documents


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    page: Optional[int]
    score: float          # similitud coseno (mayor = más parecido)
    rank: int             # 1 = más relevante


class VectorStore:
    def __init__(self, collection):
        self.collection = collection

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: List[Chunk], embedder: Embedder, batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embedder.embed_documents([c.embed_text for c in batch]),
                metadatas=[{
                    "source": c.source,
                    "title": c.title,
                    "section": c.section,
                    "page": c.page if c.page is not None else -1,
                    "chunk_index": c.index,
                } for c in batch],
            )

    def query(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self.count(), 1)),
        )
        retrieved = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        ), start=1):
            retrieved.append(RetrievedChunk(
                chunk_id=cid,
                text=doc,
                source=meta["source"],
                title=meta["title"],
                section=meta["section"],
                page=meta["page"] if meta["page"] != -1 else None,
                score=round(1.0 - float(dist), 4),
                rank=rank,
            ))
        return retrieved


def collection_name(embedder: Embedder, chunk_size: int, overlap: int, fingerprint: str) -> str:
    return f"kb_{embedder.name}_c{chunk_size}_o{overlap}_{fingerprint[:8]}"


def load_or_build_store(
    embedder: Embedder,
    chunk_size: int,
    overlap: int,
    kb_dir: Path = KNOWLEDGE_BASE_DIR,
    persist_dir: Path = CHROMA_DIR,
    rebuild: bool = False,
) -> VectorStore:
    """Devuelve el índice para esta configuración, construyéndolo si no existe."""
    fingerprint = corpus_fingerprint(kb_dir)
    name = collection_name(embedder, chunk_size, overlap, fingerprint)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    store = VectorStore(collection)

    if store.count() == 0:
        documents = load_documents(kb_dir)
        if not documents:
            raise RuntimeError(f"No se encontraron documentos soportados en {kb_dir}")
        chunks = chunk_corpus(documents, chunk_size, overlap)
        store.add_chunks(chunks, embedder)
    return store
