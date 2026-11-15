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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chromadb

from chunking import Chunk, chunk_corpus
from config import CHROMA_DIR, KNOWLEDGE_BASE_DIR, ROOT
from embeddings import Embedder
from ingest import SUPPORTED_EXTENSIONS, corpus_fingerprint, load_documents

PREBUILT_DIR = ROOT / "index_prebuilt"   # índice ya calculado (se versiona; la nube no reconstruye nada)
PREBUILT_NAME = "kb_prebuilt"


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


# ---------------------------------------------------------------- índice precalculado (para el despliegue)
def has_prebuilt(path: Path = PREBUILT_DIR) -> bool:
    if not (Path(path) / "chroma.sqlite3").exists():
        return False
    try:
        return chromadb.PersistentClient(path=str(path)).get_collection(PREBUILT_NAME).count() > 0
    except Exception:
        return False


def build_prebuilt_store(embedder: Embedder, chunk_size: int, overlap: int, kb_dir: Path = KNOWLEDGE_BASE_DIR,
                         path: Path = PREBUILT_DIR, progress=None, batch_size: int = 64) -> VectorStore:
    """Construye el índice del corpus en `path` (una sola colección). Se ejecuta en local y se sube al repo."""
    documents = load_documents(kb_dir)
    if not documents:
        raise RuntimeError(f"No se encontraron documentos soportados en {kb_dir}")
    chunks = chunk_corpus(documents, chunk_size, overlap)
    client = chromadb.PersistentClient(path=str(path))
    try:
        client.delete_collection(PREBUILT_NAME)
    except Exception:
        pass
    collection = client.create_collection(PREBUILT_NAME, metadata={
        "hnsw:space": "cosine", "chunk_size": chunk_size, "overlap": overlap, "embedder": embedder.name, "n_chunks": len(chunks)})
    store = VectorStore(collection)
    for start in range(0, len(chunks), batch_size):
        store.add_chunks(chunks[start:start + batch_size], embedder, batch_size=batch_size)
        if progress:
            progress(min(start + batch_size, len(chunks)), len(chunks))
    return store


def load_store_auto(embedder: Embedder, chunk_size: int, overlap: int, kb_dir: Path = KNOWLEDGE_BASE_DIR,
                    persist_dir: Path = CHROMA_DIR, prebuilt_dir: Path = PREBUILT_DIR, rebuild: bool = False) -> VectorStore:
    """Usa el índice precalculado cuando no hay documentos en knowledge_base/ (despliegue) o si INDEX_MODE=prebuilt;
    en cualquier otro caso construye/reutiliza el índice local como antes (desarrollo y evaluaciones)."""
    mode = os.getenv("INDEX_MODE", "auto").lower()
    kb = Path(kb_dir)
    has_docs = kb.exists() and any(p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS for p in kb.rglob("*"))
    if mode == "prebuilt" or (mode == "auto" and not has_docs):
        if has_prebuilt(prebuilt_dir):
            collection = chromadb.PersistentClient(path=str(prebuilt_dir)).get_collection(PREBUILT_NAME)
            built_with = (collection.metadata or {}).get("embedder")
            if built_with and built_with != embedder.name:
                raise RuntimeError(f"El índice precalculado se creó con embeddings '{built_with}' y ahora se usa '{embedder.name}'.")
            return VectorStore(collection)
        if mode == "prebuilt":
            raise RuntimeError("INDEX_MODE=prebuilt pero no existe index_prebuilt/. Ejecuta: python scripts/build_index.py --prebuilt")
    return load_or_build_store(embedder, chunk_size, overlap, kb_dir=kb_dir, persist_dir=persist_dir, rebuild=rebuild)
