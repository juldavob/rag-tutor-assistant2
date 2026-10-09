"""
Construye (o reutiliza) el índice vectorial del corpus y muestra estadísticas.

Uso:
    python scripts/build_index.py
    python scripts/build_index.py --chunk-size 300 --overlap 30 --rebuild
"""
import _bootstrap  # noqa: F401
import argparse
import time

from chunking import chunk_corpus
from config import KNOWLEDGE_BASE_DIR, RETRIEVAL_CONFIG
from embeddings import get_embedder
from ingest import load_documents
from vector_store import load_or_build_store

parser = argparse.ArgumentParser()
parser.add_argument("--chunk-size", type=int, default=RETRIEVAL_CONFIG.chunk_size)
parser.add_argument("--overlap", type=int, default=RETRIEVAL_CONFIG.chunk_overlap)
parser.add_argument("--rebuild", action="store_true", help="borra y reconstruye el índice")
args = parser.parse_args()

documents = load_documents(KNOWLEDGE_BASE_DIR)
chunks = chunk_corpus(documents, args.chunk_size, args.overlap)
lengths = [len(c.text) for c in chunks]
print(f"Corpus: {len({d.source for d in documents})} archivos, {len(documents)} documentos/páginas")
print(f"Chunking: tamaño={args.chunk_size} solapamiento={args.overlap} -> {len(chunks)} fragmentos "
      f"(longitud mín/prom/máx = {min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)} caracteres)")

start = time.time()
store = load_or_build_store(get_embedder(), args.chunk_size, args.overlap, rebuild=args.rebuild)
print(f"Índice listo: {store.count()} vectores en {time.time() - start:.1f} s")
