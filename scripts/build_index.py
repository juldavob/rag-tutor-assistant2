"""
Construye el índice vectorial del corpus.

Desarrollo / evaluación (índice en chroma_db/, no se versiona):
    python scripts/build_index.py [--chunk-size N --overlap N --rebuild]

DESPLIEGUE (recomendado con libros grandes): calcula el índice UNA vez en tu PC y súbelo al repo; la nube lo
carga sin leer ni procesar los libros (si knowledge_base/ no tiene documentos usa index_prebuilt/ automáticamente):
    python scripts/build_index.py --prebuilt
"""
import _bootstrap  # noqa: F401
import argparse
import time

from chunking import chunk_corpus
from config import KNOWLEDGE_BASE_DIR, RETRIEVAL_CONFIG
from embeddings import get_embedder
from ingest import load_documents
from vector_store import PREBUILT_DIR, build_prebuilt_store, load_or_build_store

parser = argparse.ArgumentParser()
parser.add_argument("--chunk-size", type=int, default=RETRIEVAL_CONFIG.chunk_size)
parser.add_argument("--overlap", type=int, default=RETRIEVAL_CONFIG.chunk_overlap)
parser.add_argument("--rebuild", action="store_true", help="borra y reconstruye el índice local")
parser.add_argument("--prebuilt", action="store_true", help="genera index_prebuilt/ para versionarlo y desplegar")
args = parser.parse_args()

documents = load_documents(KNOWLEDGE_BASE_DIR)
chunks = chunk_corpus(documents, args.chunk_size, args.overlap)
lengths = [len(c.text) for c in chunks] or [0]
print(f"Corpus: {len({d.source for d in documents})} archivos, {len(documents)} documentos/páginas")
print(f"Chunking: tamaño={args.chunk_size} solapamiento={args.overlap} -> {len(chunks)} fragmentos "
      f"(longitud mín/prom/máx = {min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)} caracteres)")

start = time.time()
if args.prebuilt:
    def progress(done, total):
        if done == total or (done // 64) % 10 == 0:
            print(f"  vectorizados {done}/{total} ({time.time() - start:.0f} s)", flush=True)
    store = build_prebuilt_store(get_embedder(), args.chunk_size, args.overlap, progress=progress)
    big = [(p, p.stat().st_size / 1e6) for p in PREBUILT_DIR.rglob("*") if p.is_file()]
    total = sum(m for _, m in big)
    print(f"Índice precalculado: {store.count()} vectores en {PREBUILT_DIR} ({total:.0f} MB, {time.time() - start:.0f} s)")
    for path, mb in big:
        if mb > 90:
            print(f"ATENCIÓN: {path.name} pesa {mb:.0f} MB y GitHub rechaza archivos de más de 100 MB; sube chunk_size o recorta el corpus.")
    print("Siguiente paso: git add index_prebuilt && git commit && git push")
else:
    store = load_or_build_store(get_embedder(), args.chunk_size, args.overlap, rebuild=args.rebuild)
    print(f"Índice listo: {store.count()} vectores en {time.time() - start:.1f} s")
