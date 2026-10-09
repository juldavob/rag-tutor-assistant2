"""
embeddings.py
-------------
Vectorización. Dos backends:

* "fastembed" (producción): modelo multilingüe
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ejecutado
  localmente con ONNX (sin torch, sin GPU, sin enviar el corpus a ningún
  servicio externo). 384 dimensiones, ~220 MB.
* "hash" (solo pruebas offline): vectorizador determinista por hashing de
  palabras. NO usar para evaluar calidad; solo sirve para probar el
  pipeline sin descargar modelos.

Por qué este modelo: el corpus está en español, el modelo es multilingüe,
es lo bastante liviano para el plan gratuito de Streamlit Community Cloud y
al correr localmente evita enviar el corpus completo a un tercero.
"""

import hashlib
import re
import unicodedata
from typing import List

import numpy as np

from config import EMBEDDING_CONFIG


class Embedder:
    name: str = "base"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = "minilm"
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._load().embed(texts)]


class HashEmbedder(Embedder):
    """Bolsa de palabras con hashing: determinista y sin descargas."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = "hash"

    @staticmethod
    def _tokens(text: str) -> List[str]:
        text = unicodedata.normalize("NFKD", text.lower())
        text = "".join(c for c in text if not unicodedata.combining(c))
        return re.findall(r"[a-z0-9]{2,}", text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)
            for token in self._tokens(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = np.linalg.norm(vector)
            vectors.append((vector / norm if norm else vector).tolist())
        return vectors


def get_embedder() -> Embedder:
    if EMBEDDING_CONFIG.backend == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(EMBEDDING_CONFIG.model_name)
