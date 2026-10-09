"""
rag_pipeline.py
---------------
Orquesta el flujo RAG completo para un turno de conversación:

  pregunta (+ historial)
     -> [reformulación si hay historial]  (LLM, pregunta autónoma)
     -> recuperación top_k fragmentos     (Chroma, similitud coseno)
     -> ensamblado del prompt             (system + few-shot + delimitadores)
     -> generación                        (LLM)
     -> parseo (Markdown + JSON) y citas

`build_pipeline()` arma todo (índice incluido) a partir de config.py.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import EMBEDDING_CONFIG, MODEL_CONFIG, RETRIEVAL_CONFIG
from embeddings import Embedder, get_embedder
from llm import BaseLLM, LLMError, get_llm
from retriever import Retriever
from system_prompt import SYSTEM_PROMPT, build_condense_prompt, build_user_prompt, parse_response
from vector_store import RetrievedChunk, load_or_build_store


@dataclass
class RAGAnswer:
    question: str
    standalone_question: str
    answer: str                      # explicación en Markdown (sin el bloque JSON)
    metadata: Dict                   # tema, fuentes, nivel_confianza
    retrieved: List[RetrievedChunk]  # todos los fragmentos recuperados (rank 1..k)
    raw: str                         # salida cruda del modelo
    latency_s: float = 0.0
    prompt: str = ""

    @property
    def cited(self) -> List[RetrievedChunk]:
        """Fragmentos que el modelo declaró haber usado (ids 1..k -> rank)."""
        ids = set(self.metadata.get("fuentes", []))
        return [c for c in self.retrieved if c.rank in ids]

    @property
    def abstained(self) -> bool:
        """True si el modelo declaró que el material no respalda la respuesta."""
        return len(self.metadata.get("fuentes", [])) == 0


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: BaseLLM):
        self.retriever = retriever
        self.llm = llm

    def standalone_question(self, question: str, history: Optional[List[Dict]]) -> str:
        if not history:
            return question
        system, user = build_condense_prompt(history, question)
        try:
            rewritten = self.llm.generate(system, user).strip().strip('"')
            return rewritten or question
        except LLMError:
            return question  # si falla la reformulación, se busca con la pregunta original

    def ask(self, question: str, history: Optional[List[Dict]] = None, top_k: Optional[int] = None) -> RAGAnswer:
        start = time.time()
        standalone = self.standalone_question(question, history)
        retrieved = self.retriever.retrieve(standalone, top_k)
        user_prompt = build_user_prompt(retrieved, question, history)
        raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        answer, metadata = parse_response(raw)
        return RAGAnswer(
            question=question,
            standalone_question=standalone,
            answer=answer,
            metadata=metadata,
            retrieved=retrieved,
            raw=raw,
            latency_s=round(time.time() - start, 2),
            prompt=user_prompt,
        )


def build_pipeline(chunk_size: Optional[int] = None, overlap: Optional[int] = None,
                   top_k: Optional[int] = None, llm: Optional[BaseLLM] = None,
                   embedder: Optional[Embedder] = None, rebuild: bool = False) -> RAGPipeline:
    chunk_size = chunk_size or RETRIEVAL_CONFIG.chunk_size
    overlap = RETRIEVAL_CONFIG.chunk_overlap if overlap is None else overlap
    top_k = top_k or RETRIEVAL_CONFIG.top_k
    embedder = embedder or get_embedder()
    store = load_or_build_store(embedder, chunk_size, overlap, rebuild=rebuild)
    return RAGPipeline(Retriever(store, embedder, top_k), llm or get_llm())
