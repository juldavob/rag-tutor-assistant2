"""
chunking.py
-----------
Estrategia de fragmentación (chunking) en dos niveles:

1. Corte por estructura: el texto se divide primero por encabezados
   Markdown (#, ##, ###). Así un fragmento nunca mezcla dos secciones
   distintas y cada fragmento sabe a qué sección pertenece.
2. Corte recursivo por tamaño: dentro de cada sección, si el texto supera
   `chunk_size` caracteres se corta en este orden de preferencia:
   párrafo -> línea -> oración -> palabra. Se prioriza no partir ideas.

Solapamiento: cada fragmento (salvo el primero de una sección) arranca con
los últimos `chunk_overlap` caracteres del fragmento anterior, alineados a
límite de palabra, para que una idea que cae en la frontera no se pierda.
El tamaño máximo real de un fragmento es chunk_size + chunk_overlap.

Cada fragmento guarda metadatos (fuente, título, sección, página, índice)
que permiten citar de dónde salió cada respuesta.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ingest import Document

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    chunk_id: str
    text: str                # texto literal del documento
    embed_text: str          # texto que se vectoriza (título + sección + texto)
    source: str
    title: str
    section: str
    page: Optional[int]
    index: int


def split_by_headings(text: str) -> List[Tuple[str, str]]:
    """Devuelve [(titulo_de_seccion, cuerpo)] según encabezados Markdown."""
    sections: List[Tuple[str, str]] = []
    current_title, buffer = "", []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            body = "\n".join(buffer).strip()
            if body:
                sections.append((current_title, body))
            current_title, buffer = match.group(2).strip(), []
        else:
            buffer.append(line)
    body = "\n".join(buffer).strip()
    if body:
        sections.append((current_title, body))
    return sections


def _split_recursive(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Parte `text` en piezas de a lo sumo `chunk_size` caracteres."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    if not separators:  # último recurso: corte duro
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, rest = separators[0], separators[1:]
    if separator not in text:
        return _split_recursive(text, chunk_size, rest)

    pieces = [p for p in text.split(separator) if p.strip()]
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + separator + piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(piece) > chunk_size:
            chunks.extend(_split_recursive(piece, chunk_size, rest))
            current = ""
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    """Últimos `overlap` caracteres de `text`, empezando en límite de palabra."""
    if overlap <= 0:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_document(document: Document, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")

    chunks: List[Chunk] = []
    counter = 0
    for section_title, body in split_by_headings(document.text):
        pieces = _split_recursive(body, chunk_size, SEPARATORS)
        for position, piece in enumerate(pieces):
            text = piece
            if position > 0:
                tail = _overlap_tail(pieces[position - 1], chunk_overlap)
                if tail:
                    text = tail + " " + piece
            header = document.title + (f" — {section_title}" if section_title else "")
            page_suffix = f"_p{document.page}" if document.page else ""
            chunks.append(Chunk(
                chunk_id=f"{document.source}{page_suffix}_{counter}",
                text=text.strip(),
                embed_text=f"{header}\n{text.strip()}",
                source=document.source,
                title=document.title,
                section=section_title,
                page=document.page,
                index=counter,
            ))
            counter += 1
    return chunks


def chunk_corpus(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    return all_chunks
