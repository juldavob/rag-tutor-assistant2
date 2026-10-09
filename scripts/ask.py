"""
Consulta el RAG desde la terminal (muestra fragmentos, respuesta y metadatos).

Uso:
    python scripts/ask.py "¿Cuándo uso ANOVA en vez de una prueba t?"
    python scripts/ask.py "..." --shot docs/screenshots/query_1.png
    python scripts/ask.py --chat --shot docs/screenshots/chat_cli.png
    python scripts/ask.py "..." --show-prompt
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from _shots import render_terminal_png
from config import MODEL_CONFIG, RETRIEVAL_CONFIG
from rag_pipeline import build_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="?")
parser.add_argument("--top-k", type=int, default=RETRIEVAL_CONFIG.top_k)
parser.add_argument("--chat", action="store_true", help="conversación interactiva con historial")
parser.add_argument("--show-prompt", action="store_true")
parser.add_argument("--shot", help="ruta PNG donde guardar una captura de la ejecución")
args = parser.parse_args()
if not args.question and not args.chat:
    parser.error("indica una pregunta o usa --chat")

pipeline = build_pipeline(top_k=args.top_k)
transcript = []
turns = []  # datos estructurados de cada turno (los usa docs/build_pdf_avance2.py)


def emit(text=""):
    print(text)
    transcript.append(text)


def run_turn(question, history):
    answer = pipeline.ask(question, history)
    emit(f"PREGUNTA: {question}")
    if answer.standalone_question != question:
        emit(f"(reformulada para la búsqueda: {answer.standalone_question})")
    emit(f"-- Fragmentos recuperados (top_k={args.top_k}) " + "-" * 20)
    for chunk in answer.retrieved:
        emit(f"[{chunk.rank}] similitud={chunk.score:.3f}  {chunk.source} > {chunk.section}")
        emit("    " + chunk.text.replace("\n", " ")[:150] + "...")
    if args.show_prompt:
        emit("-- Prompt enviado al modelo " + "-" * 20)
        emit(answer.prompt)
    emit("-- Respuesta del tutor " + "-" * 25)
    emit(answer.answer)
    emit("-- Metadatos " + "-" * 36)
    emit(json.dumps({k: v for k, v in answer.metadata.items() if k != "parse_error"}, ensure_ascii=False))
    status = "SIN respaldo en el material (abstención)" if answer.abstained else \
        "respaldada por: " + ", ".join(f"[{c.rank}] {c.source}" for c in answer.cited)
    emit(f"Estado: {status} | latencia {answer.latency_s} s | modelo {pipeline.llm.model_name}")
    turns.append({
        "question": question, "standalone_question": answer.standalone_question,
        "retrieved": [{"rank": c.rank, "source": c.source, "section": c.section, "score": c.score}
                      for c in answer.retrieved],
        "cited": [c.rank for c in answer.cited], "abstained": answer.abstained,
        "confidence": answer.metadata.get("nivel_confianza"), "topic": answer.metadata.get("tema"),
        "latency_s": answer.latency_s, "model": pipeline.llm.model_name,
        "answer_chars": len(answer.answer),
    })
    return answer


history = []
if args.chat:
    emit(f"$ python scripts/ask.py --chat   (escribe 'salir' para terminar)")
    while True:
        try:
            question = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"salir", "exit", "quit", ""}:
            break
        transcript.append(f"\nTú > {question}")
        answer = run_turn(question, history)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer.raw}]
else:
    emit(f'$ python scripts/ask.py "{args.question}"')
    run_turn(args.question, [])

if args.shot:
    render_terminal_png("\n".join(transcript), ROOT / args.shot, "ask.py — consulta al RAG")
    sidecar = (ROOT / args.shot).with_suffix(".json")
    sidecar.write_text(json.dumps({"turns": turns}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCaptura guardada en {args.shot} (datos en {sidecar.name})")
