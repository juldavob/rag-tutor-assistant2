"""
Consulta al AGENTE desde la terminal: muestra las tools usadas (con argumentos), la respuesta y las fuentes por origen.

    python scripts/ask_agent.py "Hola"
    python scripts/ask_agent.py "¿Cuándo uso ANOVA en vez de t?" --shot docs/screenshots/agente_base.png
    python scripts/ask_agent.py --chat          # conversación con historial
    python scripts/ask_agent.py "..." --no-web  # sin búsqueda web
"""
import _bootstrap  # noqa: F401
import argparse
import json

from _bootstrap import ROOT
from _shots import render_terminal_png
from agent import build_agent

parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="?")
parser.add_argument("--chat", action="store_true")
parser.add_argument("--no-web", action="store_true")
parser.add_argument("--shot", help="ruta PNG para guardar una captura (y un .json con los datos)")
args = parser.parse_args()
if not args.question and not args.chat:
    parser.error("indica una pregunta o usa --chat")

agent = build_agent(web=False if args.no_web else None)
transcript, turns = [], []


def emit(text=""):
    print(text)
    transcript.append(text)


def run_turn(question, history):
    a = agent.ask(question, history)
    emit(f"PREGUNTA: {question}")
    emit(f"-- Herramientas usadas ({len(a.tool_calls)}) " + "-" * 18)
    if not a.tool_calls:
        emit("(ninguna: respondió sin usar tools)")
    for i, t in enumerate(a.tool_calls, 1):
        emit(f"{i}. {t['name']}({json.dumps(t['args'], ensure_ascii=False)})")
        emit(f"   -> {t['summary']}")
    emit("-- Respuesta " + "-" * 34)
    emit(a.answer)
    if a.base_sources or a.web_sources:
        emit("-- Fuentes por origen " + "-" * 26)
        for s in a.base_sources:
            emit(f"[{s['ref']}] BASE {'(citada) ' if s['cited'] else ''}{s['documento']} > {s['seccion']}  sim={s['similitud']}")
        for s in a.web_sources:
            emit(f"[{s['ref']}] WEB  {'(citada) ' if s['cited'] else ''}{s['titulo']} - {s['url']}")
    emit(f"Tipo: {a.tipo or '?'} | confianza: {a.metadata.get('nivel_confianza')} | llamadas al LLM: {a.llm_calls} | "
         f"tokens: {a.prompt_tokens + a.output_tokens} | {a.latency_s} s | modelo {agent.llm.model_name}")
    turns.append(a.to_dict())
    return a


history = []
if args.chat:
    emit("$ python scripts/ask_agent.py --chat   (escribe 'salir' para terminar)")
    while True:
        try:
            q = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"salir", "exit", "quit", ""}:
            break
        transcript.append(f"\nTú > {q}")
        a = run_turn(q, history)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a.answer}]
else:
    emit(f'$ python scripts/ask_agent.py "{args.question}"')
    run_turn(args.question, [])

if args.shot:
    path = ROOT / args.shot
    render_terminal_png("\n".join(transcript), path, "ask_agent.py — agente con tools")
    path.with_suffix(".json").write_text(json.dumps({"turns": turns}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCaptura guardada en {args.shot}")
