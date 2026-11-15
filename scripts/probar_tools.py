"""
Prueba las tools SIN el LLM (con .invoke) y muestra el resultado.

    python scripts/probar_tools.py
    python scripts/probar_tools.py --shot docs/screenshots/tools_invoke.png
"""
import _bootstrap  # noqa: F401
import argparse
import json

from _bootstrap import ROOT
from _shots import render_terminal_png
from config import AGENT_CONFIG, RETRIEVAL_CONFIG
from embeddings import get_embedder
from retriever import Retriever
from tools import Session, build_tools
from vector_store import load_or_build_store

parser = argparse.ArgumentParser()
parser.add_argument("--shot")
args = parser.parse_args()

embedder = get_embedder()
store = load_or_build_store(embedder, RETRIEVAL_CONFIG.chunk_size, RETRIEVAL_CONFIG.chunk_overlap)
tools = build_tools(Retriever(store, embedder, AGENT_CONFIG.kb_max_fragments), web=True)
session = Session()
lines = ["$ python scripts/probar_tools.py",
         f"Límites en código: {AGENT_CONFIG.kb_max_fragments} fragmentos/llamada, {AGENT_CONFIG.kb_max_chars} car./fragmento, "
         f"{AGENT_CONFIG.web_max_results} resultados web, sitios: {', '.join(AGENT_CONFIG.web_allowed_domains)}", ""]
tests = [("buscar_en_base", {"consulta": "error tipo I y tipo II"}),
         ("calcular_tamano_muestra", {"sigma": 10, "error_maximo": 2, "nivel_confianza": 0.95}),
         ("calcular_intervalo_confianza_media", {"media_muestral": 50.2, "desviacion_muestral": 3.1, "n": 25}),
         ("calcular_prueba_t_una_muestra", {"media_muestral": 105, "desviacion_muestral": 8, "n": 16, "mu0": 100}),
         ("buscar_en_web", {"consulta": "scipy.stats ttest_ind"})]
for name, call_args in tests:
    result = tools[name].invoke(call_args, session)
    if name == "buscar_en_base":
        for item in result.get("resultados", []):
            item["fragmento"] = item["fragmento"][:90] + "…"
    lines.append(f"{name}.invoke({json.dumps(call_args, ensure_ascii=False)})")
    lines.append(json.dumps(result, ensure_ascii=False, indent=1))
    lines.append("")
print("\n".join(lines))
if args.shot:
    render_terminal_png("\n".join(lines), ROOT / args.shot, "probar_tools.py — tools sin LLM")
