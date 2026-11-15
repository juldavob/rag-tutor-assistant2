"""
Casos de comportamiento del agente (eval/behavior_cases.json: el comportamiento esperado se escribió ANTES de ejecutar).
Registra por caso: tools llamadas (con argumentos), llamadas al LLM, tokens y si cumplió lo esperado.

    python eval/run_behavior_cases.py [--no-web] [--rpm 14] [--only c03 c06]
Salida: eval/results/behavior_results.json y .csv
"""
import argparse
import csv
import difflib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent import build_agent  # noqa: E402
from config import AGENT_CONFIG, EVAL_DIR, KNOWLEDGE_BASE_DIR  # noqa: E402
from ingest import load_documents  # noqa: E402
from llm import LLMError  # noqa: E402
from tools import _allowed  # noqa: E402

RESULTS = EVAL_DIR / "results"


def longest_overlap(text, docs):
    best = 0
    for doc in docs:
        m = difflib.SequenceMatcher(None, text, doc, autojunk=False).find_longest_match(0, len(text), 0, len(doc))
        best = max(best, m.size)
    return best


def evaluate(case, ans, docs):
    r, names, text = case["reglas"], [t["name"] for t in ans.tool_calls], ans.answer
    norm = lambda s: s.lower().replace(",", ".")  # noqa: E731
    checks = {}
    if r.get("sin_tools"):
        checks["sin_tools"] = not names
    for t in r.get("debe_llamar", []):
        checks[f"llama {t}"] = t in names
    for t in r.get("no_debe_llamar", []):
        checks[f"no llama {t}"] = t not in names
    if "min_llamadas_base" in r:
        covers = bool(r.get("o_cubre_fuentes")) and all(any(d == s["documento"] for s in ans.base_sources) for d in r["o_cubre_fuentes"])
        checks["busca varias veces o cubre ambos temas"] = names.count("buscar_en_base") >= r["min_llamadas_base"] or covers
    if r.get("respuesta_contiene_alguno"):
        checks["resultado/frase esperada"] = any(norm(x) in norm(text) for x in r["respuesta_contiene_alguno"])
    if r.get("web_solo_permitidos"):
        checks["web solo en sitios permitidos"] = all(_allowed(s["url"]) for s in ans.web_sources)
    if r.get("debe_citar") == "base":
        checks["cita la base"] = "(Fuente:" in text
    if r.get("debe_citar") == "web":
        checks["cita la web [web: URL]"] = "[web:" in text
    if r.get("sin_volcado"):
        checks["no vuelca el corpus"] = longest_overlap(text, docs) < 350 and len(ans.base_sources) <= AGENT_CONFIG.kb_max_fragments_per_question
    if r.get("tipo_esperado"):
        checks["tipo de respuesta"] = ans.tipo == r["tipo_esperado"]
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--rpm", type=float, default=0, help="límite de llamadas por minuto (espacia los casos)")
    p.add_argument("--only", nargs="*")
    args = p.parse_args()
    cases = [c for c in json.loads((EVAL_DIR / "behavior_cases.json").read_text(encoding="utf-8"))
             if not args.only or c["id"] in args.only]
    agent = build_agent(web=False if args.no_web else None, log=True)
    web_available = "buscar_en_web" in agent.tools
    docs = [d.text for d in load_documents(KNOWLEDGE_BASE_DIR)]
    rows = []
    for case in cases:
        if "buscar_en_web" in case["reglas"].get("debe_llamar", []) and not web_available:
            rows.append({**{k: case[k] for k in ("id", "tipo", "entrada", "esperado")}, "tools": "", "llamadas_llm": 0,
                         "tokens": 0, "cumple": "OMITIDO", "fallos": "búsqueda web no disponible (falta TAVILY_API_KEY)", "respuesta": ""})
            print(f"{case['id']}: OMITIDO (sin búsqueda web)")
            continue
        try:
            ans = agent.ask(case["entrada"], [])
        except LLMError as error:
            print(f"{case['id']}: ERROR {error}")
            rows.append({**{k: case[k] for k in ("id", "tipo", "entrada", "esperado")}, "tools": "", "llamadas_llm": 0,
                         "tokens": 0, "cumple": "ERROR", "fallos": str(error)[:120], "respuesta": ""})
            continue
        checks = evaluate(case, ans, docs)
        ok = all(checks.values())
        rows.append({"id": case["id"], "tipo": case["tipo"], "entrada": case["entrada"], "esperado": case["esperado"],
                     "tools": "; ".join(f"{t['name']}({json.dumps(t['args'], ensure_ascii=False)})" for t in ans.tool_calls) or "ninguna",
                     "llamadas_llm": ans.llm_calls, "tokens": ans.prompt_tokens + ans.output_tokens,
                     "cumple": "SÍ" if ok else "NO", "fallos": ", ".join(k for k, v in checks.items() if not v),
                     "respuesta": ans.answer[:400]})
        print(f"{case['id']} [{case['tipo']}] -> {'CUMPLE' if ok else 'NO CUMPLE'} | tools: {rows[-1]['tools'][:90]} | LLM: {ans.llm_calls}")
        if args.rpm:
            time.sleep(60.0 / args.rpm * max(ans.llm_calls, 1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "behavior_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "behavior_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    total = [r for r in rows if r["cumple"] in ("SÍ", "NO")]
    print(f"\nCumplen {sum(r['cumple'] == 'SÍ' for r in total)}/{len(total)} casos. Resultados en eval/results/behavior_results.*")


if __name__ == "__main__":
    main()
