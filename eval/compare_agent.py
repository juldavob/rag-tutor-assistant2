"""
Compara el pipeline fijo (Avance 2) con el agente (Avance 3): Ragas y costo.

    python eval/compare_agent.py --a2 mejora --a2-cost pipeline --agent agente
  --a2        corrida de Ragas del Avance 2 a comparar (p. ej. la configuración final: mejora)
  --a2-cost   corrida del pipeline con costo medido (python eval/run_ragas.py --tag pipeline --no-ragas ...);
              si se omite se usa --a2 cuando tenga costo
Salidas en eval/results/: agent_ragas.csv, agent_cost.csv, agent_ragas.png
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load(tag):
    return json.loads((RESULTS / f"{tag}_summary.json").read_text(encoding="utf-8"))


def val(summary, scope, metric):
    block = summary.get(scope, {}).get(metric)
    return block["mean"] if block else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a2", default="mejora")
    p.add_argument("--a2-cost", default=None)
    p.add_argument("--agent", default="agente")
    args = p.parse_args()
    a2, ag = load(args.a2), load(args.agent)
    for tag, s in ((args.a2, a2), (args.agent, ag)):
        if not s.get("complete", True):
            print(f"AVISO: la corrida '{tag}' está incompleta ({s.get('n_completed')}/{s.get('n_questions')}).")

    rows = []
    for scope, label in (("metrics_all", "todas las preguntas"), ("metrics_in_corpus", "solo en corpus")):
        for m in METRICS:
            b, a = val(a2, scope, m), val(ag, scope, m)
            rows.append({"alcance": label, "metrica": m, "avance2_pipeline": b, "agente": a,
                         "cambio": None if b is None or a is None else round(a - b, 4)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "agent_ragas.csv", index=False, encoding="utf-8")
    print(table.to_string(index=False))
    d2, dg = a2["diagnostics"], ag["diagnostics"]
    print(f"\nAcierto de recuperación: A2={d2['retrieval_hit_rate']} agente={dg['retrieval_hit_rate']} | "
          f"abstención correcta: A2={d2['abstention_rate_out_of_corpus']} agente={dg['abstention_rate_out_of_corpus']}")

    cost_tag = args.a2_cost or args.a2
    pc, ac = load(cost_tag).get("cost"), ag.get("cost")
    if pc and ac:
        cost = pd.DataFrame([{"indicador": k, "pipeline_fijo": pc[f], "agente": ac[f],
                              "razon_agente_sobre_pipeline": round(ac[f] / pc[f], 2) if pc[f] else None}
                             for k, f in (("llamadas al LLM (prom.)", "avg_llm_calls"), ("llamadas a tools (prom.)", "avg_tool_calls"),
                                          ("tokens de entrada (prom.)", "avg_prompt_tokens"), ("tokens de salida (prom.)", "avg_output_tokens"),
                                          ("tiempo de respuesta, s (prom.)", "avg_latency_s"))])
        cost.to_csv(RESULTS / "agent_cost.csv", index=False, encoding="utf-8")
        print("\n" + cost.to_string(index=False))
    else:
        print("\nFalta el costo: ejecuta el pipeline con --no-ragas y otro --tag (ver docstring) o re-ejecuta el agente.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, (label, s, color) in enumerate(((f"Avance 2 ({args.a2})", a2, "#9AA5B1"), (f"Agente ({args.agent})", ag, "#5E35B1"))):
        values = [val(s, "metrics_all", m) or 0 for m in METRICS]
        bars = ax.bar([x + i * 0.38 for x in range(4)], values, 0.38, label=label, color=color)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks([x + 0.19 for x in range(4)])
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.12)
    ax.set_title("Ragas: pipeline fijo (Avance 2) vs. agente (Avance 3)", fontsize=11)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(RESULTS / "agent_ragas.png", dpi=160)


if __name__ == "__main__":
    main()
