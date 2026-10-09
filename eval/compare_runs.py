"""
compare_runs.py
---------------
Compara dos corridas de Ragas (antes/después de la iteración de mejora) y
genera la tabla y los gráficos que van en el PDF.

Uso:
    python eval/compare_runs.py --before baseline --after mejora

Salidas en eval/results/: comparison.csv, comparison.png, ragas_<tag>.png
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load(tag):
    return json.loads((RESULTS / f"{tag}_summary.json").read_text(encoding="utf-8"))


def metric_value(summary, scope, metric):
    block = summary.get(scope, {}).get(metric)
    return None if not block else block["mean"]


def bar_chart(series, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(series.keys())
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.8 / len(labels)
    colors = ["#9AA5B1", "#2C5F8A", "#5E35B1"]
    for i, label in enumerate(labels):
        values = [series[label].get(m) or 0 for m in METRICS]
        positions = [x + i * width for x in range(len(METRICS))]
        bars = ax.bar(positions, values, width, label=label, color=colors[i % len(colors)])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks([x + width * (len(labels) - 1) / 2 for x in range(len(METRICS))])
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("puntaje (0–1)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)
    rows = []
    for scope, scope_label in [("metrics_all", "todas las preguntas"), ("metrics_in_corpus", "solo preguntas en el corpus")]:
        for metric in METRICS:
            b, a = metric_value(before, scope, metric), metric_value(after, scope, metric)
            rows.append({"alcance": scope_label, "metrica": metric, args.before: b, args.after: a,
                         "cambio": None if b is None or a is None else round(a - b, 4)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "comparison.csv", index=False, encoding="utf-8")
    print(table.to_string(index=False))

    for tag, summary in [(args.before, before), (args.after, after)]:
        bar_chart({tag: {m: metric_value(summary, "metrics_all", m) for m in METRICS}},
                  f"Ragas — corrida '{tag}' (todas las preguntas)", RESULTS / f"ragas_{tag}.png")
    bar_chart({t: {m: metric_value(s, "metrics_all", m) for m in METRICS}
               for t, s in [(args.before, before), (args.after, after)]},
              "Ragas — antes vs. después de la iteración de mejora", RESULTS / "comparison.png")

    for tag, summary in [(args.before, before), (args.after, after)]:
        d = summary["diagnostics"]
        print(f"\n[{tag}] chunk={summary['config']['chunk_size']} overlap={summary['config']['overlap']} "
              f"top_k={summary['config']['top_k']} | acierto de recuperación={d['retrieval_hit_rate']} | "
              f"abstención correcta (fuera de corpus)={d['abstention_rate_out_of_corpus']} | "
              f"abstención errónea (en corpus)={d['false_abstention_rate_in_corpus']}")
    print("\nGuardado: eval/results/comparison.csv y comparison.png")


if __name__ == "__main__":
    main()
