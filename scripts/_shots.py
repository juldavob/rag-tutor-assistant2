"""Renderiza texto de consola como una imagen tipo terminal (para el PDF)."""
import textwrap
from pathlib import Path
from typing import List


def wrap_lines(text: str, width: int = 98) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            lines.append("")
            continue
        indent = len(raw) - len(raw.lstrip())
        lines.extend(textwrap.wrap(raw, width=width, subsequent_indent=" " * (indent + 2),
                                   replace_whitespace=False) or [""])
    return lines


def render_terminal_png(text: str, path: Path, title: str = "terminal") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = wrap_lines(text)
    line_h, top_bar, pad = 0.26, 0.45, 0.2
    fig_w = 11.5
    fig_h = top_bar + pad + line_h * len(lines)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")
    fig.patch.set_facecolor("#1e1e1e")
    ax.add_patch(plt.Rectangle((0, fig_h - top_bar), fig_w, top_bar, facecolor="#333333"))
    for cx, color in zip([0.35, 0.65, 0.95], ["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(plt.Circle((cx, fig_h - top_bar / 2), 0.11, facecolor=color))
    ax.text(fig_w / 2, fig_h - top_bar / 2, title, ha="center", va="center",
            color="#cccccc", fontsize=10, fontfamily="monospace")
    y = fig_h - top_bar - line_h * 0.6
    for line in lines:
        color = "#e6e6e6"
        if line.startswith("$"):
            color = "#7fdc7f"
        elif line.startswith("==") or line.startswith("--"):
            color = "#6fa8dc"
        ax.text(0.2, y, line, ha="left", va="top", color=color, fontsize=8.6, fontfamily="monospace")
        y -= line_h
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, facecolor="#1e1e1e")
    plt.close()
