"""
build_pdf_avance2.py
--------------------
Genera docs/avance2.pdf a partir de datos REALES del proyecto:

  * parámetros y estadísticas del corpus/chunking (calculados en el momento)
  * resultados de Ragas (eval/results/<tag>_summary.json) y comparación
  * capturas de consultas (docs/screenshots/*.png creadas con scripts/ask.py,
    con sus datos en el .json hermano) y capturas de la app desplegada
    (que se colocan manualmente en docs/screenshots/)

Lo que falte se marca en el PDF como PENDIENTE y se lista al terminar.

Uso:
    python docs/build_pdf_avance2.py --author "Tu nombre" \
        --app-url https://tu-app.streamlit.app --repo-url https://github.com/usuario/repo
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.platypus import (HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

from chunking import chunk_corpus  # noqa: E402
from config import (EMBEDDING_CONFIG, KNOWLEDGE_BASE_DIR, MODEL_CONFIG, RETRIEVAL_CONFIG,  # noqa: E402
                    SUBJECT_CONFIG)
from ingest import load_documents  # noqa: E402

DOCS = ROOT / "docs"
SHOTS = DOCS / "screenshots"
RESULTS = ROOT / "eval" / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
PENDING = []

BLUE, PURPLE, GREY = colors.HexColor("#2C5F8A"), colors.HexColor("#5E35B1"), colors.HexColor("#666666")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("Cover", fontSize=24, leading=29, alignment=TA_CENTER, textColor=BLUE, fontName="Helvetica-Bold", spaceAfter=10))
styles.add(ParagraphStyle("CoverSub", fontSize=13, leading=18, alignment=TA_CENTER, textColor=GREY, spaceAfter=6))
styles.add(ParagraphStyle("H1", fontSize=16, leading=20, textColor=BLUE, spaceBefore=14, spaceAfter=8, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("H2", fontSize=12.5, leading=16, textColor=PURPLE, spaceBefore=10, spaceAfter=6, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("Bodyx", fontSize=10.2, leading=14.5, spaceAfter=6))
styles.add(ParagraphStyle("Cell", fontSize=8.8, leading=11.5))
styles.add(ParagraphStyle("CellB", fontSize=8.8, leading=11.5, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("Cap", fontSize=9, leading=12, alignment=TA_CENTER, textColor=GREY, spaceAfter=12, fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle("Pending", fontSize=9.5, leading=13, textColor=colors.HexColor("#8A5A00")))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def pending_box(message):
    PENDING.append(message)
    table = Table([[Paragraph(f"<b>PENDIENTE</b> - {message}", styles["Pending"])]], colWidths=[6.6 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E1")),
                               ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E0A800")),
                               ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    return table


def image_flowable(path, max_w=6.5 * inch, max_h=7.4 * inch):
    width, height = ImageReader(str(path)).getSize()
    scale = min(max_w / width, max_h / height)
    return Image(str(path), width=width * scale, height=height * scale)


def figure(path, caption, missing_message):
    path = Path(path)
    if path.exists():
        return [image_flowable(path), P(caption, "Cap")]
    return [pending_box(missing_message), Spacer(1, 10)]


def grid(rows, widths, header=True):
    data = [[Paragraph(str(c), styles["CellB" if (header and r == 0) else "Cell"]) for c in row] for r, row in enumerate(rows)]
    table = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")))
    table.setStyle(TableStyle(style))
    return table


# ------------------------------------------------------------------ datos
def load_summary(tag):
    path = RESULTS / f"{tag}_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def metric(summary, scope, name):
    block = (summary or {}).get(scope, {}).get(name)
    return block["mean"] if block else None


def corpus_stats(chunk_size, overlap):
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = chunk_corpus(documents, chunk_size, overlap)
    lengths = [len(c.text) for c in chunks]
    words = sum(len(d.text.split()) for d in documents)
    return {"files": len({d.source for d in documents}), "words": words, "chunks": len(chunks),
            "avg": sum(lengths) // max(len(lengths), 1), "max": max(lengths, default=0)}


# ------------------------------------------------------------------ diagrama
def make_flow_diagram(path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.2, 5.9))
    ax.set_xlim(0, 14.6); ax.set_ylim(0, 8.4); ax.axis("off")

    def box(x, y, w, title, detail, face, edge):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.8), w, 1.6, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=face, ec=edge, lw=1.5))
        ax.text(x, y + 0.32, title, ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1b1b1b")
        ax.text(x, y - 0.28, detail, ha="center", va="center", fontsize=8.7, color="#333")

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color="#555", lw=1.3))

    ax.text(0.2, 7.95, "INDEXACIÓN  (una sola vez, o cuando cambia el corpus)", fontsize=10.5, color="#2C5F8A", fontweight="bold")
    top = [("1. Corpus", f"{cfg['files']} documentos\n(.md .pdf .docx .html)"),
           ("2. Ingesta", "extrae texto;\nPDF por página"),
           ("3. Chunking", f"por encabezados +\nrecursivo\n{cfg['chunk']} car. / solap. {cfg['overlap']}"),
           ("4. Embeddings", "MiniLM multilingüe\n(local, ONNX, 384 dim)"),
           ("5. Chroma", f"índice persistente\ncoseno + metadatos\n{cfg['chunks']} fragmentos")]
    xs_top = [1.5, 4.4, 7.3, 10.2, 13.1]
    for x, (title, detail) in zip(xs_top, top):
        box(x, 6.6, 2.5, title, detail, "#EAF2FB", "#2C5F8A")
    for a, b in zip(xs_top, xs_top[1:]):
        arrow((a + 1.27, 6.6), (b - 1.27, 6.6))

    ax.text(0.2, 4.15, "CONSULTA  (cada pregunta del estudiante)", fontsize=10.5, color="#5E35B1", fontweight="bold")
    bottom = [("Pregunta +\nhistorial", "chat Streamlit", "#FDF1E3", "#B0651E"),
              ("6. Reformulación", "pregunta autónoma\n(solo si hay historial)", "#EDE7F6", "#5E35B1"),
              ("7. Recuperación", f"similitud coseno\ntop_k = {cfg['top_k']}", "#EAF2FB", "#2C5F8A"),
              ("8. Prompt", "system + few-shot +\n<contexto> <historial>\n<pregunta>", "#EDE7F6", "#5E35B1"),
              ("9. Generación", f"Gemini\ntemp. {cfg['temp']}", "#E8F5E9", "#2E7D32"),
              ("10. Respuesta", "Markdown + JSON\n+ fuentes citadas", "#FFF3E0", "#B0651E")]
    xs_bot = [1.15, 3.55, 5.95, 8.35, 10.75, 13.15]
    for x, (title, detail, face, edge) in zip(xs_bot, bottom):
        box(x, 2.3, 2.1, title, detail, face, edge)
    for a, b in zip(xs_bot, xs_bot[1:]):
        arrow((a + 1.05, 2.3), (b - 1.05, 2.3))
    # Chroma -> Recuperación (conector en codo)
    ax.plot([13.1, 13.1, 5.95], [5.8, 4.5, 4.5], color="#555", lw=1.3)
    arrow((5.95, 4.5), (5.95, 3.12))
    ax.text(9.9, 4.62, "búsqueda en el índice", fontsize=8, color="#555", ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------------------------------------------------ análisis automático
COMPONENT = {
    "faithfulness": ("generación (prompt y modelo)",
                     "la respuesta contiene afirmaciones que los fragmentos recuperados no sostienen, es decir, el modelo agrega o extrapola información",
                     "reforzar en el prompt la regla de usar solo el contexto, exigir una cita por afirmación, mantener la temperatura baja o probar un modelo más fiel"),
    "answer_relevancy": ("generación (prompt)",
                         "las respuestas incluyen información accesoria o no abordan de forma directa lo preguntado; además, una abstención correcta en preguntas fuera del corpus no 'responde' la pregunta y reduce esta métrica",
                         "pedir respuestas más directas en el prompt, ajustar los ejemplos few-shot y analizar por separado las preguntas dentro y fuera del corpus"),
    "context_precision": ("recuperación",
                          "entre los fragmentos recuperados hay varios que no son relevantes, o los relevantes no aparecen en las primeras posiciones (top_k alto o fragmentos poco específicos)",
                          "reducir top_k, usar fragmentos más específicos o añadir un paso de re-ranking"),
    "context_recall": ("chunking y recuperación",
                       "los fragmentos recuperados no contienen toda la información necesaria para construir la respuesta de referencia (fragmentos muy pequeños que parten una idea, o top_k insuficiente)",
                       "aumentar el tamaño y el solapamiento de los fragmentos, o aumentar top_k"),
}


def lowest_metric(summary, scope):
    values = {m: metric(summary, scope, m) for m in METRICS}
    values = {m: v for m, v in values.items() if v is not None}
    return min(values, key=values.get) if values else None


def comparison_paragraphs(before, after, tag_b, tag_a):
    out = []
    cb, ca = before["config"], after["config"]
    out.append(P(f"<b>Cambio aplicado en la iteración.</b> Se modificaron los parámetros de recuperación: tamaño de fragmento "
                 f"{cb['chunk_size']} a {ca['chunk_size']} caracteres, solapamiento {cb['overlap']} a {ca['overlap']} y top_k "
                 f"{cb['top_k']} a {ca['top_k']}. El resto del pipeline (corpus, modelo de embeddings, prompt, modelo de generación y juez) se mantuvo igual."))
    lines = []
    for m in METRICS:
        b, a = metric(before, "metrics_all", m), metric(after, "metrics_all", m)
        if b is None or a is None:
            lines.append(f"<b>{m}</b>: no se pudo calcular en alguna de las corridas.")
            continue
        delta = a - b
        verb = "se mantuvo estable" if abs(delta) < 0.03 else ("mejoró" if delta > 0 else "empeoró")
        lines.append(f"<b>{m}</b> {verb}: {b:.3f} a {a:.3f} ({delta:+.3f}).")
    out.append(P("<b>Efecto por métrica (todas las preguntas).</b> " + " ".join(lines)))
    means = {t: [metric(s, "metrics_all", m) for m in METRICS] for t, s in [(tag_b, before), (tag_a, after)]}
    if all(None not in v for v in means.values()):
        avg = {t: sum(v) / len(v) for t, v in means.items()}
        best = max(avg, key=avg.get)
        out.append(P(f"<b>Balance.</b> Promediando las cuatro métricas, la corrida '{best}' obtuvo el valor más alto "
                     f"({avg[tag_b]:.3f} para '{tag_b}' frente a {avg[tag_a]:.3f} para '{tag_a}'). "
                     f"Ese promedio es solo una referencia: conviene decidir según la métrica que más importa para un tutor, "
                     f"que es no inventar contenido (faithfulness) y recuperar todo lo necesario (context_recall)."))
    low = lowest_metric(after, "metrics_all")
    if low:
        comp, cause, fix = COMPONENT[low]
        out.append(P(f"<b>Métrica más baja tras la mejora: {low} ({metric(after, 'metrics_all', low):.3f}).</b> "
                     f"Se atribuye principalmente al componente de <b>{comp}</b>: {cause}. "
                     f"Para mejorarla se propone: {fix}."))
    low_b = lowest_metric(before, "metrics_all")
    if low_b and low_b != low:
        out.append(P(f"En la corrida inicial la métrica más baja era {low_b} ({metric(before, 'metrics_all', low_b):.3f}); "
                     f"la iteración desplazó el punto débil hacia {low}."))
    out.append(P("<b>Limitaciones de la evaluación.</b> El conjunto tiene pocas preguntas, por lo que diferencias pequeñas "
                 "(del orden de 0,05) pueden deberse al azar o a la variabilidad del modelo juez. Además, el juez de Ragas es un "
                 "modelo de lenguaje de la misma familia que el generador, lo que puede sesgar las puntuaciones a favor del sistema. "
                 "Las preguntas fuera del corpus no tienen contexto relevante por construcción, por lo que se reporta también el subconjunto "
                 "de preguntas dentro del corpus."))
    return out


def capture_reading(sidecar):
    """Texto factual generado a partir de los datos reales de una consulta."""
    if not sidecar.exists():
        return None
    turns = json.loads(sidecar.read_text(encoding="utf-8")).get("turns", [])
    parts = []
    for t in turns:
        sources = ", ".join(f"[{r['rank']}] {r['source']} (sim. {r['score']:.2f})" for r in t["retrieved"])
        if t["abstained"]:
            verdict = "El tutor <b>se abstuvo</b>: declaró que el material no respalda la respuesta y no citó ningún fragmento."
        else:
            verdict = (f"El tutor respondió citando los fragmentos {', '.join('[%d]' % c for c in t['cited'])} "
                       f"con confianza <b>{t['confidence']}</b>.")
        rewritten = (f" La pregunta se reformuló para la búsqueda como: <i>{t['standalone_question']}</i>."
                     if t["standalone_question"] != t["question"] else "")
        parts.append(f"<b>Consulta:</b> <i>{t['question']}</i>.{rewritten} Fragmentos recuperados: {sources}. {verdict} "
                     f"Latencia: {t['latency_s']} s (modelo {t['model']}).")
    return parts


def user_notes():
    notes, current = {}, None
    path = DOCS / "analisis_capturas.md"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                current = line[3:].strip()
                notes[current] = []
            elif current and line.strip():
                notes[current].append(line.strip())
    return {k: " ".join(v) for k, v in notes.items()}


# ------------------------------------------------------------------ documento
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default="[Nombre del estudiante]")
    parser.add_argument("--app-url", default="")
    parser.add_argument("--repo-url", default="")
    parser.add_argument("--before", default="baseline")
    parser.add_argument("--after", default="mejora")
    parser.add_argument("--out", default=str(DOCS / "avance2.pdf"))
    args = parser.parse_args()

    before, after = load_summary(args.before), load_summary(args.after)
    final = after["config"] if after else {"chunk_size": RETRIEVAL_CONFIG.chunk_size, "overlap": RETRIEVAL_CONFIG.chunk_overlap,
                                           "top_k": RETRIEVAL_CONFIG.top_k, "generation_model": MODEL_CONFIG.model_name,
                                           "judge_model": MODEL_CONFIG.model_name}
    stats = corpus_stats(final["chunk_size"], final["overlap"])
    stats_before = corpus_stats(before["config"]["chunk_size"], before["config"]["overlap"]) if before else None
    flow_cfg = {"files": stats["files"], "chunk": final["chunk_size"], "overlap": final["overlap"], "chunks": stats["chunks"],
                "top_k": final["top_k"], "temp": MODEL_CONFIG.temperature}
    diagram = DOCS / "images" / "flujo_rag.png"
    make_flow_diagram(diagram, flow_cfg)

    app_url = args.app_url or ""
    repo_url = args.repo_url or ""
    story = []

    # ---- portada
    story += [Spacer(1, 1.1 * inch), P("Tutor Académico Personalizado", "Cover"),
              P("Asistente Experto basado en RAG y Agentes", "CoverSub"), Spacer(1, 6),
              P("Avance 2 - Flujo RAG completo, evaluación con Ragas y despliegue conversacional", "CoverSub"),
              Spacer(1, 0.3 * inch), HRFlowable(width="60%", color=BLUE, thickness=1.2, hAlign="CENTER"), Spacer(1, 0.3 * inch),
              P(f"Asignatura piloto: {SUBJECT_CONFIG.name}", "CoverSub"), P(f"Autor: {args.author}", "CoverSub"), Spacer(1, 0.3 * inch)]
    if app_url:
        story.append(P(f"Aplicación desplegada: <link href='{app_url}' color='blue'>{app_url}</link>", "CoverSub"))
    else:
        story.append(pending_box("escribir la URL pública de la aplicación (opción --app-url al generar este PDF)."))
    story.append(Spacer(1, 6))
    if repo_url:
        story.append(P(f"Repositorio: <link href='{repo_url}' color='blue'>{repo_url}</link>", "CoverSub"))
    else:
        story.append(pending_box("escribir el enlace al repositorio de GitHub (opción --repo-url)."))
    story.append(PageBreak())

    # ---- 1. flujo
    story += [P("1. Flujo RAG implementado", "H1"),
              P("El sistema responde preguntas de un estudiante usando únicamente un material de estudio. Se divide en dos fases: una "
                "fase de <b>indexación</b>, que prepara el material una vez, y una fase de <b>consulta</b>, que se ejecuta con cada pregunta. "
                "Al modelo de lenguaje solo se le envían la pregunta, el historial reciente y los fragmentos recuperados; nunca el corpus completo."),
              image_flowable(diagram, max_h=4.4 * inch),
              P("Figura 1. Diagrama del flujo RAG implementado, con los parámetros efectivamente utilizados.", "Cap")]

    # ---- 2. parámetros
    rows = [["Etapa", "Decisión", "Justificación"],
            ["Selección de documentos",
             f"{stats['files']} documentos en Markdown (~{stats['words']} palabras) sobre {SUBJECT_CONFIG.name}: variables y escalas, "
             "estimación, pruebas de hipótesis, t/ANOVA, pruebas no paramétricas y chi-cuadrado, correlación/regresión y guía de selección de pruebas.",
             "Cubre los temas del curso en los que un estudiante necesita apoyo y permite preguntas que combinan varios documentos (por ejemplo, elegir una prueba según la escala y los supuestos). "
             "El formato con encabezados facilita citar la sección de origen."],
            ["Ingesta", "Lectores para TXT/MD, PDF (por página), DOCX y HTML. Los encabezados se normalizan a Markdown.",
             "Soportar los formatos pedidos y conservar la estructura (secciones y páginas) para poder citar la fuente."],
            ["Chunking",
             f"Corte por encabezados y luego recursivo (párrafo, línea, oración, palabra). Tamaño {final['chunk_size']} caracteres, solapamiento {final['overlap']} "
             f"({stats['chunks']} fragmentos; longitud media {stats['avg']} car.)"
             + (f". Configuración inicial: {before['config']['chunk_size']}/{before['config']['overlap']} ({stats_before['chunks']} fragmentos)." if before else "."),
             "Cada fragmento pertenece a una sola sección, de modo que no mezcla temas. El tamaño se eligió para que una definición o un procedimiento completo "
             "(2-4 oraciones) quepa en un fragmento; el solapamiento (~14%) evita perder ideas en la frontera. Los valores definitivos se decidieron con la evaluación Ragas (sección 4)."],
            ["Vectorización", f"{EMBEDDING_CONFIG.model_name.split('/')[-1]} (384 dimensiones) ejecutado localmente con fastembed/ONNX. Se vectoriza 'título - sección + texto'.",
             "Es multilingüe (el corpus está en español), liviano para desplegar en un plan gratuito y, al ser local, el corpus no se envía a terceros para generar embeddings. "
             "Limitación: fue entrenado con secuencias de ~128 tokens, por lo que fragmentos más largos pierden algo de precisión."],
            ["Base vectorial", "Chroma persistente, distancia coseno. Metadatos: archivo, título, sección, página e índice. Una colección por (modelo, tamaño, solapamiento, hash del corpus).",
             "Los metadatos permiten citar documento y fragmento en cada respuesta; el hash del corpus reconstruye el índice automáticamente si el material cambia."],
            ["Recuperación", f"Similitud coseno, top_k = {final['top_k']}. Las preguntas de seguimiento se reformulan como preguntas autónomas antes de buscar.",
             "La búsqueda vectorial no entiende el historial, por eso 'y si no se cumple?' se reescribe usando la conversación. top_k se ajustó en la evaluación."],
            ["Generación", f"Gemini ({final.get('generation_model', MODEL_CONFIG.model_name)}), temperatura {MODEL_CONFIG.temperature}. System prompt, ejemplos few-shot, delimitadores XML y formato Markdown + JSON del Avance 1, refinados.",
             "Temperatura baja para priorizar fidelidad. Refinamientos: fragmentos numerados con citas [n], historial delimitado, tercer ejemplo few-shot de seguimiento, abstención explícita y "
             "eliminación de etiquetas falsas dentro de los documentos (anti-inyección)."]]
    story += [P("2. Decisiones técnicas y justificación de parámetros", "H1"), grid(rows, [1.15, 2.85, 2.7]), Spacer(1, 8)]

    story += [P("Refinamiento del prompt respecto al Avance 1", "H2"),
              P("(1) Los fragmentos llegan numerados con su fuente y el modelo debe citarlos como [n] y devolver en el JSON la lista de ids usados (campo 'fuentes'), "
                "lo que permite a la interfaz mostrar qué fragmento respalda cada respuesta. (2) Se añadió un bloque &lt;historial&gt; delimitado. "
                "(3) Se añadió un paso de reformulación para preguntas de seguimiento. (4) Se añadió un tercer ejemplo few-shot que usa el historial. "
                "(5) El system prompt se envía como instrucción de sistema y los datos como mensaje de usuario. "
                "(6) Se eliminan de los textos recuperados las etiquetas que imiten los delimitadores, para que un documento no pueda cerrar el bloque de contexto e inyectar órdenes.")]
    story.append(PageBreak())

    # ---- 3. capturas
    story.append(P("3. Ejecución de consultas sobre el RAG", "H1"))
    story.append(P("Las siguientes capturas corresponden a la ejecución real de <font face='Courier'>scripts/ask.py</font>: muestran los fragmentos "
                   "recuperados con su similitud, la respuesta del tutor, los metadatos devueltos y la latencia."))
    notes = user_notes()
    captures = [("query_1", "Consulta dentro del corpus (comparación de pruebas)"),
                ("query_2", "Consulta dentro del corpus (cálculo o interpretación)"),
                ("query_fuera_corpus", "Consulta fuera del corpus (control de alucinaciones)")]
    for name, title in captures:
        story.append(P(title, "H2"))
        story += figure(SHOTS / f"{name}.png", f"Captura de {name}.png", f"crear docs/screenshots/{name}.png con: python scripts/ask.py \"tu pregunta\" --shot docs/screenshots/{name}.png")
        reading = capture_reading(SHOTS / f"{name}.json")
        for text in reading or []:
            story.append(P(f"<b>Lectura de la ejecución.</b> {text}"))
        if name in notes:
            story.append(P(f"<b>Análisis.</b> {notes[name]}"))
        elif (SHOTS / f"{name}.png").exists():
            PENDING.append(f"añadir tu análisis de {name} en docs/analisis_capturas.md (sección '## {name}')")
    story.append(PageBreak())

    # ---- 4. Ragas
    story.append(P("4. Evaluación con Ragas", "H1"))
    if not after:
        story.append(pending_box("ejecutar las evaluaciones (eval/run_ragas.py) para generar los resultados y volver a construir este PDF."))
    else:
        n = after["n_questions"]; d = after["diagnostics"]
        story.append(P(f"El conjunto de evaluación tiene <b>{n} preguntas</b> ({d['n_in_corpus']} dentro del corpus, con respuesta de referencia, y "
                       f"{d['n_out_of_corpus']} fuera del corpus, cuya respuesta correcta es abstenerse). Se calcularon las cuatro métricas de Ragas con "
                       f"{after['config']['judge_model']} como modelo juez y el mismo modelo de embeddings local para answer_relevancy."))
        story.append(P("faithfulness: qué parte de las afirmaciones de la respuesta está respaldada por el contexto. answer_relevancy: qué tan pertinente es la respuesta a la pregunta. "
                       "context_precision: qué parte de los fragmentos recuperados es relevante y está bien ordenada. context_recall: qué parte de la respuesta de referencia está cubierta por los fragmentos recuperados."))
        header = ["Métrica"] + ([f"{args.before} (todas)", f"{args.after} (todas)", "cambio", f"{args.before} (en corpus)", f"{args.after} (en corpus)"] if before else [f"{args.after} (todas)", f"{args.after} (en corpus)"])
        table_rows = [header]
        for m in METRICS:
            if before:
                b_all, a_all = metric(before, "metrics_all", m), metric(after, "metrics_all", m)
                delta = "-" if b_all is None or a_all is None else f"{a_all - b_all:+.3f}"
                table_rows.append([m, fmt(b_all), fmt(a_all), delta, fmt(metric(before, "metrics_in_corpus", m)), fmt(metric(after, "metrics_in_corpus", m))])
            else:
                table_rows.append([m, fmt(metric(after, "metrics_all", m)), fmt(metric(after, "metrics_in_corpus", m))])
        story += [P("Resultados", "H2"), grid(table_rows, [1.4, 1.0, 1.0, 0.8, 1.25, 1.25] if before else [2.2, 2.2, 2.2]),
                  P("Tabla 1. Promedio de cada métrica (0 a 1; mayor es mejor).", "Cap")]
        chart = RESULTS / "comparison.png"
        story += figure(chart, "Figura 2. Comparación de métricas antes y después de la iteración de mejora.",
                        "ejecutar python eval/compare_runs.py --before baseline --after mejora para generar el gráfico.")
        diag_rows = [["Diagnóstico (sin juez LLM)", args.before if before else "", args.after]]
        diag_rows[0] = [c for c in diag_rows[0] if c != ""]
        for label, key in [("Acierto de recuperación (la fuente correcta aparece en top_k)", "retrieval_hit_rate"),
                           ("Abstención correcta en preguntas fuera del corpus", "abstention_rate_out_of_corpus"),
                           ("Abstención errónea en preguntas dentro del corpus", "false_abstention_rate_in_corpus")]:
            diag_rows.append([label] + ([fmt(before["diagnostics"][key])] if before else []) + [fmt(d[key])])
        story += [P("Control de alucinaciones y recuperación", "H2"), grid(diag_rows, [4.0, 1.3, 1.3] if before else [4.0, 1.5]),
                  P("Tabla 2. La abstención correcta mide si, ante preguntas que el material no cubre, el tutor lo declara en lugar de inventar una respuesta.", "Cap")]
        story.append(P("Análisis e iteración de mejora", "H2"))
        if before:
            story += comparison_paragraphs(before, after, args.before, args.after)
        else:
            low = lowest_metric(after, "metrics_all")
            if low:
                comp, cause, fix = COMPONENT[low]
                story.append(P(f"<b>Métrica más baja: {low}.</b> Se atribuye a {comp}: {cause}. Mejora propuesta: {fix}."))
            story.append(pending_box("ejecutar la segunda corrida (iteración de mejora) para documentar la comparación antes/después."))
    story.append(PageBreak())

    # ---- 5. interfaz
    story.append(P("5. Interfaz de chat desplegada", "H1"))
    story.append(P("La aplicación es una interfaz de chat en Streamlit desplegada en la nube. Mantiene el historial de la conversación, permite preguntas de seguimiento "
                   "(reformuladas antes de buscar), muestra las fuentes que respaldan cada respuesta y, cuando la información no está en el corpus, lo indica sin presentar fuentes como respaldo."))
    ui = [("chat_seguimiento", "Conversación con preguntas de seguimiento", "Figura 3. Conversación que incluye una pregunta de seguimiento que depende del turno anterior."),
          ("chat_fuentes", "Fuentes mostradas bajo la respuesta", "Figura 4. Panel de fuentes (documento, sección y fragmento) que respaldan la respuesta."),
          ("chat_fuera_corpus", "Pregunta fuera del alcance del corpus", "Figura 5. El tutor declara que el material no cubre el tema y no inventa una respuesta.")]
    for name, title, caption in ui:
        story.append(P(title, "H2"))
        story += figure(SHOTS / f"{name}.png", caption, f"tomar la captura de la app desplegada y guardarla como docs/screenshots/{name}.png")
        if name in notes:
            story.append(P(f"<b>Análisis.</b> {notes[name]}"))

    # ---- 6. enlaces
    story += [P("6. Enlaces", "H1")]
    story.append(P(f"<b>URL pública de la aplicación:</b> <link href='{app_url}' color='blue'>{app_url}</link>") if app_url
                 else pending_box("URL pública de la aplicación."))
    story.append(Spacer(1, 6))
    story.append(P(f"<b>Repositorio GitHub:</b> <link href='{repo_url}' color='blue'>{repo_url}</link>") if repo_url
                 else pending_box("enlace al repositorio de GitHub."))

    SimpleDocTemplate(args.out, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch, leftMargin=0.75 * inch,
                      rightMargin=0.75 * inch, title="Avance 2 - Tutor Académico Personalizado (RAG)").build(story)
    print(f"PDF generado: {args.out}")
    if PENDING:
        print("\nPendiente para dejar el PDF completo:")
        for item in dict.fromkeys(PENDING):
            print(f"  - {item}")


if __name__ == "__main__":
    main()
