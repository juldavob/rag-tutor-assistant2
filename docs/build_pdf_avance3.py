"""
build_pdf_avance3.py — genera docs/avance3.pdf con datos REALES del proyecto (lo que falte se marca PENDIENTE).

Toma: límites del código (config.AGENT_CONFIG), resultados de eval/results (Ragas del Avance 2 y del agente, costo,
casos de comportamiento), capturas de docs/screenshots (consola: scripts/ask_agent.py y scripts/probar_tools.py con
--shot; interfaz desplegada: manuales) y, opcionalmente, docs/reflexion_avance3.md (tu reflexión, máx. 1 página).

    python docs/build_pdf_avance3.py --author "Tu nombre" --app-url <URL> --repo-url <URL> [--a2 mejora --agent agente]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pdf_avance2 import (PENDING, RESULTS, ROOT, SHOTS, BLUE, P, fmt, figure, grid, image_flowable,  # noqa: E402
                               metric, pending_box)
from config import AGENT_CONFIG as C, MODEL_CONFIG  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import HRFlowable, PageBreak, SimpleDocTemplate, Spacer  # noqa: E402

DOCS = ROOT / "docs"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load(tag):
    path = RESULTS / f"{tag}_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def make_diagram(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 8.2); ax.axis("off")

    def box(x, y, w, h, title, detail, face, edge, lw=1.6, ls="-"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc=face, ec=edge, lw=lw, ls=ls))
        ax.text(x, y + 0.2, title, ha="center", va="center", fontsize=9.8, fontweight="bold")
        ax.text(x, y - 0.3, detail, ha="center", va="center", fontsize=8, color="#333")

    def arrow(p, q, color="#555", ls="-", label=None, lx=None, ly=None, fs=7.6):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.4, ls=ls))
        if label:
            ax.text(lx if lx is not None else (p[0] + q[0]) / 2, ly if ly is not None else (p[1] + q[1]) / 2 + 0.18, label,
                    ha="center", fontsize=fs, color=color)

    RED = "#C62828"
    box(1.4, 4.1, 2.2, 1.5, "Estudiante", "chat Streamlit\n(URL pública)", "#FDF1E3", "#B0651E")
    box(5.0, 4.1, 2.9, 2.6, "AGENTE (bucle)", f"decide, ejecuta tools y\ndevuelve resultados.\nmáx. {C.max_iterations} iteraciones,\n{C.max_tool_calls} llamadas a tools", "#EDE7F6", "#5E35B1")
    box(5.0, 7.2, 2.9, 1.2, "LLM (Gemini API)", "servicio EXTERNO", "#FFEBEE", RED, 2, "--")
    box(9.4, 6.0, 2.9, 1.2, "buscar_en_base", f"local · máx. {C.kb_max_fragments} frag. · {C.kb_max_chars} car.", "#E8F5E9", "#2E7D32")
    box(9.4, 4.1, 2.9, 1.2, "calcular_* (3 tools)", "local · tamaño de muestra,\nIC de la media, prueba t", "#E8F5E9", "#2E7D32")
    box(9.4, 2.2, 2.9, 1.2, "buscar_en_web", f"max. {C.web_max_results} result. · dominios fijos", "#FFEBEE", RED, 2, "--")
    box(13.0, 6.0, 2.3, 1.2, "Chroma + embeddings", "LOCALES\n(el corpus no sale)", "#E8F5E9", "#2E7D32")
    box(13.0, 2.2, 2.3, 1.2, "Tavily (buscador)", "servicio EXTERNO", "#FFEBEE", RED, 2, "--")
    box(5.0, 1.2, 2.9, 1.0, "Registro de uso", "logs/uso_tools.jsonl (local)", "#EAF2FB", "#2C5F8A")
    arrow((2.5, 4.1), (3.55, 4.1), label="pregunta", ly=4.3)
    arrow((5.0, 5.4), (5.0, 6.6), RED, "--", "SALE: pregunta + historial\n+ resultados acotados de tools", 7.4, 6.0, 7.2)
    arrow((6.45, 4.7), (7.95, 5.8)); arrow((6.45, 4.1), (7.95, 4.1)); arrow((6.45, 3.5), (7.95, 2.4))
    ax.text(7.2, 5.55, "args", fontsize=7.6, color="#555")
    arrow((10.85, 6.0), (11.85, 6.0), "#2E7D32")
    arrow((10.85, 2.2), (11.85, 2.2), RED, "--", "SALE: consulta (max. %d car.)" % C.web_query_max_chars, 11.35, 2.7, 7.2)
    arrow((5.0, 2.8), (5.0, 1.7), "#2C5F8A")
    ax.text(0.3, 0.45, "Línea roja discontinua = información que sale del sistema (a Gemini: pregunta, historial y resultados acotados de tools; a Tavily: solo la consulta de búsqueda).\n"
            "Nunca sale: el corpus completo, el índice vectorial ni los embeddings. Los resultados de las tools vuelven al agente y de ahí al LLM, que redacta la respuesta final.",
            fontsize=8, color="#444")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def readings(sidecar):
    if not sidecar.exists():
        return []
    out = []
    for t in json.loads(sidecar.read_text(encoding="utf-8")).get("turns", []):
        tools = "; ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in t["tool_calls"]) or "ninguna"
        out.append(f"<b>Consulta:</b> <i>{t['question']}</i>. <b>Tools:</b> {tools}. <b>Llamadas al LLM:</b> {t['llm_calls']}; "
                   f"tokens: {t['prompt_tokens'] + t['output_tokens']}; tiempo: {t['latency_s']} s; tipo de respuesta: {t['metadata'].get('tipo') or '?'}.")
    return out


EXPLAIN = {
    "faithfulness": ("sube", "el agente redacta solo con lo que devuelven las tools", "baja", "el agente complementa con conocimiento propio o sus resultados de cálculo no se reconocen como contexto"),
    "answer_relevancy": ("sube", "las respuestas son más directas y citadas", "baja", "las respuestas incluyen más texto accesorio (explicaciones de pasos, citas) o hay más abstenciones"),
    "context_precision": ("sube", "las consultas reformuladas por el LLM traen fragmentos más específicos", "baja", "varias llamadas a la base acumulan fragmentos poco relevantes"),
    "context_recall": ("sube", "las búsquedas múltiples cubren más aspectos de la pregunta", "baja", "el LLM formula consultas más estrechas o decide no buscar"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="[Nombre del estudiante]")
    ap.add_argument("--app-url", default="")
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--a2", default="mejora", help="corrida de Ragas del Avance 2")
    ap.add_argument("--agent", default="agente")
    ap.add_argument("--out", default=str(DOCS / "avance3.pdf"))
    args = ap.parse_args()
    a2, ag = load(args.a2), load(args.agent)
    diagram = DOCS / "images" / "arquitectura_agente.png"
    make_diagram(diagram)
    S = []

    # portada
    S += [Spacer(1, 1.1 * inch), P("Tutor Académico Personalizado", "Cover"), P("Avance 3 - Agente RAG con herramientas", "CoverSub"),
          Spacer(1, 10), HRFlowable(width="60%", color=BLUE, thickness=1.2, hAlign="CENTER"), Spacer(1, 14),
          P("Asignatura piloto: Estadística Inferencial", "CoverSub"), P(f"Autor: {args.author}", "CoverSub"), Spacer(1, 14)]
    S.append(P(f"Aplicación: <link href='{args.app_url}' color='blue'>{args.app_url}</link>", "CoverSub") if args.app_url
             else pending_box("URL pública de la aplicación (opción --app-url)."))
    S.append(Spacer(1, 6))
    S.append(P(f"Repositorio: <link href='{args.repo_url}' color='blue'>{args.repo_url}</link>", "CoverSub") if args.repo_url
             else pending_box("enlace al repositorio (opción --repo-url)."))
    S.append(PageBreak())

    # 1. arquitectura
    S += [P("1. Arquitectura del agente", "H1"),
          P("El agente reutiliza la ingesta, el chunking y el índice del Avance 2. Cambia cómo se conecta la búsqueda con el LLM: la recuperación "
            "pasó a ser una tool y el LLM decide, en un bucle controlado por el programa, si consulta la base, calcula, busca en la web o responde directamente."),
          image_flowable(diagram, max_h=4.9 * inch),
          P("Figura 1. Flujo pregunta - LLM - tool - resultado - respuesta, indicando qué información sale del sistema y hacia dónde.", "Cap")]
    rows = [["Componente", "Qué hace", "Límites aplicados EN EL CÓDIGO"],
            ["buscar_en_base", "Recupera fragmentos del índice local; cada uno con documento, sección y página.",
             f"{C.kb_max_fragments} fragmentos por llamada; {C.kb_max_chars} caracteres por fragmento; {C.kb_max_fragments_per_question} fragmentos por pregunta; consulta de hasta {C.query_max_chars} caracteres; se sanean las etiquetas de delimitadores."],
            ["calcular_tamano_muestra, calcular_intervalo_confianza_media, calcular_prueba_t_una_muestra",
             "Tareas del dominio: cálculos exactos que el LLM recibe por argumentos extraídos de la pregunta (z, t de Student, valor p con SciPy).",
             "Validación de argumentos y rangos (sigma, E, n, nivel de confianza); no tienen acceso a red ni al corpus; devuelven los pasos del cálculo."],
            ["buscar_en_web", "Búsqueda controlada con Tavily bajo la política «primero la base».",
             f"Máx. {C.web_max_results} resultados (título, URL, extracto de {C.web_snippet_chars} car., marcados «web»); consulta de hasta {C.web_query_max_chars} car.; máx. {C.web_max_calls} búsquedas por pregunta; "
             "solo si antes se llamó a buscar_en_base; lista de dominios fija y filtro adicional de cada URL."],
            ["Bucle del agente", "El LLM decide; el programa ejecuta la tool y devuelve el resultado.",
             f"Máx. {C.max_iterations} iteraciones y {C.max_tool_calls} llamadas a tools por pregunta; al agotarse, una llamada final sin tools. Un saludo se responde sin tools."],
            ["Registro de uso", "Una línea JSON por pregunta: tools con argumentos, llamadas al LLM, tokens, iteraciones y tiempo.", "logs/uso_tools.jsonl (local)."]]
    S += [P("2. Tools y límites", "H1"), grid(rows, [1.5, 2.3, 2.9]), Spacer(1, 8)]
    S += [P("3. Política de búsqueda web", "H1"),
          P(f"<b>Cuándo se usa:</b> solo si la base no responde y la pregunta trata de información pública o actual de estadística o de su software (por ejemplo una función o versión reciente). "
            f"Si la base responde, no se busca. La regla está en el docstring de la tool, en el System Prompt y se refuerza en código (la tool se rechaza si no se consultó antes la base).")
          , P(f"<b>Sitios permitidos:</b> {', '.join(C.web_allowed_domains)}. Son documentación oficial o de referencia de métodos estadísticos y de su software; se excluyen foros y blogs "
              "por calidad y verificabilidad. La lista está fija en el código (include_domains y filtro posterior de URL) y el LLM no puede cambiarla.")
          , P(f"<b>Qué sale del sistema:</b> únicamente la consulta de búsqueda (máx. {C.web_query_max_chars} caracteres); nunca texto de la base. "
              f"<b>Qué vuelve:</b> máximo {C.web_max_results} resultados con título, URL y extracto corto, marcados con origen «web». "
              "La respuesta separa lo que viene de la base («Fuente: documento, sección o página») de lo que viene de la web («[web: URL]»).")]
    S += [P("4. Prompt del sistema actualizado", "H1"),
          P("El prompt ya no lleva el contexto. Indica cuándo usar cada tool, exige responder solo con lo que devuelvan las tools y citar la fuente, "
            "mantiene la respuesta exacta «El material entregado no cubre ese tema.» cuando la información no está, trata los resultados de las tools como dato y no como instrucciones, "
            "y rechaza volcar documentos. Incluye ejemplos few-shot de comportamiento (saludo, base, tarea, web, sin información y rechazo) y un formato de salida con bloque JSON "
            "(tipo, tema, fuentes con referencias B1/W1, nivel de confianza).")]
    S.append(PageBreak())

    # 5. capturas
    S += [P("5. Ejecución: tools en uso", "H1")]
    shots = [("tools_invoke", "Tools probadas sin el LLM (.invoke)"), ("agente_saludo", "Un saludo, sin tools"),
             ("agente_base", "Pregunta respondida con la base"), ("agente_tarea", "Tarea del dominio (cálculo)"), ("agente_web", "Búsqueda web")]
    for name, title in shots:
        S.append(P(title, "H2"))
        S += figure(SHOTS / f"{name}.png", f"Captura {name}.png", f"crear docs/screenshots/{name}.png (scripts/ask_agent.py ... --shot o scripts/probar_tools.py --shot)")
        for text in readings(SHOTS / f"{name}.json"):
            S.append(P(f"<b>Lectura de la ejecución.</b> {text}"))
    S.append(PageBreak())

    # 6. Ragas
    S.append(P("6. Evaluación con Ragas: pipeline fijo (Avance 2) frente al agente", "H1"))
    if not (a2 and ag):
        S.append(pending_box(f"ejecutar las corridas de Ragas ('{args.a2}' y '{args.agent}') para completar esta sección."))
    else:
        S.append(P(f"Se ejecutó el mismo conjunto de {ag['n_questions']} preguntas del Avance 2 sobre el agente. Los retrieved_contexts del agente son los textos que devolvieron sus tools "
                   "(fragmentos de la base y resultados de las calculadoras). En esta comparación la búsqueda web se mantuvo desactivada para medir el comportamiento frente a la misma base; "
                   f"su comportamiento se evalúa en los casos de la sección 7. Juez: {ag['config']['judge_model']}."))
        rows = [["Métrica", f"Avance 2 ({args.a2})", f"Agente ({args.agent})", "Cambio", "A2 en corpus", "Agente en corpus"]]
        changes = {}
        for m in METRICS:
            b, a = metric(a2, "metrics_all", m), metric(ag, "metrics_all", m)
            changes[m] = None if b is None or a is None else a - b
            rows.append([m, fmt(b), fmt(a), "-" if changes[m] is None else f"{changes[m]:+.3f}", fmt(metric(a2, "metrics_in_corpus", m)), fmt(metric(ag, "metrics_in_corpus", m))])
        S += [grid(rows, [1.4, 1.1, 1.1, 0.8, 1.1, 1.2]), P("Tabla 1. Promedio de cada métrica (0 a 1).", "Cap")]
        S += figure(RESULTS / "agent_ragas.png", "Figura 2. Ragas: pipeline fijo vs. agente.", "ejecutar python eval/compare_agent.py para generar el gráfico.")
        d2, dg = a2["diagnostics"], ag["diagnostics"]
        S.append(P(f"<b>Diagnóstico sin juez.</b> Acierto de recuperación: {fmt(d2['retrieval_hit_rate'])} (Avance 2) frente a {fmt(dg['retrieval_hit_rate'])} (agente). "
                   f"Abstención correcta fuera del corpus: {fmt(d2['abstention_rate_out_of_corpus'])} frente a {fmt(dg['abstention_rate_out_of_corpus'])}; "
                   f"abstención errónea dentro del corpus: {fmt(d2['false_abstention_rate_in_corpus'])} frente a {fmt(dg['false_abstention_rate_in_corpus'])}."))
        S.append(P("Análisis: qué mejoró, qué empeoró y por qué", "H2"))
        for m in METRICS:
            ch = changes[m]
            if ch is None:
                S.append(P(f"<b>{m}</b>: no se pudo calcular en alguna de las corridas."))
                continue
            up_word, up_cause, down_word, down_cause = EXPLAIN[m]
            if abs(ch) < 0.03:
                S.append(P(f"<b>{m}</b> se mantuvo estable ({ch:+.3f}); la diferencia está dentro del ruido esperable con tan pocas preguntas."))
            else:
                S.append(P(f"<b>{m}</b> {'mejoró' if ch > 0 else 'empeoró'} ({ch:+.3f}). Explicación probable: {up_cause if ch > 0 else down_cause}. "
                           "Conviene contrastarla con las respuestas por pregunta (eval/results/*_per_question.csv)."))
        S.append(P("<b>Limitaciones.</b> Pocas preguntas, juez de la misma familia que el generador y contextos del agente de naturaleza distinta (incluyen resultados de cálculo); "
                   "las diferencias pequeñas no son concluyentes."))
    S.append(PageBreak())

    # 7. casos
    S.append(P("7. Casos de comportamiento", "H1"))
    bpath = RESULTS / "behavior_results.json"
    if not bpath.exists():
        S.append(pending_box("ejecutar python eval/run_behavior_cases.py para generar la tabla de casos."))
    else:
        cases = json.loads(bpath.read_text(encoding="utf-8"))
        rows = [["Caso", "Esperado (escrito antes de ejecutar)", "Obtenido", "Tools usadas", "Llam. LLM", "Cumple"]]
        for c in cases:
            got = c["respuesta"][:170].replace("\n", " ") + ("..." if len(c["respuesta"]) > 170 else "")
            if c["cumple"] == "NO":
                got += f" [Falló: {c['fallos']}]"
            elif c["cumple"] in ("OMITIDO", "ERROR"):
                got = c["fallos"]
            rows.append([f"{c['id']} - {c['tipo']}: {c['entrada'][:110]}", c["esperado"], got, c["tools"][:150], c["llamadas_llm"], c["cumple"]])
        S += [grid(rows, [1.35, 1.35, 1.6, 1.25, 0.5, 0.6]), P("Tabla 2. Casos de comportamiento: esperado, obtenido, tools usadas y llamadas al LLM.", "Cap")]

    # 8. costo
    S.append(P("8. Costo: pipeline fijo frente a agente", "H1"))
    cpath = RESULTS / "agent_cost.csv"
    if not cpath.exists():
        S.append(pending_box("ejecutar python eval/compare_agent.py (requiere una corrida del pipeline con --no-ragas y la del agente)."))
        ratio = {}
    else:
        data = list(csv.DictReader(cpath.open(encoding="utf-8")))
        S += [grid([["Indicador (promedio por pregunta)", "Pipeline fijo", "Agente", "Agente / pipeline"]] +
                   [[r["indicador"], r["pipeline_fijo"], r["agente"], r["razon_agente_sobre_pipeline"] or "-"] for r in data], [2.8, 1.3, 1.3, 1.3]),
              P("Tabla 3. Costo promedio sobre las mismas preguntas.", "Cap")]
        ratio = {r["indicador"]: r for r in data}
    S.append(P("Reflexión", "H2"))
    reflex = DOCS / "reflexion_avance3.md"
    if reflex.exists():
        for para in reflex.read_text(encoding="utf-8").split("\n\n"):
            S.append(P(para.strip().replace("\n", " ")))
    else:
        calls = ratio.get("llamadas al LLM (prom.)")
        txt = ("El agente aporta flexibilidad: decide cuándo buscar, puede buscar varias veces, hace cálculos exactos que el modelo no debe improvisar y puede completar con la web de forma controlada. "
               "A cambio, cada pregunta cuesta más: ")
        txt += (f"en promedio {calls['agente']} llamadas al LLM frente a {calls['pipeline_fijo']} del pipeline fijo, con el consiguiente aumento de tokens y de tiempo de respuesta. "
                if calls else "requiere más llamadas al LLM, tokens y tiempo que el pipeline fijo. ")
        txt += ("Para preguntas simples de la base el pipeline fijo basta y es más barato y predecible; el agente se justifica cuando hay tareas o información que la base no contiene. "
                "Los límites en código (iteraciones, fragmentos, resultados web) acotan ese costo y el riesgo de fuga de información.")
        S.append(P(txt))
        PENDING.append("revisar y adaptar la reflexión (máx. 1 página); puedes escribir la tuya en docs/reflexion_avance3.md")
    S.append(PageBreak())

    # 9. UI
    S += [P("9. Interfaz de chat desplegada", "H1"),
          P("La interfaz usa el agente, mantiene el historial y, en cada respuesta, muestra las tools usadas con sus argumentos, las fuentes separadas por origen (base: documento y sección o página; web: título y URL) "
            "y el costo (llamadas al LLM, tokens y tiempo). Si la información no está en la base ni en los sitios permitidos, lo indica sin inventar. Las claves (Gemini y Tavily) viven en los secrets de la plataforma.")]
    for name, title in [("ui_tools", "Tools usadas con sus argumentos"), ("ui_fuentes", "Fuentes de la base (documento y sección/página)"),
                        ("ui_web", "Fuentes web (título y URL)"), ("ui_sin_info", "Pregunta sin información en la base ni en los sitios permitidos")]:
        S.append(P(title, "H2"))
        S += figure(SHOTS / f"{name}.png", f"Captura {name}.png de la aplicación desplegada.", f"tomar la captura de la app desplegada y guardarla como docs/screenshots/{name}.png")
    S.append(P("10. Enlaces", "H1"))
    S.append(P(f"<b>URL pública:</b> <link href='{args.app_url}' color='blue'>{args.app_url}</link>") if args.app_url else pending_box("URL pública."))
    S.append(Spacer(1, 6))
    S.append(P(f"<b>Repositorio:</b> <link href='{args.repo_url}' color='blue'>{args.repo_url}</link>") if args.repo_url else pending_box("enlace al repositorio."))

    SimpleDocTemplate(args.out, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch, leftMargin=0.75 * inch,
                      rightMargin=0.75 * inch, title="Avance 3 - Agente RAG con herramientas").build(S)
    print(f"PDF generado: {args.out}")
    if PENDING:
        print("\nPendiente para dejar el PDF completo:")
        for item in dict.fromkeys(PENDING):
            print(f"  - {item}")


if __name__ == "__main__":
    main()
