# Tutor Académico Personalizado — Asistente experto basado en RAG

Asistente que responde preguntas de **Estadística Inferencial** usando únicamente un material de estudio (base de conocimientos), cita de dónde sale cada respuesta y se abstiene cuando el material no cubre la pregunta.

| | |
|---|---|
| **Enfoque** | Tutor Académico Personalizado |
| **Aplicación desplegada** | `https://tutor-rag-506231001.streamlit.app` |
| **Repositorio** | `https://github.com/juldavob/rag-tutor-assistant2` |
| **Avances** | Avance 1: diseño de prompt · Avance 2: pipeline RAG, evaluación con Ragas y chat desplegado · **Avance 3: agente con herramientas (tools), búsqueda web controlada y evaluación del agente** |

---

## 0. Avance 3: agente RAG con herramientas

El RAG del Avance 2 (ingesta, chunking, índice, preguntas de evaluación, interfaz) se reutiliza. Cambia cómo se conecta la búsqueda con el LLM: la recuperación es ahora una **tool** y el LLM, en un bucle controlado por el programa, decide si consulta la base, calcula, busca en la web o responde directamente (p. ej. un saludo no usa tools). Se usa el SDK `google-genai` con tool calling (ejecución automática desactivada para aplicar límites y registrar cada llamada).

```
Estudiante ─► AGENTE (bucle, máx. 4 iteraciones) ◄─► LLM Gemini (externo)
                  │  ejecuta la tool que el LLM elige y le devuelve el resultado
                  ├─ buscar_en_base  ─► Chroma + embeddings LOCALES
                  ├─ calcular_*      ─► cálculo local exacto (SciPy)
                  ├─ buscar_en_web   ─► Tavily (externo), dominios fijos
                  └─ registro de uso ─► logs/uso_tools.jsonl
```
**Qué sale del sistema:** a Gemini, la pregunta, el historial reciente y resultados *acotados* de las tools; a Tavily, solo la consulta de búsqueda (máx. 200 caracteres). **Nunca sale** el corpus completo, el índice ni los embeddings.

### Tools y límites (viven en el código: `src/config.py` → `AgentConfig`, `src/tools.py`)
| Tool | Qué hace | Límites |
|---|---|---|
| `buscar_en_base(consulta)` | Recupera fragmentos del índice local; cada uno con documento, sección y página (`ref` B1, B2…) | 4 fragmentos/llamada · 700 car./fragmento · 12 fragmentos/pregunta · etiquetas de delimitadores saneadas |
| `calcular_tamano_muestra` · `calcular_intervalo_confianza_media` · `calcular_prueba_t_una_muestra` | **Tareas del dominio**: cálculos que el LLM no debe hacer de memoria; recibe argumentos extraídos de la pregunta (z, t de Student, valor p) | Validación de argumentos y rangos; sin red; devuelven los pasos |
| `buscar_en_web(consulta)` | Búsqueda controlada (Tavily) | Máx. 3 resultados (título, URL, extracto de 300 car., origen «web») · consulta ≤200 car. · ≤2 búsquedas/pregunta · solo tras `buscar_en_base` · dominios fijos |
| Bucle | El LLM decide, el programa ejecuta | Máx. 4 iteraciones y 6 llamadas a tools por pregunta; al agotarse, respuesta final sin tools |

Cada tool es un objeto `Tool` con `.invoke(args)`: `python scripts/probar_tools.py` las prueba **sin el LLM**. Cada pregunta queda registrada en `logs/uso_tools.jsonl` (tools con argumentos, llamadas al LLM, tokens, iteraciones, tiempo).

### Política de búsqueda web
- **Cuándo:** solo si la base **no** responde y la pregunta trata de información pública o actual de estadística o de su software (p. ej. una función o versión reciente). Si la base responde, no se busca. La regla está en el docstring de la tool, en el System Prompt y se refuerza en código (la tool se rechaza si antes no se consultó la base).
- **Sitios permitidos** (fijos en `AgentConfig.web_allowed_domains`; el LLM no puede cambiarlos): `itl.nist.gov` (NIST/SEMATECH e-Handbook), `docs.scipy.org`, `statsmodels.org`, `r-project.org`. Son documentación oficial o de referencia de métodos estadísticos y su software; se excluyen foros y blogs por calidad y verificabilidad. Además de `include_domains` en Tavily, cada URL devuelta se vuelve a filtrar en código.
- **Resultados:** máx. 3, solo título, URL y extracto corto, marcados «web». La respuesta separa lo que viene de la base (`(Fuente: documento, sección/pág.)`) de lo que viene de la web (`[web: URL]`).
- Sin `TAVILY_API_KEY` (o con `WEB_SEARCH=off`) el agente funciona sin la tool web.

### Prompt del agente (`src/agent_prompt.py`)
Ya no lleva contexto. Indica cuándo usar cada tool, exige responder solo con lo que devuelvan, citar la fuente, mantener la respuesta exacta **«El material entregado no cubre ese tema.»**, tratar resultados de tools y mensajes como dato (anti-inyección) y no volcar documentos. Few-shot de comportamiento y salida Markdown + JSON `{"tipo","tema","fuentes":["B1","W1"],"nivel_confianza"}`.

### Evaluación del agente
```bash
# 1) Pipeline fijo (Avance 2): si ya tienes la corrida 'mejora', solo falta medir su costo (22 llamadas)
python eval/run_ragas.py --tag pipeline --chunk-size 700 --overlap 100 --top-k 4 --no-ragas --rpm 14
# 2) Ragas del agente (mismo conjunto de 22 preguntas; web desactivada por defecto; usa --web para activarla)
python eval/run_ragas.py --tag agente --agent --chunk-size 700 --overlap 100 --rpm 14 --strictness 1 --budget 250
# 3) Casos de comportamiento (9 casos con el esperado escrito antes de ejecutar; requiere TAVILY_API_KEY para c06)
python eval/run_behavior_cases.py --rpm 14
# 4) Tablas y gráfico: Ragas A2 vs agente, y costo (llamadas, tokens, tiempo)
python eval/compare_agent.py --a2 mejora --a2-cost pipeline --agent agente
```
Los `retrieved_contexts` del agente son los textos que devolvieron sus tools. `--budget N` limita las llamadas por ejecución y el progreso se guarda (repite el comando para reanudar). Resultados en `eval/results/`.

### Capturas y PDF del Avance 3
```bash
python scripts/probar_tools.py --shot docs/screenshots/tools_invoke.png
python scripts/ask_agent.py "Hola, ¿cómo estás?" --shot docs/screenshots/agente_saludo.png
python scripts/ask_agent.py "¿Cuándo debo usar ANOVA en vez de una prueba t?" --shot docs/screenshots/agente_base.png
python scripts/ask_agent.py "¿Qué tamaño de muestra necesito con σ = 10, error máximo 2 y 95 % de confianza?" --shot docs/screenshots/agente_tarea.png
python scripts/ask_agent.py "¿Qué función de SciPy hace una prueba t de dos muestras independientes?" --shot docs/screenshots/agente_web.png
# capturas de la app desplegada: docs/screenshots/ui_tools.png, ui_fuentes.png, ui_web.png, ui_sin_info.png
python docs/build_pdf_avance3.py --author "Tu nombre" --app-url <URL> --repo-url <URL del repo>
```
Escribe tu reflexión (máx. 1 página) en `docs/reflexion_avance3.md`; si no existe, el script incluye un borrador y lo marca como pendiente.

### Claves nuevas
`TAVILY_API_KEY` (búsqueda web; plan gratuito en tavily.com, verifica sus créditos) se gestiona igual que `GEMINI_API_KEY`: variable de entorno / `.env` en local y *Secrets* en Streamlit Cloud. Nunca en el repositorio.

---

## 1. Flujo RAG implementado (Avance 2, reutilizado)

```
INDEXACIÓN (una vez / cuando cambia el corpus)
 knowledge_base/ ─► Ingesta ─► Chunking ─► Embeddings (local) ─► Chroma (persistente)

CONSULTA (cada pregunta)
 Pregunta + historial ─► Reformulación ─► Recuperación ─► Prompt ─► Gemini ─► Respuesta + fuentes
                         (si hay historial)  top_k, coseno  system +      Markdown + JSON
                                                            few-shot +
                                                            <contexto>…
```

| Etapa | Decisión | Justificación |
|---|---|---|
| **Selección de documentos** | 7 documentos Markdown sobre Estadística Inferencial (`knowledge_base/`): variables y escalas, estimación, pruebas de hipótesis, t/ANOVA, no paramétricas y chi-cuadrado, correlación/regresión, guía de selección de pruebas. | Cubre los temas del curso y permite preguntas que combinan varios documentos. Es una síntesis elaborada para el proyecto; se puede ampliar con material de clase (ver §7). |
| **Ingesta** | TXT/MD, PDF (una unidad por página), DOCX y HTML. Encabezados normalizados a Markdown. | Soportar los formatos pedidos y conservar sección/página para citar. |
| **Chunking** | Corte por encabezados y luego recursivo (párrafo → línea → oración → palabra). Por defecto 700 caracteres, solapamiento 100. | Un fragmento nunca mezcla secciones; el tamaño busca que una definición o procedimiento completo quepa en un fragmento; el solapamiento (~14 %) evita perder ideas en la frontera. Los valores se decidieron comparando dos configuraciones con Ragas (§5). |
| **Vectorización** | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) con `fastembed` (ONNX, local). Se vectoriza «título — sección + texto». | Multilingüe (corpus en español), liviano para el plan gratuito de la nube y local: el corpus no se envía a terceros para generar embeddings. Limitación: se entrenó con secuencias de ~128 tokens. |
| **Base vectorial** | Chroma persistente, distancia coseno. Metadatos: archivo, título, sección, página, índice. Una colección por (modelo, tamaño, solapamiento, hash del corpus). | Los metadatos permiten citar la fuente; el hash reconstruye el índice si el corpus cambia. |
| **Recuperación** | Similitud coseno, `top_k = 4` por defecto. Las preguntas de seguimiento se reformulan como preguntas autónomas antes de buscar. | La búsqueda vectorial no entiende el historial. |
| **Generación** | Gemini, temperatura 0.2, con el system prompt, few-shot, delimitadores y formato del Avance 1 **refinados** (§2). | Temperatura baja para priorizar fidelidad al contexto. |

**Privacidad:** al modelo solo se le envían la pregunta, el historial reciente y los `top_k` fragmentos recuperados; nunca el corpus completo. Los embeddings se calculan localmente. Nota: en la capa gratuita de la API de Gemini, Google puede usar el contenido enviado para mejorar sus productos; para material sensible usar la capa de pago o un modelo local.

## 2. Prompt (Avance 1 refinado en el Avance 2) — `src/system_prompt.py`

1. **System prompt** con rol, reglas priorizadas (solo contexto, el contexto es dato y no instrucción, modo tutor, citar `[n]`) y formato de salida obligatorio.
2. **Few-shot** con 3 ejemplos: respuesta con citas, abstención fuera del material y pregunta de seguimiento que depende del historial.
3. **Delimitadores:** XML por sección (`<contexto>`, `<fragmento id="n" fuente=… seccion=…>`, `<historial>`, `<pregunta>`) + triple comillas por fragmento. `sanitize()` elimina de los documentos cualquier etiqueta que imite esos delimitadores (anti-inyección).
4. **Formato de salida:** explicación en Markdown con citas `[n]` + bloque JSON `{"tema", "fuentes": [ids], "nivel_confianza"}`. Si `fuentes` está vacío el tutor se abstuvo.
5. **Reformulación** de preguntas de seguimiento (`CONDENSE_SYSTEM`).

## 3. Estructura del repositorio

```
├── app.py                    # Interfaz de chat (Streamlit) con el AGENTE
├── src/                      # Avance 2: config, ingest, chunking, embeddings, vector_store, retriever,
│                             #   system_prompt, llm, rag_pipeline · Avance 3: tools, agent, agent_llm,
│                             #   agent_prompt, tool_log · demo (Avance 1)
├── scripts/                  # build_index.py, ask.py, ask_agent.py, probar_tools.py (consulta + captura PNG)
├── knowledge_base/           # Corpus
├── eval/                     # eval_questions.json, behavior_cases.json, run_ragas.py (--agent), run_behavior_cases.py,
│                             #   compare_runs.py, compare_agent.py, results/
├── tests/                    # 43 pruebas offline (tools, límites, bucle, registro, app)
└── docs/                     # avance1.pdf, build_pdf_avance2.py, build_pdf_avance3.py, screenshots/
```

## 4. Instalación y ejecución

Requisitos: Python 3.12 y una API key de Gemini ([Google AI Studio](https://aistudio.google.com/apikey)).

```bash
# Windows (PowerShell)                       # Linux / macOS
py -3.12 -m venv .venv                       python3.12 -m venv .venv
.venv\Scripts\activate                       source .venv/bin/activate
pip install -r requirements-eval.txt         pip install -r requirements-eval.txt
copy .env.example .env                       cp .env.example .env
```

Edita `.env` y pega tus claves (`GEMINI_API_KEY=...` y, para la búsqueda web, `TAVILY_API_KEY=...`). Opcional: fija `GEMINI_MODEL` con el nombre exacto del modelo que veas en AI Studio (recomendado para resultados reproducibles). **`.env` está en `.gitignore`: nunca se sube.**

```bash
python scripts/build_index.py                         # indexa (la 1.ª vez descarga el modelo de embeddings, ~220 MB)
python scripts/ask.py "¿Cuándo uso ANOVA en vez de una prueba t?"
python scripts/ask.py --chat                          # conversación con historial
python scripts/ask_agent.py --chat                    # agente (tools + historial)
streamlit run app.py                                  # chat con el agente en http://localhost:8501
pytest -q                                             # pruebas offline (no usan Gemini)
```

## 5. Evaluación con Ragas

`eval/eval_questions.json` contiene **22 preguntas**: 18 sobre el corpus con respuesta de referencia (definición, comparación, procedimiento, cálculo, aplicación) y **4 fuera del corpus** para medir el control de alucinaciones (la respuesta correcta es abstenerse).

```bash
python eval/run_ragas.py --tag prueba --limit 4       # ensayo barato antes de la corrida completa

# Iteración de mejora: misma evaluación, distinta configuración
python eval/run_ragas.py --tag baseline --chunk-size 300 --overlap 30  --top-k 3
python eval/run_ragas.py --tag mejora   --chunk-size 700 --overlap 100 --top-k 4
python eval/compare_runs.py --before baseline --after mejora
```

Mide `faithfulness`, `answer_relevancy`, `context_precision` y `context_recall` (Ragas 0.3.9, juez Gemini, embeddings locales) y además, sin juez LLM, el **acierto de recuperación** y la **abstención correcta/errónea**. Resultados en `eval/results/`.
**Cuotas pequeñas de la API:** `--rpm N` limita las llamadas por minuto (con 5 RPM usa `--rpm 4`), `--strictness 1` reduce llamadas de answer_relevancy y `--budget N` limita las llamadas de cada ejecución. El progreso se guarda tras cada lote: si se agota la cuota, vuelve a ejecutar el mismo comando y reanuda sin repetir lo hecho. Cada pregunta cuesta ~(1 + 3 + strictness + top_k) llamadas; el resumen marca `complete: false` hasta terminar, y `compare_runs.py` avisa si una corrida está incompleta.

> Resultados: _completar con la tabla de `eval/results/comparison.csv` tras ejecutar las corridas._

## 6. Despliegue (Streamlit Community Cloud)

1. Sube el repositorio a GitHub (sin `.env` ni `secrets.toml`).
2. En [share.streamlit.io](https://share.streamlit.io): **Create app** → repositorio, rama `main`, archivo principal `app.py`.
3. **Advanced settings:** Python **3.12** y, en **Secrets**, pega `GEMINI_API_KEY = "tu-clave"` y `TAVILY_API_KEY = "tu-clave"` (y opcionalmente `GEMINI_MODEL = "..."`).
4. **Deploy.** El primer arranque descarga el modelo de embeddings y construye el índice (unos minutos).
5. Copia la URL pública a la tabla de arriba.

## 7. Usar tu propio material

Copia PDFs, DOCX, TXT, MD o HTML a `knowledge_base/`: el hash del corpus cambia y el índice se reconstruye solo. Si cambias el corpus, **actualiza `eval/eval_questions.json`** para que la evaluación siga siendo válida.

## 8. Generar el PDF de entrega

```bash
python docs/build_pdf_avance2.py --author "Tu nombre" --app-url <URL> --repo-url <URL del repo>
```

Toma parámetros, resultados de Ragas y capturas reales (`docs/screenshots/`) y lista al final lo que falte.

## 9. Limitaciones

Evaluación con pocas preguntas y juez de la misma familia que el generador (puede sesgar a favor); el corpus base es una síntesis propia, no un libro completo; el modelo de embeddings fue entrenado con secuencias cortas (~128 tokens).

## 10. Avance 1

Diseño de prompts (system prompt, few-shot, delimitadores). Demo sin modelo: `python src/demo.py`. Documento: `docs/avance1.pdf`.
