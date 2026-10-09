# Tutor Académico Personalizado — Asistente experto basado en RAG

Asistente que responde preguntas de **Estadística Inferencial** usando únicamente un material de estudio (base de conocimientos), cita de dónde sale cada respuesta y se abstiene cuando el material no cubre la pregunta.

| | |
|---|---|
| **Enfoque** | Tutor Académico Personalizado |
| **Aplicación desplegada** | `https://PEGAR-AQUI-LA-URL-PUBLICA.streamlit.app` |
| **Repositorio** | `https://github.com/PEGAR-USUARIO/PEGAR-REPO` |
| **Avances** | Avance 1: diseño de prompts · **Avance 2: pipeline RAG, evaluación con Ragas y chat desplegado** |

---

## 1. Flujo RAG implementado

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
├── app.py                    # Interfaz de chat (Streamlit)
├── src/                      # config, ingest, chunking, embeddings, vector_store,
│                             # retriever, system_prompt, llm, rag_pipeline, demo (Avance 1)
├── scripts/                  # build_index.py, ask.py (consulta + captura PNG)
├── knowledge_base/           # Corpus
├── eval/                     # eval_questions.json, run_ragas.py, compare_runs.py, results/
├── tests/                    # 22 pruebas offline
└── docs/                     # avance1.pdf, avance2.pdf, build_pdf_avance2.py, screenshots/
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

Edita `.env` y pega tu clave (`GEMINI_API_KEY=...`). Opcional: fija `GEMINI_MODEL` con el nombre exacto del modelo que veas en AI Studio (recomendado para resultados reproducibles). **`.env` está en `.gitignore`: nunca se sube.**

```bash
python scripts/build_index.py                         # indexa (la 1.ª vez descarga el modelo de embeddings, ~220 MB)
python scripts/ask.py "¿Cuándo uso ANOVA en vez de una prueba t?"
python scripts/ask.py --chat                          # conversación con historial
streamlit run app.py                                  # chat en http://localhost:8501
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
Si aparecen errores 429 (límite de la capa gratuita): usa `--workers 1` y repite con `--reuse-answers` para no regenerar respuestas.

> Resultados: _completar con la tabla de `eval/results/comparison.csv` tras ejecutar las corridas._

## 6. Despliegue (Streamlit Community Cloud)

1. Sube el repositorio a GitHub (sin `.env` ni `secrets.toml`).
2. En [share.streamlit.io](https://share.streamlit.io): **Create app** → repositorio, rama `main`, archivo principal `app.py`.
3. **Advanced settings:** Python **3.12** y, en **Secrets**, pega `GEMINI_API_KEY = "tu-clave"` (y opcionalmente `GEMINI_MODEL = "..."`).
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
