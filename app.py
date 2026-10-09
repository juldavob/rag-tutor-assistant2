"""
app.py — Interfaz de chat del Tutor Académico (Streamlit).

Ejecutar localmente:   streamlit run app.py
Desplegar:             Streamlit Community Cloud (archivo principal: app.py)

Características exigidas en el Avance 2:
  * Conversación real: se mantiene el historial y se pueden hacer preguntas
    de seguimiento (el pipeline reformula la pregunta usando el historial).
  * Cada respuesta muestra las fuentes (documento, sección y fragmento) que
    la respaldan.
  * Si la información no está en el corpus, el tutor lo dice y la interfaz
    lo señala sin presentar fuentes como si respaldaran la respuesta.
  * La API key se lee de variables de entorno / st.secrets, nunca del repo.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:  # Algunos entornos (Streamlit Cloud) traen un sqlite3 antiguo; solo se usa si está instalado
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st  # noqa: E402

from config import EMBEDDING_CONFIG, MODEL_CONFIG, RETRIEVAL_CONFIG, SUBJECT_CONFIG  # noqa: E402
from llm import LLMError, get_api_key  # noqa: E402
from rag_pipeline import build_pipeline  # noqa: E402

st.set_page_config(page_title=f"Tutor de {SUBJECT_CONFIG.name}", page_icon="📊", layout="centered")

EXAMPLES = [
    "¿Cuándo debo usar ANOVA en vez de una prueba t?",
    "¿Cómo se interpreta un intervalo de confianza del 95%?",
    "¿Qué prueba uso para comparar tres proveedores?",
    "¿Cómo se calcula la transformada de Laplace?",  # fuera del corpus (demuestra el control de alucinaciones)
]


@st.cache_resource(show_spinner="Cargando la base de conocimientos (la primera vez puede tardar un par de minutos)…")
def load_pipeline():
    return build_pipeline()


def render_sources(sources, title):
    if not sources:
        return
    with st.expander(title):
        for s in sources:
            page = f", pág. {s['page']}" if s["page"] else ""
            st.markdown(f"**[{s['rank']}] {s['source']}** — {s['section'] or 'sin sección'}{page} · similitud {s['score']:.2f}")
            st.caption(s["text"])


def render_assistant(msg):
    st.markdown(msg["content"])
    meta = msg["metadata"]
    if msg["abstained"]:
        st.info("ℹ️ El material cargado no respalda esta consulta, así que el tutor no inventa una respuesta.")
        render_sources(msg["sources"], "Fragmentos consultados (no respaldan la respuesta)")
    else:
        st.caption(f"Tema: {meta.get('tema', '—')} · Confianza: {meta.get('nivel_confianza', '—')}")
        cited = [s for s in msg["sources"] if s["cited"]]
        others = [s for s in msg["sources"] if not s["cited"]]
        render_sources(cited, f"📚 Fuentes que respaldan la respuesta ({len(cited)})")
        render_sources(others, f"Otros fragmentos recuperados ({len(others)})")


# ---------------------------------------------------------------- estado
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Tutor Académico")
    st.write(f"Asignatura: **{SUBJECT_CONFIG.name}**")
    st.caption("Responde solo con el material cargado y muestra de dónde sale cada respuesta. "
               "Al modelo solo se le envían tu pregunta, el historial reciente y los fragmentos recuperados, "
               "nunca el material completo.")
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.subheader("Prueba con estas preguntas")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    with st.expander("Configuración del sistema"):
        st.write(f"Modelo: `{MODEL_CONFIG.model_name}`")
        st.write(f"Embeddings: `{EMBEDDING_CONFIG.model_name.split('/')[-1]}` (local)")
        st.write(f"Chunk: {RETRIEVAL_CONFIG.chunk_size} car. · solapamiento {RETRIEVAL_CONFIG.chunk_overlap} · top_k {RETRIEVAL_CONFIG.top_k}")

st.title(f"📊 Tutor de {SUBJECT_CONFIG.name}")
st.caption("Pregúntame sobre variables y escalas, intervalos de confianza, pruebas de hipótesis, t/ANOVA, chi-cuadrado, "
           "correlación y regresión. Puedes hacer preguntas de seguimiento.")

if MODEL_CONFIG.provider == "gemini" and not get_api_key():
    st.error("Falta la API key de Gemini. Defínela como `GEMINI_API_KEY` en los *Secrets* de la plataforma "
             "(o como variable de entorno / archivo `.env` si ejecutas en local).")
    st.stop()

pipeline = load_pipeline()

# ---------------------------------------------------------------- historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(message)

# ---------------------------------------------------------------- nuevo turno
question = st.chat_input("Escribe tu pregunta…") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["raw"] if m["role"] == "assistant" else m["content"]}
               for m in st.session_state.messages]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando en el material y preparando la respuesta…"):
                result = pipeline.ask(question, history)
        except LLMError as error:
            st.error(f"No pude generar la respuesta: {error}")
            st.stop()
        cited_ranks = {c.rank for c in result.cited}
        assistant_message = {
            "role": "assistant", "content": result.answer, "raw": result.raw,
            "metadata": result.metadata, "abstained": result.abstained,
            "standalone": result.standalone_question,
            "sources": [{"rank": c.rank, "source": c.source, "section": c.section, "page": c.page,
                         "score": c.score, "text": c.text, "cited": c.rank in cited_ranks}
                        for c in result.retrieved],
        }
        render_assistant(assistant_message)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(assistant_message)
