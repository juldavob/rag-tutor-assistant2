"""
config.py
---------
Configuración central del Tutor Académico (Avance 1 + Avance 2).

Todo parámetro del pipeline que se quiera ajustar (tamaño de chunk, top_k,
modelo, etc.) vive aquí o se puede sobrescribir con variables de entorno,
de modo que la evaluación con Ragas pueda comparar configuraciones sin
tocar el resto del código.

Las credenciales NUNCA se escriben aquí: se leen de variables de entorno
(GEMINI_API_KEY), de un archivo .env local (ignorado por git) o de los
secrets de la plataforma de despliegue.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Carga un archivo .env sencillo (KEY=VALUE) sin dependencias externas.
    Solo define variables que no existan ya en el entorno."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# Asignatura
# ---------------------------------------------------------------------------

@dataclass
class SubjectConfig:
    name: str
    description: str
    allowed_topics: List[str] = field(default_factory=list)


SUBJECT_CONFIG = SubjectConfig(
    name="Estadística Inferencial",
    description=(
        "Curso de ingeniería sobre inferencia estadística: tipos de "
        "variables, escalas de medición, estimación, pruebas de hipótesis "
        "y selección de pruebas estadísticas según el tipo de dato."
    ),
    allowed_topics=[
        "tipos de variables (cualitativas/cuantitativas)",
        "escalas de medición (nominal, ordinal, intervalo, razón)",
        "estimación puntual e intervalos de confianza",
        "pruebas de hipótesis",
        "comparación de medias (t, ANOVA) y pruebas no paramétricas",
        "pruebas para datos categóricos (chi-cuadrado)",
        "correlación y regresión lineal simple",
        "selección de pruebas estadísticas",
    ],
)


# ---------------------------------------------------------------------------
# Modelo de lenguaje (generación)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # "gemini" (producción) | "mock" (pruebas sin red)
    provider: str = os.getenv("LLM_BACKEND", "gemini")
    # Alias "-latest" por defecto porque los modelos fijos de Gemini se
    # retiran con frecuencia. Para evaluaciones reproducibles se recomienda
    # fijar el nombre exacto con la variable de entorno GEMINI_MODEL.
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    temperature: float = 0.2     # baja: preferimos precisión sobre creatividad
    max_tokens: int = 2048       # incluye margen por si el modelo "piensa"


# ---------------------------------------------------------------------------
# Embeddings y recuperación
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    # "fastembed" (modelo local, ONNX) | "hash" (solo pruebas offline)
    backend: str = os.getenv("EMBEDDING_BACKEND", "fastembed")
    model_name: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )


@dataclass
class RetrievalConfig:
    # Valores por defecto = configuración "mejorada" (ver README y eval/).
    # La configuración baseline de la evaluación es 300 / 30 / 3.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))     # caracteres
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "4"))


@dataclass
class ConversationConfig:
    history_turns: int = 4           # turnos previos que se envían al modelo
    history_answer_chars: int = 600  # recorte de respuestas previas en el historial


SUBJECT = SUBJECT_CONFIG
MODEL_CONFIG = ModelConfig()
EMBEDDING_CONFIG = EmbeddingConfig()
RETRIEVAL_CONFIG = RetrievalConfig()
CONVERSATION_CONFIG = ConversationConfig()

KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
CHROMA_DIR = ROOT / "chroma_db"
EVAL_DIR = ROOT / "eval"
