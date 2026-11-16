"""
llm.py
------
Clientes del modelo de lenguaje (Gemini y un simulador para pruebas).
Las claves se leen de variables de entorno / st.secrets; nunca del repositorio.
Cada cliente lleva un contador `usage` (llamadas y tokens) para comparar costos.

Privacidad: al modelo solo se le envían pregunta, historial reciente y
fragmentos/resultados acotados; nunca el corpus completo. En la capa gratuita
de Gemini, Google puede usar el contenido enviado para mejorar sus productos.
"""
import os
import re
import time
from typing import Optional

from config import MODEL_CONFIG


class LLMError(RuntimeError):
    pass


class BaseLLM:
    model_name: str = "base"

    def __init__(self):
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "retries": 0, "wait_s": 0.0}

    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError


def get_secret(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    try:  # secrets de Streamlit Community Cloud
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


def get_api_key() -> Optional[str]:
    return get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")


def thinking_config():
    """GEMINI_THINKING_LEVEL (opcional: minimal|low|medium|high) reduce el tiempo de razonamiento del modelo.
    Si el modelo no lo admite, quítalo del entorno."""
    level = os.getenv("GEMINI_THINKING_LEVEL", "").strip().lower()
    if not level:
        return None
    from google.genai import types

    return types.ThinkingConfig(thinking_level=level)


class GeminiLLM(BaseLLM):
    RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE", "timed out", "Timeout", "timeout")

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None,
                 temperature: float = MODEL_CONFIG.temperature,
                 max_tokens: int = MODEL_CONFIG.max_tokens, max_retries: int = 6):
        super().__init__()
        api_key = api_key or get_api_key()
        if not api_key:
            raise LLMError("No se encontró GEMINI_API_KEY. Defínela como variable de entorno, "
                           "en un archivo .env local o en los secrets de la plataforma.")
        from google import genai

        from google.genai import types as genai_types

        timeout_ms = int(float(os.getenv("GEMINI_TIMEOUT_S", "90")) * 1000)   # una llamada colgada no debe bloquear minutos
        self._client = genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(timeout=timeout_ms))
        self.model_name = model_name or MODEL_CONFIG.model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _record(self, response) -> None:
        usage = getattr(response, "usage_metadata", None)
        self.usage["calls"] += 1
        if usage:
            self.usage["prompt_tokens"] += usage.prompt_token_count or 0
            self.usage["output_tokens"] += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)

    def _generate_content(self, contents, config):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(model=self.model_name, contents=contents, config=config)
            except Exception as error:  # límites de uso y errores transitorios
                last_error = error
                text = str(error)
                if "PerDay" in text or "per day" in text.lower():   # cuota diaria: reintentar no sirve de nada
                    raise LLMError("Se agotó la cuota DIARIA de Gemini para este modelo. Prueba con otro modelo en GEMINI_MODEL "
                                   "o espera al reinicio de la cuota.") from error
                if any(code in text for code in self.RETRYABLE) and attempt < self.max_retries - 1:
                    delay = min(2 ** attempt * 3, 60)
                    self.usage["retries"] += 1
                    self.usage["wait_s"] += delay
                    print(f"[gemini] intento {attempt + 1}/{self.max_retries} falló ({text[:110]!r}); reintento en {delay}s", flush=True)
                    time.sleep(delay)
                    continue
                raise LLMError(f"Error al llamar a Gemini: {error}") from error
            self._record(response)
            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content or not candidate.content.parts:
                raise LLMError("El modelo devolvió una respuesta vacía (posible bloqueo de seguridad).")
            return response
        raise LLMError(f"Gemini no respondió tras {self.max_retries} intentos: {last_error}")

    def generate(self, system: str, user: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system, temperature=self.temperature,
                                             max_output_tokens=self.max_tokens,
                                             thinking_config=thinking_config())
        text = self._generate_content(user, config).text
        if not text:
            raise LLMError("El modelo devolvió una respuesta vacía.")
        return text


class MockLLM(BaseLLM):
    """Simula al modelo del pipeline fijo (Avance 2): pruebas sin red ni clave."""

    model_name = "mock"
    OUT_OF_SCOPE = re.compile(r"laplace|arima|bayes|kolmogorov", re.IGNORECASE)

    def generate(self, system: str, user: str) -> str:
        self.usage["calls"] += 1
        if "REFORMULAR" in system:
            match = re.search(r"<pregunta_actual>\s*(.*?)\s*</pregunta_actual>", user, re.DOTALL)
            return match.group(1) if match else user
        question = re.findall(r"<pregunta>\s*(.*?)\s*</pregunta>", user, re.DOTALL)[-1]
        fragments = re.findall(r'<fragmento id="(\d+)"[^>]*>\s*"""(.*?)"""', user.split("<historial>")[-1], re.DOTALL)
        if self.OUT_OF_SCOPE.search(question) or not fragments:
            return ("## No puedo responder con el material disponible\n\n"
                    "El material entregado no cubre ese tema.\n\n"
                    '```json\n{"tema": "fuera del material", "fuentes": [], "nivel_confianza": "bajo"}\n```')
        first_id, first_text = fragments[0]
        sentence = first_text.strip().split(". ")[0].strip()
        return (f"## Respuesta (simulada)\n\n{sentence}. [{first_id}]\n\n"
                f'```json\n{{"tema": "simulado", "fuentes": [{first_id}], "nivel_confianza": "medio"}}\n```')


def get_llm() -> BaseLLM:
    if MODEL_CONFIG.provider == "mock":
        return MockLLM()
    return GeminiLLM()
