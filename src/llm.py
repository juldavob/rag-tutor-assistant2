"""
llm.py
------
Cliente del modelo de lenguaje.

* GeminiLLM: usa el SDK oficial google-genai. La API key se lee de la
  variable de entorno GEMINI_API_KEY (o de st.secrets en Streamlit) y NUNCA
  se escribe en el código ni en el repositorio.
* MockLLM: respuestas deterministas para probar el pipeline sin red ni clave.

Privacidad: al modelo solo se le envían la pregunta, el historial reciente y
los fragmentos recuperados (top_k); nunca el corpus completo. Nota: en la
capa gratuita de la API de Gemini, Google puede usar el contenido enviado
para mejorar sus productos; para datos sensibles usar la capa de pago o un
modelo local.
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

    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError


def get_api_key() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:  # secrets de Streamlit Community Cloud
        import streamlit as st

        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


class GeminiLLM(BaseLLM):
    RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE")

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None,
                 temperature: float = MODEL_CONFIG.temperature,
                 max_tokens: int = MODEL_CONFIG.max_tokens, max_retries: int = 6):
        api_key = api_key or get_api_key()
        if not api_key:
            raise LLMError(
                "No se encontró GEMINI_API_KEY. Defínela como variable de entorno, "
                "en un archivo .env local o en los secrets de la plataforma."
            )
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name or MODEL_CONFIG.model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(self, system: str, user: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name, contents=user, config=config
                )
                text = response.text
                if not text:
                    raise LLMError("El modelo devolvió una respuesta vacía (posible bloqueo de seguridad).")
                return text
            except LLMError:
                raise
            except Exception as error:  # límites de uso y errores transitorios
                last_error = error
                if any(code in str(error) for code in self.RETRYABLE) and attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt * 3, 60))
                    continue
                raise LLMError(f"Error al llamar a Gemini: {error}") from error
        raise LLMError(f"Gemini no respondió tras {self.max_retries} intentos: {last_error}")


class MockLLM(BaseLLM):
    """Simula al modelo: útil para probar el flujo completo sin API key."""

    model_name = "mock"
    OUT_OF_SCOPE = re.compile(r"laplace|arima|bayes|kolmogorov", re.IGNORECASE)

    def generate(self, system: str, user: str) -> str:
        if "REFORMULAR" in system:  # paso de reformulación de preguntas de seguimiento
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
