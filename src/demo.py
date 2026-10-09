"""
demo.py
-------
DEMO DEL AVANCE 1 (solo ensamblado de prompt, sin modelo ni índice).
Para consultas reales sobre el RAG completo usa: python scripts/ask.py "tu pregunta"

Ejecuta el flujo de ensamblado de prompt de punta a punta para una
pregunta de ejemplo, usando contexto "recuperado" simulado (en el
siguiente avance este contexto vendrá del vector store real).

Uso:
    python demo.py
"""

import json
from system_prompt import build_prompt
from config import MODEL_CONFIG


# Contexto simulado: en producción esto vendría de la búsqueda por
# similitud sobre los embeddings de la base de conocimiento.
SIMULATED_RETRIEVED_CONTEXT = [
    """Para comparar las medias de dos grupos independientes se usa la
    prueba t de Student, siempre que los datos sean aproximadamente
    normales. Cuando se comparan tres o más grupos, se usa ANOVA de un
    factor en lugar de repetir la prueba t varias veces, porque repetir
    la prueba t infla el error tipo I (más probabilidad de un falso
    positivo).""",
    """El supuesto de normalidad puede verificarse con la prueba de
    Shapiro-Wilk. Si no se cumple, se recomienda usar la alternativa no
    paramétrica: U de Mann-Whitney (dos grupos) o Kruskal-Wallis (tres o
    más grupos).""",
]

QUESTION = "¿Cuándo debo usar ANOVA en vez de una prueba t?"


def call_local_model(prompt: str) -> str:
    """
    Punto de integración con un modelo servido localmente (ej. Ollama).
    Se deja implementado pero comentado para no requerir infraestructura
    externa al correr este avance; solo describe cómo se conectaría.

        import requests
        response = requests.post(
            f"{MODEL_CONFIG.base_url}/api/generate",
            json={"model": MODEL_CONFIG.model_name, "prompt": prompt, "stream": False},
        )
        return response.json()["response"]
    """
    raise NotImplementedError("Conectar aquí con Ollama u otro runtime local.")


def simulated_model_response() -> str:
    """
    Respuesta simulada que ilustra el formato de salida esperado
    (Markdown + bloque JSON), tal como debería producirlo el modelo
    real siguiendo el system prompt y los few-shot examples.
    """
    markdown = (
        "## ANOVA vs. prueba t\n\n"
        "- Usa la **prueba t** cuando comparas las medias de **exactamente "
        "dos grupos** independientes [1].\n"
        "- Usa **ANOVA de un factor** cuando comparas las medias de "
        "**tres o más grupos**. No se recomienda repetir la prueba t por "
        "parejas de grupos, porque cada comparación adicional aumenta la "
        "probabilidad de un falso positivo (inflación del error tipo I) [1].\n\n"
        "Antes de aplicar cualquiera de las dos, verifica el supuesto de "
        "normalidad (ej. con Shapiro-Wilk). Si no se cumple, la alternativa "
        "no paramétrica es Mann-Whitney (2 grupos) o Kruskal-Wallis (3+ "
        "grupos) [2]."
    )
    metadata = {
        "tema": "selección de pruebas estadísticas",
        "fuentes": [1, 2],
        "nivel_confianza": "alto",
    }
    return f"{markdown}\n\n```json\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n```"


def main():
    prompt = build_prompt(
        context_chunks=SIMULATED_RETRIEVED_CONTEXT,
        question=QUESTION,
        history=[],
    )

    print("=" * 80)
    print("PROMPT ENSAMBLADO (lo que recibiría el modelo):")
    print("=" * 80)
    print(prompt)

    print("=" * 80)
    print("RESPUESTA DEL MODELO:")
    print("=" * 80)
    try:
        output = call_local_model(prompt)
    except NotImplementedError:
        output = simulated_model_response()
        print("(usando respuesta simulada porque no hay un modelo local "
              "conectado en este entorno — ver call_local_model)\n")
    print(output)


if __name__ == "__main__":
    main()
