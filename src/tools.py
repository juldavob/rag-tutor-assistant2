"""
tools.py
--------
Herramientas (tools) del agente. Los LÍMITES están en el código (config.AGENT_CONFIG),
no en el prompt: el LLM no puede saltárselos aunque se lo pidan.

  buscar_en_base      recuperación del Avance 2 convertida en tool (local)
  calcular_*          tareas del dominio (cálculos exactos que el LLM no debe hacer de memoria)
  buscar_en_web       búsqueda web controlada (Tavily), política «primero la base»

Cada tool es un objeto `Tool` con `.invoke(args)`, así se puede probar SIN el LLM.
"""
import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from config import AGENT_CONFIG as C
from llm import get_secret
from system_prompt import sanitize


@dataclass
class Session:
    """Estado de UNA pregunta: contadores de límites y evidencia devuelta por las tools."""
    kb_calls: int = 0
    web_calls: int = 0
    fragments_returned: int = 0
    b_counter: int = 0
    w_counter: int = 0
    base_results: List[dict] = field(default_factory=list)
    web_results: List[dict] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)   # textos devueltos por las tools (para Ragas)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON Schema de los argumentos que el LLM puede enviar
    func: Callable

    def invoke(self, args: Optional[dict] = None, session: Optional[Session] = None) -> dict:
        try:
            return self.func(session=session or Session(), **(args or {}))
        except TypeError as error:
            return {"error": f"Argumentos inválidos para {self.name}: {error}"}
        except (ValueError, ArithmeticError) as error:
            return {"error": str(error)}


def _cut(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _int(value, name):
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{name} debe ser un entero.")
    return int(number)


# ---------------------------------------------------------------- base de conocimientos
def make_buscar_en_base(retriever) -> Tool:
    def buscar_en_base(consulta: str, session: Session) -> dict:
        """Busca en la base de conocimientos del curso (apuntes de Estadística Inferencial).
        Úsala para cualquier pregunta conceptual de la asignatura: definiciones, supuestos, criterios
        de selección de pruebas, interpretación de resultados. Puedes llamarla varias veces, una por
        tema, con consultas cortas y autónomas. No la uses para saludos ni para cálculos numéricos.
        Devuelve como máximo unos pocos fragmentos, cada uno con documento, sección y página."""
        consulta = (consulta or "").strip()
        if not consulta:
            return {"error": "La consulta está vacía.", "resultados": []}
        remaining = C.kb_max_fragments_per_question - session.fragments_returned
        if remaining <= 0:
            return {"error": "Se alcanzó el máximo de fragmentos permitidos para esta pregunta.", "resultados": []}
        chunks = retriever.retrieve(consulta[:C.query_max_chars], top_k=min(C.kb_max_fragments, remaining))
        session.kb_calls += 1
        results = []
        for chunk in chunks:
            session.b_counter += 1
            text = _cut(sanitize(chunk.text), C.kb_max_chars)
            item = {"ref": f"B{session.b_counter}", "origen": "base", "documento": chunk.source,
                    "seccion": chunk.section, "pagina": chunk.page, "similitud": chunk.score, "fragmento": text}
            results.append(item)
            session.base_results.append(item)
            session.contexts.append(text)
        session.fragments_returned += len(results)
        return {"resultados": results,
                "nota": "Si ningún fragmento responde la pregunta, la base no cubre el tema."}

    return Tool("buscar_en_base", inspect.getdoc(buscar_en_base),
                {"type": "object", "properties": {"consulta": {"type": "string", "description": "Consulta corta y autónoma sobre un solo tema."}},
                 "required": ["consulta"]}, buscar_en_base)


# ---------------------------------------------------------------- tareas del dominio
def _z(nivel: float) -> float:
    from scipy import stats
    if not 0.5 <= nivel < 1:
        raise ValueError("nivel_confianza debe estar entre 0,5 y 1 (por ejemplo 0,95).")
    return float(stats.norm.ppf(1 - (1 - nivel) / 2))


def make_calcular_tamano_muestra() -> Tool:
    def calcular_tamano_muestra(sigma: float, error_maximo: float, nivel_confianza: float = 0.95, session=None) -> dict:
        """Calcula el tamaño de muestra necesario para estimar una MEDIA poblacional con un error máximo E,
        desviación estándar poblacional sigma y un nivel de confianza (n = (z·sigma/E)², redondeado hacia arriba).
        Úsala siempre que el usuario pida «qué tamaño de muestra necesito». No calcules esto de memoria."""
        sigma, error_maximo = float(sigma), float(error_maximo)
        if sigma <= 0 or error_maximo <= 0:
            raise ValueError("sigma y error_maximo deben ser positivos.")
        z = _z(float(nivel_confianza))
        n = (z * sigma / error_maximo) ** 2
        n_up = math.ceil(round(n, 9))
        return {"z": round(z, 4), "n_exacto": round(n, 4), "n_redondeado": n_up,
                "pasos": f"z = {z:.4f}; n = ({z:.4f} × {sigma:g} / {error_maximo:g})² = {n:.4f}; se redondea hacia arriba: {n_up}."}

    return Tool("calcular_tamano_muestra", inspect.getdoc(calcular_tamano_muestra),
                {"type": "object", "properties": {
                    "sigma": {"type": "number", "description": "Desviación estándar poblacional (>0)."},
                    "error_maximo": {"type": "number", "description": "Error máximo admisible E (>0), en las mismas unidades."},
                    "nivel_confianza": {"type": "number", "description": "Entre 0,5 y 1; por defecto 0,95."}},
                 "required": ["sigma", "error_maximo"]}, calcular_tamano_muestra)


def make_calcular_ic_media() -> Tool:
    def calcular_intervalo_confianza_media(media_muestral: float, desviacion_muestral: float, n: int,
                                           nivel_confianza: float = 0.95, session=None) -> dict:
        """Calcula el intervalo de confianza para una MEDIA con sigma desconocida (distribución t de Student,
        n-1 grados de libertad) a partir de la media muestral, la desviación estándar muestral s y el tamaño n.
        Úsala siempre que el usuario dé estos datos y pida un intervalo de confianza. No calcules de memoria."""
        from scipy import stats
        n = _int(n, "n")
        s, media, nivel = float(desviacion_muestral), float(media_muestral), float(nivel_confianza)
        if n < 2 or s <= 0:
            raise ValueError("n debe ser al menos 2 y la desviación estándar positiva.")
        if not 0.5 <= nivel < 1:
            raise ValueError("nivel_confianza debe estar entre 0,5 y 1.")
        t = float(stats.t.ppf(1 - (1 - nivel) / 2, n - 1))
        se = s / math.sqrt(n)
        margen = t * se
        return {"gl": n - 1, "t_critico": round(t, 4), "error_estandar": round(se, 4), "margen_error": round(margen, 4),
                "limite_inferior": round(media - margen, 4), "limite_superior": round(media + margen, 4),
                "pasos": f"gl = {n - 1}; t = {t:.4f}; EE = {s:g}/√{n} = {se:.4f}; margen = {margen:.4f}; "
                         f"IC = {media:g} ± {margen:.4f} = ({media - margen:.4f}; {media + margen:.4f})."}

    return Tool("calcular_intervalo_confianza_media", inspect.getdoc(calcular_intervalo_confianza_media),
                {"type": "object", "properties": {
                    "media_muestral": {"type": "number"}, "desviacion_muestral": {"type": "number", "description": "s (>0)."},
                    "n": {"type": "integer", "description": "Tamaño de muestra (>=2)."},
                    "nivel_confianza": {"type": "number", "description": "Entre 0,5 y 1; por defecto 0,95."}},
                 "required": ["media_muestral", "desviacion_muestral", "n"]}, calcular_intervalo_confianza_media)


def make_calcular_prueba_t() -> Tool:
    def calcular_prueba_t_una_muestra(media_muestral: float, desviacion_muestral: float, n: int, mu0: float,
                                      alfa: float = 0.05, alternativa: str = "dos_colas", session=None) -> dict:
        """Realiza una prueba t de UNA muestra de H0: mu = mu0 a partir de la media muestral, la desviación
        estándar muestral s y n. Devuelve el estadístico t, los grados de libertad, el valor p y la decisión.
        alternativa: 'dos_colas', 'mayor' o 'menor'. Úsala siempre que el usuario dé estos datos y pida
        contrastar una media; no calcules el valor p de memoria."""
        from scipy import stats
        n = _int(n, "n")
        s, media, mu0, alfa = float(desviacion_muestral), float(media_muestral), float(mu0), float(alfa)
        if n < 2 or s <= 0 or not 0 < alfa < 1:
            raise ValueError("n debe ser al menos 2, s positiva y alfa entre 0 y 1.")
        t = (media - mu0) / (s / math.sqrt(n))
        if alternativa == "dos_colas":
            p = 2 * float(stats.t.sf(abs(t), n - 1))
        elif alternativa == "mayor":
            p = float(stats.t.sf(t, n - 1))
        elif alternativa == "menor":
            p = float(stats.t.cdf(t, n - 1))
        else:
            raise ValueError("alternativa debe ser 'dos_colas', 'mayor' o 'menor'.")
        decision = "Se rechaza H0" if p <= alfa else "No se rechaza H0"
        return {"t": round(t, 4), "gl": n - 1, "valor_p": round(p, 4), "alfa": alfa, "alternativa": alternativa,
                "decision": decision,
                "pasos": f"t = ({media:g} − {mu0:g}) / ({s:g}/√{n}) = {t:.4f}; gl = {n - 1}; p = {p:.4f}; "
                         f"{decision} con α = {alfa:g}."}

    return Tool("calcular_prueba_t_una_muestra", inspect.getdoc(calcular_prueba_t_una_muestra),
                {"type": "object", "properties": {
                    "media_muestral": {"type": "number"}, "desviacion_muestral": {"type": "number"},
                    "n": {"type": "integer"}, "mu0": {"type": "number", "description": "Valor de H0."},
                    "alfa": {"type": "number", "description": "Nivel de significancia; por defecto 0,05."},
                    "alternativa": {"type": "string", "enum": ["dos_colas", "mayor", "menor"]}},
                 "required": ["media_muestral", "desviacion_muestral", "n", "mu0"]}, calcular_prueba_t_una_muestra)


# ---------------------------------------------------------------- búsqueda web controlada
def _default_post(url, payload, headers):
    import requests
    response = requests.post(url, json=payload, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()


def _allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in C.web_allowed_domains)


def make_buscar_en_web(http_post: Optional[Callable] = None, api_key_getter: Optional[Callable] = None) -> Tool:
    post = http_post or _default_post
    get_key = api_key_getter or (lambda: get_secret("TAVILY_API_KEY"))
    sites = ", ".join(C.web_allowed_domains)

    description = (
        "Busca en la web, SOLO en sitios oficiales de referencia (" + sites + "). Política «primero la base»: "
        "úsala únicamente si ya llamaste a buscar_en_base y la base NO responde la pregunta, y la pregunta trata de "
        "información pública o actual de estadística o de su software (por ejemplo una función o versión reciente). "
        "Nunca la uses si la base responde, ni para saludos o cálculos. Escribe una consulta corta de búsqueda "
        "propia; no copies texto de la base. Devuelve como máximo 3 resultados con título, URL y extracto corto.")

    def run(consulta: str, session: Session) -> dict:
        key = get_key()
        if not key:
            return {"error": "La búsqueda web no está configurada (falta TAVILY_API_KEY).", "resultados": []}
        if session.kb_calls == 0:
            return {"error": "Política «primero la base»: llama antes a buscar_en_base.", "resultados": []}
        if session.web_calls >= C.web_max_calls:
            return {"error": "Se alcanzó el máximo de búsquedas web para esta pregunta.", "resultados": []}
        consulta = (consulta or "").strip()
        if not consulta:
            return {"error": "La consulta está vacía.", "resultados": []}
        if len(consulta) > C.web_query_max_chars:
            return {"error": f"La consulta web debe tener como máximo {C.web_query_max_chars} caracteres "
                             "(no envíes texto de la base al buscador).", "resultados": []}
        session.web_calls += 1
        payload = {"query": consulta, "max_results": C.web_max_results, "include_domains": list(C.web_allowed_domains),
                   "search_depth": "basic", "include_answer": False, "include_raw_content": False}
        try:
            data = post("https://api.tavily.com/search", payload, {"Authorization": f"Bearer {key}"})
        except Exception as error:  # no se filtra la clave en el mensaje
            return {"error": f"No se pudo consultar el buscador ({type(error).__name__}).", "resultados": []}
        results = []
        for raw in (data.get("results") or []):
            if len(results) >= C.web_max_results:
                break
            url = raw.get("url", "")
            if not _allowed(url):   # defensa adicional: se descarta lo que no sea de los sitios permitidos
                continue
            session.w_counter += 1
            item = {"ref": f"W{session.w_counter}", "origen": "web", "titulo": _cut(sanitize(raw.get("title", "")), 150),
                    "url": url, "extracto": _cut(sanitize(raw.get("content", "")), C.web_snippet_chars)}
            results.append(item)
            session.web_results.append(item)
            session.contexts.append(item["extracto"])
        out = {"resultados": results}
        if not results:
            out["nota"] = "Sin resultados en los sitios permitidos."
        return out

    return Tool("buscar_en_web", description,
                {"type": "object", "properties": {"consulta": {"type": "string", "description": "Consulta de búsqueda corta (máx. 200 caracteres)."}},
                 "required": ["consulta"]}, run)


def build_tools(retriever, web: bool = False, http_post=None, api_key_getter=None) -> Dict[str, Tool]:
    tools = [make_buscar_en_base(retriever), make_calcular_tamano_muestra(), make_calcular_ic_media(), make_calcular_prueba_t()]
    if web:
        tools.append(make_buscar_en_web(http_post, api_key_getter))
    return {t.name: t for t in tools}
