"""
agent.py
--------
Bucle del agente: el LLM decide si llama una tool; el PROGRAMA la ejecuta y le devuelve el resultado.

Límites (en código): máximo de iteraciones por pregunta, máximo de llamadas a tools, y tools con sus
propios topes (fragmentos, longitud, resultados web). Si se agotan las iteraciones, se hace una última
llamada SIN tools para obtener la mejor respuesta con lo reunido.
Cada pregunta queda registrada en logs/uso_tools.jsonl (tools, argumentos, llamadas al LLM, tokens).
"""
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent_llm import get_agent_llm
from agent_prompt import NO_INFO_MESSAGE, build_system_prompt, parse_agent_response
from config import AGENT_CONFIG as C, EMBEDDING_CONFIG, RETRIEVAL_CONFIG
from embeddings import get_embedder
from llm import get_secret
from retriever import Retriever
from system_prompt import sanitize
from tool_log import ToolLog
from tools import Session, Tool, build_tools
from vector_store import load_store_auto

KB_TOOLS = ("buscar_en_base", "buscar_en_web")


def _summarize(name: str, result: dict) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    if name == "buscar_en_base":
        docs = sorted({r["documento"] for r in result.get("resultados", [])})
        return f"{len(result.get('resultados', []))} fragmentos de {', '.join(docs) or '—'}"
    if name == "buscar_en_web":
        return f"{len(result.get('resultados', []))} resultados web"
    return result.get("pasos", json.dumps(result, ensure_ascii=False))[:200]


@dataclass
class AgentAnswer:
    question: str
    answer: str
    metadata: dict
    tool_calls: List[dict]
    llm_calls: int
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    iterations: int
    hit_limit: bool
    base_sources: List[dict] = field(default_factory=list)
    web_sources: List[dict] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    raw: str = ""

    @property
    def cited_refs(self) -> set:
        return set(self.metadata.get("fuentes", []))

    @property
    def tipo(self) -> str:
        return self.metadata.get("tipo", "")

    @property
    def abstained(self) -> bool:
        if self.metadata.get("parse_error"):
            return NO_INFO_MESSAGE.lower().rstrip(".") in self.answer.lower()
        return self.tipo == "sin_informacion"

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "metadata": self.metadata, "tool_calls": self.tool_calls,
                "llm_calls": self.llm_calls, "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
                "latency_s": self.latency_s, "iterations": self.iterations, "hit_limit": self.hit_limit,
                "base_sources": self.base_sources, "web_sources": self.web_sources, "contexts": self.contexts}


class Agent:
    def __init__(self, llm, tools: Dict[str, Tool], log: Optional[ToolLog] = None, system_prompt: Optional[str] = None):
        self.llm = llm
        self.tools = tools
        self.log = log
        self.system_prompt = system_prompt or build_system_prompt(web="buscar_en_web" in tools)

    def ask(self, question: str, history: Optional[List[Dict]] = None) -> AgentAnswer:
        started = time.time()
        session, tool_calls = Session(), []
        usage = {"calls": 0, "prompt": 0, "output": 0}
        contents = self.llm.start(history or [], question)

        def call_llm(tools):
            step = self.llm.step(self.system_prompt, contents, tools)
            usage["calls"] += 1
            usage["prompt"] += step.prompt_tokens
            usage["output"] += step.output_tokens
            return step

        final, hit_limit, iterations = None, False, 0
        for iterations in range(1, C.max_iterations + 1):
            step = call_llm(list(self.tools.values()))
            if not step.calls:
                final = step
                break
            results = []
            for call in step.calls:
                tool = self.tools.get(call.name)
                started_tool = time.time()
                if len(tool_calls) >= C.max_tool_calls:
                    result = {"error": "Se alcanzó el máximo de llamadas a herramientas para esta pregunta."}
                elif tool is None:
                    result = {"error": f"Herramienta desconocida: {call.name}"}
                else:
                    result = tool.invoke(call.args, session)
                    if call.name not in KB_TOOLS and "error" not in result:
                        session.contexts.append(f"[{call.name}] " + json.dumps(result, ensure_ascii=False))
                tool_calls.append({"name": call.name, "args": call.args, "summary": _summarize(call.name, result),
                                   "error": result.get("error"), "duration_s": round(time.time() - started_tool, 3)})
                results.append((call, result))
            self.llm.append_results(contents, step, results)
        if final is None:  # límite de iteraciones: última llamada sin tools
            hit_limit = True
            self.llm.append_user(contents, "Has alcanzado el límite de iteraciones. Responde ahora con lo que obtuviste, sin llamar herramientas.")
            final = call_llm([])

        markdown, metadata = parse_agent_response(final.text)
        cited = set(metadata.get("fuentes", []))
        base = [{**r, "cited": r["ref"] in cited} for r in session.base_results]
        web = [{**r, "cited": r["ref"] in cited} for r in session.web_results]
        answer = AgentAnswer(question=question, answer=markdown, metadata=metadata, tool_calls=tool_calls,
                             llm_calls=usage["calls"], prompt_tokens=usage["prompt"], output_tokens=usage["output"],
                             latency_s=round(time.time() - started, 2), iterations=iterations, hit_limit=hit_limit,
                             base_sources=base, web_sources=web, contexts=list(session.contexts), raw=final.text)
        if self.log:
            self.log.write({"question": question, "tools": [{k: t[k] for k in ("name", "args", "summary", "error", "duration_s")} for t in tool_calls],
                            "llm_calls": answer.llm_calls, "prompt_tokens": answer.prompt_tokens, "output_tokens": answer.output_tokens,
                            "iterations": iterations, "hit_limit": hit_limit, "tipo": answer.tipo, "abstained": answer.abstained,
                            "latency_s": answer.latency_s})
        return answer


def build_agent(web: Optional[bool] = None, llm=None, embedder=None, log: bool = True, chunk_size=None, overlap=None) -> Agent:
    """Arma el agente completo (índice incluido). La web solo se activa si hay TAVILY_API_KEY y no está desactivada."""
    embedder = embedder or get_embedder()
    store = load_store_auto(embedder, chunk_size or RETRIEVAL_CONFIG.chunk_size,
                                RETRIEVAL_CONFIG.chunk_overlap if overlap is None else overlap)
    retriever = Retriever(store, embedder, C.kb_max_fragments)
    web_on = (C.web_enabled if web is None else web) and bool(get_secret("TAVILY_API_KEY"))
    return Agent(llm or get_agent_llm(), build_tools(retriever, web=web_on), ToolLog() if log else None)
