"""
agent_llm.py
------------
Clientes con llamadas a herramientas (tool calling) para el agente.

Interfaz que usa el bucle (agent.py):
    start(history, question)           -> contents (estado de la conversación del turno)
    step(system, contents, tools)      -> Step(text, calls, raw, tokens)   (1 llamada al LLM)
    append_results(contents, step, [(call, result_dict), ...])
    append_user(contents, text)

GeminiAgentLLM: SDK google-genai con la ejecución automática de funciones DESACTIVADA
(el bucle lo controla el programa, no el SDK, para aplicar límites y registrar cada llamada).
MockAgentLLM: heurístico y determinista, solo para pruebas sin red.
"""
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from config import MODEL_CONFIG
from llm import BaseLLM, GeminiLLM, thinking_config


@dataclass
class ToolCall:
    name: str
    args: dict
    id: Optional[str] = None


@dataclass
class Step:
    text: str = ""
    calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None
    prompt_tokens: int = 0
    output_tokens: int = 0


class GeminiAgentLLM(GeminiLLM):
    def start(self, history, question):
        from google.genai import types

        contents = []
        for message in history or []:
            role = "user" if message["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=message["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
        return contents

    def step(self, system, contents, tools) -> Step:
        from google.genai import types

        declarations = [types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                        for t in tools]
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature, max_output_tokens=self.max_tokens,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            thinking_config=thinking_config(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        before = dict(self.usage)
        response = self._generate_content(contents, config)
        content = response.candidates[0].content
        calls = [ToolCall(p.function_call.name, dict(p.function_call.args or {}), getattr(p.function_call, "id", None))
                 for p in content.parts if getattr(p, "function_call", None)]
        text = "".join(p.text for p in content.parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        return Step(text=text, calls=calls, raw=content,
                    prompt_tokens=self.usage["prompt_tokens"] - before["prompt_tokens"],
                    output_tokens=self.usage["output_tokens"] - before["output_tokens"])

    def append_results(self, contents, step, results) -> None:
        from google.genai import types

        contents.append(step.raw)  # conserva el turno del modelo tal cual (incluidas las firmas de razonamiento)
        contents.append(types.Content(role="user", parts=[
            types.Part.from_function_response(name=call.name, response={"result": result}) for call, result in results]))

    def append_user(self, contents, text) -> None:
        from google.genai import types

        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))


class MockAgentLLM(BaseLLM):
    """Simula un LLM con tools: decide con reglas simples. Solo para pruebas."""

    model_name = "mock-agent"
    GREETING = re.compile(r"^\s*(hola|buenas|buenos días|buen día|gracias|hey|qué tal)\b", re.IGNORECASE)
    OUT_OF_SCOPE = re.compile(r"laplace|arima|bayes|fútbol|futbol|francia", re.IGNORECASE)
    NUM = re.compile(r"\d+(?:[.,]\d+)?")

    def start(self, history, question):
        return [{"kind": "user", "text": question}]

    def append_results(self, contents, step, results):
        contents.append({"kind": "results", "results": results})

    def append_user(self, contents, text):
        contents.append({"kind": "note", "text": text})

    @staticmethod
    def _json(tipo, tema, fuentes, nivel="alto"):
        import json
        return "\n\n```json\n" + json.dumps({"tipo": tipo, "tema": tema, "fuentes": fuentes, "nivel_confianza": nivel}, ensure_ascii=False) + "\n```"

    def _count(self, text):
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += 100
        self.usage["output_tokens"] += 20
        return Step(text=text, prompt_tokens=100, output_tokens=20)

    def step(self, system, contents, tools) -> Step:
        question = contents[0]["text"]
        q = question.lower()
        results = [r for c in contents if c["kind"] == "results" for r in c["results"]]
        forced = any(c["kind"] == "note" for c in contents)
        nums = [float(n.replace(",", ".")) for n in self.NUM.findall(question)]
        step = self._count("")
        if not results and not forced:
            if self.GREETING.match(question):
                step.text = "¡Hola! Soy tu tutor de Estadística Inferencial. ¿Qué tema repasamos?" + self._json("conversacion", "saludo", [])
            elif "ignora" in q and ("reglas" in q or "instrucciones" in q):
                step.text = ("No puedo mostrar documentos completos ni saltarme mis reglas. Puedo resumir el tema o responder preguntas concretas."
                             + self._json("rechazo", "instrucción maliciosa", []))
            elif "tamaño de muestra" in q and len(nums) >= 3:
                step.calls = [ToolCall("calcular_tamano_muestra", {"sigma": nums[0], "error_maximo": nums[1], "nivel_confianza": nums[2] / 100})]
            elif "intervalo de confianza" in q and len(nums) >= 3:
                step.calls = [ToolCall("calcular_intervalo_confianza_media", {"n": nums[0], "media_muestral": nums[1], "desviacion_muestral": nums[2]})]
            elif "rechazo" in q and "h0" in q:
                v = [float(n.replace(",", ".")) for n in self.NUM.findall(re.sub(r"h0", "", question, flags=re.IGNORECASE))]
                step.calls = [ToolCall("calcular_prueba_t_una_muestra", {"n": v[0], "media_muestral": v[1], "desviacion_muestral": v[2], "mu0": v[3], "alfa": v[4]})]
            elif ", y " in q:
                first, second = question.split(", y ", 1)
                step.calls = [ToolCall("buscar_en_base", {"consulta": first}), ToolCall("buscar_en_base", {"consulta": second})]
            else:
                step.calls = [ToolCall("buscar_en_base", {"consulta": question})]
            return step
        web_tool = any(t.name == "buscar_en_web" for t in tools)
        names = [call.name for call, _ in results]
        if (not forced and web_tool and "buscar_en_web" not in names and re.search(r"scipy|versión|actual", q)):
            step.calls = [ToolCall("buscar_en_web", {"consulta": "scipy ttest_ind"})]
            return step
        web = [r for _, r in results if r.get("resultados") and r["resultados"][0].get("origen") == "web"]
        calc = [r for call, r in results if call.name.startswith("calcular_") and "error" not in r]
        base = [i for _, r in results if r.get("resultados") for i in r["resultados"] if i.get("origen") == "base"]
        if calc:
            step.text = "Resultado del cálculo: " + calc[0]["pasos"] + self._json("tarea", "cálculo", [])
        elif web:
            item = web[0]["resultados"][0]
            step.text = f"Según la documentación oficial: {item['extracto']} [web: {item['url']}]" + self._json("web", "software", [item["ref"]], "medio")
        elif self.OUT_OF_SCOPE.search(question) or not base:
            step.text = "El material entregado no cubre ese tema." + self._json("sin_informacion", "fuera del material", [], "bajo")
        else:
            item = base[0]
            sentence = item["fragmento"].split(". ")[0].strip()
            step.text = f"{sentence}. (Fuente: {item['documento']}, {item['seccion']})" + self._json("base", item["seccion"], [item["ref"]])
        return step


def get_agent_llm():
    if MODEL_CONFIG.provider == "mock":
        return MockAgentLLM()
    return GeminiAgentLLM()
