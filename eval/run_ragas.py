"""
run_ragas.py
------------
Evalúa el pipeline RAG con Ragas (faithfulness, answer_relevancy,
context_precision, context_recall) sobre eval/eval_questions.json.

Pensado para cuotas pequeñas de la API (p. ej. capa gratuita):
  --rpm N      limita las llamadas a N por minuto (con 5 RPM usa --rpm 4)
  --budget N   máximo de llamadas a Gemini en ESTA ejecución; procesa las
               preguntas que quepan y se detiene. Vuelve a ejecutar el mismo
               comando otro día: reanuda donde quedó (no repite lo ya hecho).
  --strictness K  generaciones de answer_relevancy (3 = por defecto de Ragas;
               1 = menos llamadas, métrica algo menos estable)

Ejemplos:
    python eval/run_ragas.py --tag baseline --chunk-size 300 --overlap 30 --top-k 3 --rpm 4 --strictness 1
    python eval/run_ragas.py --tag baseline ... --budget 18      # un lote por día
    python eval/run_ragas.py --tag prueba --limit 4              # ensayo
    python eval/run_ragas.py --tag agente --agent --rpm 14 --strictness 1   # AGENTE (Avance 3)

Salidas en eval/results/: <tag>_answers.json, <tag>_per_question.csv, <tag>_summary.json
"""
import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import EVAL_DIR, MODEL_CONFIG, RETRIEVAL_CONFIG  # noqa: E402
from embeddings import get_embedder  # noqa: E402
from llm import LLMError, get_api_key  # noqa: E402
from rag_pipeline import build_pipeline  # noqa: E402

RESULTS_DIR = EVAL_DIR / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class Throttle:
    """Espacia las llamadas para no superar `rpm` peticiones por minuto."""
    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.time()
            if now < self.next:
                time.sleep(self.next - now)
            self.next = max(now, self.next) + self.interval


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--chunk-size", type=int, default=RETRIEVAL_CONFIG.chunk_size)
    p.add_argument("--overlap", type=int, default=RETRIEVAL_CONFIG.chunk_overlap)
    p.add_argument("--top-k", type=int, default=RETRIEVAL_CONFIG.top_k)
    p.add_argument("--questions", default=str(EVAL_DIR / "eval_questions.json"))
    p.add_argument("--limit", type=int, help="evaluar solo N preguntas repartidas por todo el set")
    p.add_argument("--rpm", type=float, default=0, help="máximo de llamadas por minuto (0 = sin límite)")
    p.add_argument("--budget", type=int, default=0, help="máximo de llamadas a Gemini en esta ejecución (0 = sin límite)")
    p.add_argument("--strictness", type=int, default=3, help="generaciones de answer_relevancy")
    p.add_argument("--batch", type=int, default=2, help="preguntas por llamada a Ragas (se guarda tras cada lote)")
    p.add_argument("--workers", type=int, default=2, help="llamadas simultáneas de Ragas (con --rpm se fuerza 1)")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--judge-model", default=MODEL_CONFIG.model_name)
    p.add_argument("--no-ragas", action="store_true", help="solo genera respuestas y diagnósticos")
    p.add_argument("--agent", action="store_true", help="evaluar el AGENTE (Avance 3) en vez del pipeline fijo")
    p.add_argument("--web", action="store_true", help="con --agent: permitir búsqueda web (por defecto desactivada para comparar solo con la base)")
    return p.parse_args()


def select_questions(questions, limit):
    if not limit or limit >= len(questions):
        return questions
    idx = sorted(set(np.linspace(0, len(questions) - 1, limit).astype(int).tolist()))
    return [questions[i] for i in idx]


def config_of(args):
    return {"chunk_size": args.chunk_size, "overlap": args.overlap, "top_k": args.top_k, "agent": args.agent, "web": bool(args.agent and args.web)}


def load_answers(args):
    path = RESULTS_DIR / f"{args.tag}_answers.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("config") == config_of(args):  # formato viejo (lista) se ignora
            return data["answers"]
        print("Aviso: el archivo de respuestas es de otra configuración; se ignora y se regenera.")
    return {}


def save_answers(args, answers):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.tag}_answers.json").write_text(
        json.dumps({"config": config_of(args), "answers": answers}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scores(args):
    path = RESULTS_DIR / f"{args.tag}_per_question.csv"
    if path.exists():
        return pd.read_csv(path).set_index("id").to_dict("index")
    return {}


def row_done(row, abstained):
    missing = sum(pd.isna(row.get(m)) for m in METRICS)
    return missing <= (1 if abstained else 0)   # una abstención puede dejar faithfulness vacía


def plan(args, questions, answers, scores):
    """Elige las preguntas que caben en el presupuesto de llamadas de esta ejecución."""
    judge_cost = 2 + args.strictness + (5 if args.agent else args.top_k) + 1
    gen_cost = 3 if args.agent else 1
    used, planned = 0, []
    for q in questions:
        need_gen = q["id"] not in answers
        need_judge = (not args.no_ragas) and not (q["id"] in scores and row_done(
            scores[q["id"]], answers.get(q["id"], {}).get("abstained", False)) and not need_gen)
        if not need_gen and not need_judge:
            continue
        cost = gen_cost * int(need_gen) + (judge_cost if need_judge else 0)
        if args.budget and used + cost > args.budget:
            break
        used += cost
        planned.append((q, need_gen, need_judge))
    return planned, used, judge_cost


def make_runner(args):
    if args.agent:
        from agent import build_agent
        runner = build_agent(web=args.web, log=False, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"Agente: web {'ACTIVA' if 'buscar_en_web' in runner.tools else 'desactivada'}")
        return runner
    return build_pipeline(args.chunk_size, args.overlap, args.top_k)


def generate(args, planned, answers, runner, throttle):
    for n, (q, need_gen, _) in enumerate(planned, start=1):
        if not need_gen:
            continue
        throttle.wait()
        try:
            if args.agent:
                r = runner.ask(q["question"], [])
                entry = {"response": r.answer, "retrieved_contexts": r.contexts,
                         "retrieved_sources": [s["documento"] for s in r.base_sources], "abstained": r.abstained,
                         "cited_ids": sorted(r.cited_refs), "confidence": r.metadata.get("nivel_confianza"),
                         "tipo": r.tipo, "tools": [t["name"] for t in r.tool_calls], "tool_calls": r.tool_calls,
                         "iterations": r.iterations, "hit_limit": r.hit_limit}
            else:
                r = runner.ask(q["question"], history=None, top_k=args.top_k)
                entry = {"response": r.answer, "retrieved_contexts": [c.text for c in r.retrieved],
                         "retrieved_sources": [c.source for c in r.retrieved], "abstained": r.abstained,
                         "cited_ids": r.metadata.get("fuentes", []), "confidence": r.metadata.get("nivel_confianza")}
        except LLMError as error:
            print(f"Se detiene la generación: {error}\nVuelve a ejecutar el mismo comando cuando haya cuota (reanuda).")
            return False
        usage_calls = getattr(r, "llm_calls", 0)
        answers[q["id"]] = {"id": q["id"], "question": q["question"], **entry, "latency_s": r.latency_s,
                            "llm_calls": usage_calls, "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens}
        save_answers(args, answers)
        print(f"  respuesta {q['id']} {'(abstención)' if entry['abstained'] else ''} {r.latency_s}s, {usage_calls} llamadas LLM")
        if args.rpm and args.agent and usage_calls > 1:
            time.sleep(60.0 / args.rpm * (usage_calls - 1))
    return True


def cost_summary(answers):
    rows = [a for a in answers.values() if "llm_calls" in a]
    if not rows:
        return None
    avg = lambda key: round(float(np.mean([a.get(key, 0) or 0 for a in rows])), 3)  # noqa: E731
    return {"n": len(rows), "avg_llm_calls": avg("llm_calls"), "avg_prompt_tokens": avg("prompt_tokens"),
            "avg_output_tokens": avg("output_tokens"), "avg_latency_s": avg("latency_s"),
            "avg_tool_calls": round(float(np.mean([len(a.get("tools", [])) for a in rows])), 3)}


def make_judge(args):
    from langchain_core.embeddings import Embeddings
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    api_key = get_api_key()
    if not api_key:
        sys.exit("Falta GEMINI_API_KEY (variable de entorno o archivo .env) para el modelo juez de Ragas.")

    class LocalEmbeddings(Embeddings):
        def __init__(self):
            self._e = get_embedder()
        def embed_documents(self, texts):
            return self._e.embed_documents(texts)
        def embed_query(self, text):
            return self._e.embed_query(text)

    kwargs = {"model": args.judge_model, "google_api_key": api_key, "temperature": 0}
    if args.rpm:
        kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=args.rpm / 60, check_every_n_seconds=0.2, max_bucket_size=1)
        kwargs["max_retries"] = 2
    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(**kwargs), bypass_n=True)  # Gemini no admite n>1
    answer_relevancy.strictness = args.strictness
    run_config = RunConfig(timeout=args.timeout, max_retries=4 if args.rpm else 8, max_wait=65 if args.rpm else 90,
                           max_workers=1 if args.rpm else args.workers)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    return judge, LangchainEmbeddingsWrapper(LocalEmbeddings()), metrics, run_config


def judge_batches(args, planned, answers, scores):
    from ragas import EvaluationDataset, evaluate

    todo = [q for q, _, need_judge in planned if need_judge and q["id"] in answers]
    if not todo:
        return True
    judge, embeddings, metrics, run_config = make_judge(args)
    for start in range(0, len(todo), args.batch):
        batch = todo[start:start + args.batch]
        dataset = EvaluationDataset.from_list([{
            "user_input": q["question"], "retrieved_contexts": answers[q["id"]]["retrieved_contexts"],
            "response": answers[q["id"]]["response"], "reference": q["ground_truth"]} for q in batch])
        print(f"Evaluando con Ragas: {[q['id'] for q in batch]} (juez {args.judge_model})")
        frame = evaluate(dataset, metrics=metrics, llm=judge, embeddings=embeddings, run_config=run_config,
                         raise_exceptions=False, show_progress=False).to_pandas().reset_index(drop=True)
        ok = True
        for q, (_, row) in zip(batch, frame.iterrows()):
            scores[q["id"]] = {"type": q["type"], "in_corpus": q["in_corpus"],
                               "abstained": answers[q["id"]]["abstained"], **{m: row.get(m) for m in METRICS}}
            ok &= row_done(row, answers[q["id"]]["abstained"])
        save_scores(args, scores)
        if not ok:
            print("Hay métricas sin calcular en este lote (probablemente cuota agotada). Se guarda el progreso y se detiene.\n"
                  "Vuelve a ejecutar el mismo comando cuando haya cuota: solo repetirá lo pendiente.")
            return False
    return True


def save_scores(args, scores):
    frame = pd.DataFrame.from_dict(scores, orient="index").rename_axis("id").reset_index()
    frame.to_csv(RESULTS_DIR / f"{args.tag}_per_question.csv", index=False, encoding="utf-8")


def diagnostics(questions, answers):
    hits = in_abs = out_abs = n_in = n_out = 0
    for q in questions:
        a = answers.get(q["id"])
        if not a:
            continue
        if q["in_corpus"]:
            n_in += 1
            hits += int(q["expected_source"] in a["retrieved_sources"])
            in_abs += int(a["abstained"])
        else:
            n_out += 1
            out_abs += int(a["abstained"])
    return {"retrieval_hit_rate": round(hits / n_in, 3) if n_in else None,
            "abstention_rate_out_of_corpus": round(out_abs / n_out, 3) if n_out else None,
            "false_abstention_rate_in_corpus": round(in_abs / n_in, 3) if n_in else None,
            "n_in_corpus": n_in, "n_out_of_corpus": n_out}


def summarize(frame, mask):
    out = {}
    for m in METRICS:
        values = frame.loc[mask, m].dropna() if m in frame else []
        out[m] = {"mean": round(float(np.mean(values)), 4) if len(values) else None, "n_valid": int(len(values))}
    return out


def main():
    args = parse_args()
    questions = select_questions(json.loads(Path(args.questions).read_text(encoding="utf-8")), args.limit)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    answers, scores = load_answers(args), load_scores(args)

    planned, used, judge_cost = plan(args, questions, answers, scores)
    pending = sum(1 for q in questions if not (q["id"] in answers and (args.no_ragas or (q["id"] in scores and row_done(
        scores[q["id"]], answers[q["id"]]["abstained"])))))
    print(f"Preguntas: {len(questions)} | pendientes: {pending} | esta ejecución: {len(planned)} "
          f"(~{used} llamadas; ~{judge_cost} por pregunta en el juez + 1 de generación)")
    if pending and not planned:
        sys.exit("El presupuesto (--budget) no alcanza ni para una pregunta. Súbelo o usa otro modelo.")

    throttle = Throttle(args.rpm)
    completed = True
    if any(need_gen for _, need_gen, _ in planned):
        runner = make_runner(args)
        print(f"Generando respuestas ({'AGENTE' if args.agent else 'pipeline fijo'}) con {runner.llm.model_name} "
              f"(chunk={args.chunk_size}, overlap={args.overlap}, top_k={args.top_k})")
        completed = generate(args, planned, answers, runner, throttle)
    if completed and not args.no_ragas:
        completed = judge_batches(args, planned, answers, scores)

    frame = pd.DataFrame.from_dict(scores, orient="index").rename_axis("id").reset_index() if scores else pd.DataFrame()
    done_ids = {q["id"] for q in questions if q["id"] in answers and (args.no_ragas or (q["id"] in scores and row_done(
        scores[q["id"]], answers[q["id"]]["abstained"])))}
    summary = {"tag": args.tag, "timestamp": datetime.now().isoformat(timespec="seconds"),
               "config": {**config_of(args), "generation_model": MODEL_CONFIG.model_name, "judge_model": args.judge_model,
                          "embedding_backend": get_embedder().name, "strictness": args.strictness,
                          "mode": "agent" if args.agent else "pipeline", "web": bool(args.agent and args.web)},
               "n_questions": len(questions), "n_completed": len(done_ids), "complete": len(done_ids) == len(questions),
               "diagnostics": diagnostics(questions, answers), "cost": cost_summary(answers)}
    if not args.no_ragas and len(frame):
        in_mask = frame["in_corpus"] == True  # noqa: E712
        summary["metrics_all"] = summarize(frame, frame.index == frame.index)
        summary["metrics_in_corpus"] = summarize(frame, in_mask)
        summary["metrics_out_of_corpus"] = summarize(frame, ~in_mask)
    (RESULTS_DIR / f"{args.tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nProgreso: {summary['n_completed']}/{summary['n_questions']} preguntas completas.")
    if summary["complete"]:
        print("Evaluación COMPLETA. Resultados en eval/results/" + args.tag + "_*")
        print(json.dumps(summary.get("metrics_all", {}), ensure_ascii=False, indent=2))
    else:
        print("Evaluación INCOMPLETA: vuelve a ejecutar el mismo comando para continuar (no uses estos resultados todavía).")


if __name__ == "__main__":
    main()
