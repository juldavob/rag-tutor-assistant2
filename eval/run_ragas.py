"""
run_ragas.py
------------
Evalúa el pipeline RAG con Ragas (faithfulness, answer_relevancy,
context_precision, context_recall) sobre eval/eval_questions.json.

Cada ejecución usa una configuración (chunk_size, overlap, top_k) y un
`--tag` que nombra la corrida. Para la iteración de mejora se ejecuta dos
veces (baseline y mejora) y luego se comparan con compare_runs.py.

Ejemplos:
    # prueba rápida con 4 preguntas (gasta pocas llamadas a la API)
    python eval/run_ragas.py --tag prueba --limit 4

    # corrida baseline y corrida mejorada
    python eval/run_ragas.py --tag baseline --chunk-size 300 --overlap 30 --top-k 3
    python eval/run_ragas.py --tag mejora   --chunk-size 700 --overlap 100 --top-k 4

    # si Ragas falló a medias por límites de la API, reutiliza las respuestas ya generadas
    python eval/run_ragas.py --tag baseline --chunk-size 300 --overlap 30 --top-k 3 --reuse-answers

Salidas en eval/results/: <tag>_answers.json, <tag>_per_question.csv, <tag>_summary.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from config import EVAL_DIR, MODEL_CONFIG, RETRIEVAL_CONFIG  # noqa: E402
from embeddings import get_embedder  # noqa: E402
from llm import get_api_key  # noqa: E402
from rag_pipeline import build_pipeline  # noqa: E402

RESULTS_DIR = EVAL_DIR / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="nombre de la corrida (ej. baseline, mejora)")
    p.add_argument("--chunk-size", type=int, default=RETRIEVAL_CONFIG.chunk_size)
    p.add_argument("--overlap", type=int, default=RETRIEVAL_CONFIG.chunk_overlap)
    p.add_argument("--top-k", type=int, default=RETRIEVAL_CONFIG.top_k)
    p.add_argument("--questions", default=str(EVAL_DIR / "eval_questions.json"))
    p.add_argument("--limit", type=int, help="evaluar solo N preguntas repartidas por todo el set")
    p.add_argument("--workers", type=int, default=2, help="llamadas simultáneas de Ragas (bajar si hay errores 429)")
    p.add_argument("--timeout", type=int, default=300, help="segundos máximos por llamada del juez")
    p.add_argument("--sleep", type=float, default=0.0, help="pausa (s) entre preguntas al generar respuestas")
    p.add_argument("--judge-model", default=MODEL_CONFIG.model_name, help="modelo Gemini que actúa como juez")
    p.add_argument("--reuse-answers", action="store_true", help="reutiliza eval/results/<tag>_answers.json")
    p.add_argument("--no-ragas", action="store_true", help="solo genera respuestas y diagnósticos (sin Ragas)")
    return p.parse_args()


def select_questions(questions, limit):
    if not limit or limit >= len(questions):
        return questions
    idx = sorted(set(np.linspace(0, len(questions) - 1, limit).astype(int).tolist()))
    return [questions[i] for i in idx]


def generate_answers(args, questions):
    path = RESULTS_DIR / f"{args.tag}_answers.json"
    if args.reuse_answers and path.exists():
        print(f"Reutilizando respuestas de {path.name}")
        saved = {a["id"]: a for a in json.loads(path.read_text(encoding="utf-8"))}
        if all(q["id"] in saved for q in questions):
            return [saved[q["id"]] for q in questions]
        print("  (faltan preguntas en el archivo; se vuelven a generar todas)")

    pipeline = build_pipeline(args.chunk_size, args.overlap, args.top_k)
    print(f"Generando {len(questions)} respuestas con {pipeline.llm.model_name} "
          f"(chunk={args.chunk_size}, overlap={args.overlap}, top_k={args.top_k})")
    answers = []
    for n, q in enumerate(questions, start=1):
        result = pipeline.ask(q["question"], history=None, top_k=args.top_k)
        answers.append({
            "id": q["id"], "question": q["question"], "response": result.answer,
            "retrieved_contexts": [c.text for c in result.retrieved],
            "retrieved_sources": [c.source for c in result.retrieved],
            "cited_ids": result.metadata.get("fuentes", []),
            "abstained": result.abstained,
            "confidence": result.metadata.get("nivel_confianza"),
            "latency_s": result.latency_s,
        })
        print(f"  [{n}/{len(questions)}] {q['id']} {'(abstención)' if result.abstained else ''} {result.latency_s}s")
        if args.sleep:
            time.sleep(args.sleep)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    return answers


def diagnostics(questions, answers):
    """Métricas sin LLM: acierto de recuperación y control de alucinaciones."""
    hits, out_abstained, in_abstained, n_in, n_out = 0, 0, 0, 0, 0
    for q, a in zip(questions, answers):
        if q["in_corpus"]:
            n_in += 1
            hits += int(q["expected_source"] in a["retrieved_sources"])
            in_abstained += int(a["abstained"])
        else:
            n_out += 1
            out_abstained += int(a["abstained"])
    return {
        "retrieval_hit_rate": round(hits / n_in, 3) if n_in else None,
        "abstention_rate_out_of_corpus": round(out_abstained / n_out, 3) if n_out else None,
        "false_abstention_rate_in_corpus": round(in_abstained / n_in, 3) if n_in else None,
        "n_in_corpus": n_in, "n_out_of_corpus": n_out,
    }


def run_ragas(args, questions, answers):
    from langchain_core.embeddings import Embeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    api_key = get_api_key()
    if not api_key:
        sys.exit("Falta GEMINI_API_KEY (variable de entorno o archivo .env) para el modelo juez de Ragas.")

    class LocalEmbeddings(Embeddings):
        """Reutiliza el mismo modelo de embeddings local del pipeline."""
        def __init__(self):
            self._embedder = get_embedder()
        def embed_documents(self, texts):
            return self._embedder.embed_documents(texts)
        def embed_query(self, text):
            return self._embedder.embed_query(text)

    judge = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=args.judge_model, google_api_key=api_key, temperature=0),
        bypass_n=True,  # Gemini no soporta n>1 completions: Ragas hace llamadas separadas
    )
    embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())

    dataset = EvaluationDataset.from_list([{
        "user_input": q["question"],
        "retrieved_contexts": a["retrieved_contexts"],
        "response": a["response"],
        "reference": q["ground_truth"],
    } for q, a in zip(questions, answers)])

    print(f"Evaluando con Ragas (juez: {args.judge_model}, workers={args.workers}). Puede tardar varios minutos…")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge, embeddings=embeddings,
        run_config=RunConfig(timeout=args.timeout, max_retries=8, max_wait=90, max_workers=args.workers),
        raise_exceptions=False, show_progress=True,
    )
    return result.to_pandas()


def summarize(frame, mask):
    out = {}
    for metric in METRICS:
        values = frame.loc[mask, metric].dropna() if metric in frame else []
        out[metric] = {"mean": round(float(np.mean(values)), 4) if len(values) else None, "n_valid": int(len(values))}
    return out


def main():
    args = parse_args()
    questions = select_questions(json.loads(Path(args.questions).read_text(encoding="utf-8")), args.limit)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    answers = generate_answers(args, questions)
    diag = diagnostics(questions, answers)
    print("Diagnóstico sin LLM:", diag)

    summary = {
        "tag": args.tag, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"chunk_size": args.chunk_size, "overlap": args.overlap, "top_k": args.top_k,
                   "generation_model": MODEL_CONFIG.model_name, "judge_model": args.judge_model,
                   "embedding_backend": get_embedder().name},
        "n_questions": len(questions), "diagnostics": diag,
    }

    if not args.no_ragas:
        import pandas as pd
        frame = run_ragas(args, questions, answers)
        meta = pd.DataFrame([{"id": q["id"], "type": q["type"], "in_corpus": q["in_corpus"],
                              "abstained": a["abstained"]} for q, a in zip(questions, answers)])
        frame = pd.concat([meta, frame.reset_index(drop=True)], axis=1)
        frame.to_csv(RESULTS_DIR / f"{args.tag}_per_question.csv", index=False, encoding="utf-8")
        in_mask = frame["in_corpus"] == True  # noqa: E712
        summary["metrics_all"] = summarize(frame, frame.index == frame.index)
        summary["metrics_in_corpus"] = summarize(frame, in_mask)
        summary["metrics_out_of_corpus"] = summarize(frame, ~in_mask)
        missing = {m: int(frame[m].isna().sum()) for m in METRICS if m in frame}
        summary["missing_values"] = missing
        if any(missing.values()):
            print(f"ATENCIÓN: hay valores sin calcular {missing} (probablemente límites de la API). "
                  f"Repite con --reuse-answers y menos --workers.")

    (RESULTS_DIR / f"{args.tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("metrics_all")}, ensure_ascii=False, indent=2))
    print(f"Resultados guardados en eval/results/{args.tag}_*")


if __name__ == "__main__":
    main()
