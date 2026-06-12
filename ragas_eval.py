"""
ragas_eval.py — Phase 2 of the eval (run in a SEPARATE ragas venv, NOT your app venv).

RAGAS is incompatible with LangChain 1.x (your app stack), so it lives in its own
environment. Setup (one time):

    python -m venv ragas-env
    source ragas-env/bin/activate          # macOS
    pip install -r ragas_requirements.txt

Then, with eval_results.json produced by run_golden.py in the same directory:

    OPENAI_API_KEY=<your Nebius key> python ragas_eval.py
    # (or keep using the same .env — load_dotenv picks it up)

Judge LLM:   meta-llama/Llama-3.3-70B-Instruct (Nebius)
Embeddings:  Qwen/Qwen3-Embedding-8B          (Nebius, for response relevancy)

Metrics:
    faithfulness        — is every claim in the answer grounded in retrieved context?
    response relevancy  — does the answer address the question?
    context precision   — are the retrieved chunks relevant (vs reference)?
    context recall      — did retrieval fetch everything the reference needs?
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

load_dotenv()

NEBIUS_BASE = "https://api.tokenfactory.nebius.com/v1/"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"

RESULTS_PATH = Path("eval_results.json")
SCORES_PATH = Path("ragas_scores.json")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY (Nebius key) not set")

    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["scored"]

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["final_answer"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in rows
        if r["retrieved_contexts"]  # skip anything that escalated without retrieval
    ]
    dataset = EvaluationDataset(samples=samples)
    print(f"Scoring {len(samples)} samples with judge={JUDGE_MODEL}")

    judge = LangchainLLMWrapper(ChatOpenAI(
        base_url=NEBIUS_BASE, model=JUDGE_MODEL, temperature=0, api_key=api_key))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        base_url=NEBIUS_BASE, model=EMBED_MODEL,
        check_embedding_ctx_length=False, api_key=api_key))

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(timeout=180, max_workers=2),  # gentle on Nebius
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in
                   ("user_input", "response", "retrieved_contexts", "reference")]

    print("\n=== Per-question scores ===")
    for i, row in df.iterrows():
        qid = rows[i]["id"]
        scores = "  ".join(f"{c}={row[c]:.2f}" for c in metric_cols)
        print(f"[{qid}] {scores}")

    print("\n=== Averages ===")
    averages = {}
    for c in metric_cols:
        avg = float(df[c].mean())
        averages[c] = round(avg, 4)
        flag = "PASS" if avg >= 0.95 else "below 95% target"
        print(f"{c:<28} {avg:.4f}   {flag}")

    SCORES_PATH.write_text(json.dumps(
        {"averages": averages,
         "per_question": [
             {"id": rows[i]["id"],
              **{c: (None if df.loc[i, c] != df.loc[i, c] else round(float(df.loc[i, c]), 4))
                 for c in metric_cols}}
             for i in range(len(df))
         ]},
        indent=2), encoding="utf-8")
    print(f"\nWrote {SCORES_PATH}")


if __name__ == "__main__":
    main()
