"""
run_golden.py — Phase 1 of the RAGAS eval (run in your APP venv, alongside tda_pipeline.py).

Runs every golden question through the real pipeline, checks routing/language,
and dumps answers + retrieved contexts to eval_results.json for Phase 2 scoring.

Usage:
    python run_golden.py            # expects golden_dataset.json in the same dir

Each question uses a FRESH thread_id so condense_question never fires
(single-turn eval — we're scoring RAG quality, not memory).
"""

import json
import time
import uuid
from pathlib import Path

from tda_pipeline import build_app

GOLDEN_PATH = Path("golden_dataset.json")
OUT_PATH = Path("eval_results.json")


def run_one(app, question: str) -> dict:
    """Invoke the graph directly (not answer()) so we can capture retrieved context."""
    cfg = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    t0 = time.perf_counter()
    out = app.invoke({"text": question}, cfg)
    elapsed = time.perf_counter() - t0
    return {
        "final_answer": out.get("final_answer", ""),
        "route": out.get("route", ""),
        "language": out.get("language", ""),
        "retrieved_contexts": [d.page_content for d in out.get("context", []) or []],
        "latency_s": round(elapsed, 2),
    }


def main():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print("Building app (embedding corpus)...")
    app = build_app()

    results = {"scored": [], "routing_only": []}
    route_pass = lang_pass = 0
    route_total = lang_total = 0

    print("\n--- Scored questions (feed RAGAS in Phase 2) ---")
    for item in golden["scored"]:
        r = run_one(app, item["question"])
        ok = r["route"] == item["expected_route"]
        route_total += 1
        route_pass += ok
        results["scored"].append({**item, **r, "route_ok": ok})
        print(f"[{item['id']}] route={r['route']:<12} "
              f"{'OK ' if ok else 'MISS(' + item['expected_route'] + ')'} "
              f"{r['latency_s']}s  {r['final_answer'][:60]!r}")

    print("\n--- Routing-only probes ---")
    for item in golden["routing_only"]:
        r = run_one(app, item["question"])
        r_ok = r["route"] == item["expected_route"]
        l_ok = r["language"].lower().startswith(item["expected_language"].lower())
        route_total += 1
        route_pass += r_ok
        lang_total += 1
        lang_pass += l_ok
        results["routing_only"].append(
            {**item, **r, "route_ok": r_ok, "language_ok": l_ok})
        print(f"[{item['id']}] route={r['route']:<12} {'OK' if r_ok else 'MISS'}  "
              f"lang={r['language']:<8} {'OK' if l_ok else 'MISS'}  "
              f"{r['latency_s']}s")

    summary = {
        "routing_accuracy": f"{route_pass}/{route_total}",
        "language_accuracy": f"{lang_pass}/{lang_total}",
    }
    results["summary"] = summary
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\nRouting accuracy : {summary['routing_accuracy']}")
    print(f"Language accuracy: {summary['language_accuracy']}")
    print(f"\nWrote {OUT_PATH} — now score it in the ragas venv:")
    print("    python ragas_eval.py")


if __name__ == "__main__":
    main()
