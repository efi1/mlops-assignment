"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------
def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    q_text, db_id, gold_sql = question["question"], question["db_id"], question["gold_sql"]

    # 1. Gold rows (ground truth)
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    # 2. Ask the agent
    try:
        resp = httpx.post(
            agent_url,
            json={"question": q_text, "db": db_id, "tags": {"run": "eval"}},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        agent_error = None
    except Exception as e:  # noqa: BLE001
        data = {"history": [], "iterations": 0}
        agent_error = f"{type(e).__name__}: {e}"

    # 3. All SQL attempts, in order (generate_sql first, then each revise)
    attempts = [h["sql"] for h in data.get("history", []) if h.get("node") in ("generate_sql", "revise")]

    # 4. Re-execute each attempt against the same DB and compare to gold
    per_iteration: list[bool] = []
    for sql in attempts:
        ok, rows, _err = run_sql(db_id, sql)
        per_iteration.append(bool(gold_ok and ok and matches(gold_rows, rows)))

    return {
        "question": q_text,
        "db_id": db_id,
        "gold_ok": gold_ok,
        "gold_error": gold_err,
        "agent_error": agent_error,
        "n_attempts": len(attempts),
        "attempts": attempts,
        "per_iteration_pass": per_iteration,          # e.g. [False, True]
        "final_pass": per_iteration[-1] if per_iteration else False,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.
    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    from agent.graph import MAX_ITERATIONS  # MAX_ITERATIONS=3

    pass_at: dict[str, float] = {}
    for k in range(1, MAX_ITERATIONS + 1):
        passed = 0
        for r in results:
            pi = r["per_iteration_pass"]
            if not pi:
                continue  # agent produced nothing; counts as fail
            # carry-forward: if the agent stopped at iteration j < k,
            # its result at k is its result at j (the last attempt)
            idx = min(k, len(pi)) - 1
            passed += pi[idx]
        pass_at[f"iter_{k}"] = passed / n if n else 0.0

    return {
        "n_questions": n,
        "final_accuracy": sum(r["final_pass"] for r in results) / n if n else 0.0,
        "pass_rate_by_iteration": pass_at,
        "n_agent_errors": sum(1 for r in results if r["agent_error"]),
        "n_gold_errors": sum(1 for r in results if not r["gold_ok"]),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
