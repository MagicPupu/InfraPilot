"""Deterministic local evals. Never run in CI (needs Ollama and, for RAG cases, Postgres).

Dataset format (evals/dataset.jsonl, one JSON object per line):
  id              unique case id
  kind            "question" | "action" (informational)
  input           the user request
  expect_route    "question" | "action"                      -> routing accuracy
  expect_policy   "approve" | "needs_approval" | "refuse"    -> plan validity + policy accuracy
  expect_citation true: answer must cite ids that were retrieved   -> citation validity
  expect_refusal  true: must answer "insufficient evidence"        -> refusal-when-no-evidence

Usage: python evals/run_evals.py --models qwen3:4b,llama3.1:8b [--judge]
"""

import argparse
import asyncio
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from infrapilot.config import get_settings
from infrapilot.graph.build import Deps, build_graph
from infrapilot.graph.planner import make_planner
from infrapilot.graph.policy import evaluate_plan
from infrapilot.graph.rag_nodes import INSUFFICIENT
from infrapilot.graph.router import make_router
from infrapilot.llm import get_chat_model, get_embeddings
from infrapilot.rag.retrieve import PgVectorRetriever
from infrapilot.state import Plan
from infrapilot.structured import invoke_structured

DATASET = Path(__file__).parent / "dataset.jsonl"


class Metrics(BaseCallbackHandler):
    """Collects Ollama's own generation counters (eval_count / eval_duration in ns)."""

    def __init__(self) -> None:
        self.tokens = 0
        self.nanos = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for gens in response.generations:
            for gen in gens:
                meta = getattr(getattr(gen, "message", None), "response_metadata", {}) or {}
                self.tokens += int(meta.get("eval_count", 0))
                self.nanos += int(meta.get("eval_duration", 0))


class NullInvoker:
    """Evals never execute tools."""

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("evals must not execute tools")


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5)


def hardware_line() -> str:
    cpu = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in Path("/proc/cpuinfo").read_text().splitlines()
            if ln.startswith("model name")
        ),
        platform.processor() or "unknown CPU",
    )
    mem_kb = next(
        (
            int(ln.split()[1])
            for ln in Path("/proc/meminfo").read_text().splitlines()
            if ln.startswith("MemTotal")
        ),
        0,
    )
    gpu = "none detected"
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        gpu = out.stdout.strip().replace("\n", "; ") or gpu
    return f"Hardware: CPU {cpu} | GPU {gpu} | RAM visible {mem_kb / 1024 / 1024:.1f} GB"


def miss(case: dict[str, Any], what: str, detail: str) -> None:
    print(f"[miss] {case['id']}: {what} ({detail})", file=sys.stderr)


def pct(hits: int, total: int) -> str:
    return f"{100 * hits / total:.0f}%" if total else "n/a"


def pctl(values: list[float], q: int) -> str:
    if not values:
        return "n/a"
    return f"{statistics.quantiles(values, n=100)[q - 1] if len(values) > 1 else values[0]:.1f}s"


async def evaluate_model(model: str, cases: list[dict[str, Any]], judge: bool) -> dict[str, str]:
    s = get_settings()
    metrics = Metrics()
    llm = get_chat_model(s, model=model).with_config(callbacks=[metrics])
    router, planner = make_router(llm), make_planner(llm)  # type: ignore[arg-type]
    try:
        candidate = PgVectorRetriever(s.database_url, get_embeddings(s))
        candidate.retrieve("ping", 1)
        retriever: PgVectorRetriever | None = candidate
    except Exception as exc:
        print(f"[warn] RAG cases skipped, retriever unavailable: {exc}", file=sys.stderr)
        retriever = None
    graph = build_graph(Deps(llm, retriever, NullInvoker()), None) if retriever else None  # type: ignore[arg-type]

    llm.invoke("ping")  # warm-up: exclude cold model load from the latency figures
    if graph is not None:  # also warm the full RAG path (embedder + LLM co-resident on small GPUs)
        await graph.ainvoke({"request": "warm-up question about the documentation"})
    route_hits = policy_hits = policy_total = plan_valid = cite_hits = cite_total = 0
    refusal_hits = refusal_total = 0
    latencies: list[float] = []
    per_case: list[tuple[float, str]] = []
    judge_scores: list[int] = []
    for case in cases:
        start = time.perf_counter()
        route = router({"request": case["input"]}).get("route")
        route_hits += route == case["expect_route"]
        if route != case["expect_route"]:
            miss(case, "routing", f"got {route!r}, expected {case['expect_route']!r}")
        if "expect_policy" in case:
            policy_total += 1
            out = planner({"request": case["input"]})
            if "plan" in out:
                plan_valid += 1
                outcome = evaluate_plan(Plan.model_validate(out["plan"])).outcome
                policy_hits += outcome == case["expect_policy"]
                if outcome != case["expect_policy"]:
                    miss(
                        case,
                        "policy",
                        f"got {outcome}, expected {case['expect_policy']}; "
                        f"plan={out['plan']['calls']}",
                    )
            else:
                miss(case, "plan invalid", str(out.get("error")))
        elif graph is not None:
            state = await graph.ainvoke({"request": case["input"]})
            if case.get("expect_citation"):
                cite_total += 1
                retrieved = {c["id"] for c in state.get("retrieved", [])}
                cited = state.get("cited_ids", [])
                cite_hits += bool(cited) and set(cited) <= retrieved
                if not (bool(cited) and set(cited) <= retrieved):
                    miss(case, "citation", f"final={state.get('final', '')[:90]!r}")
                if judge and cited and get_settings().judge_model:
                    judge_scores.append(_judge(case["input"], state["answer"]))
            if case.get("expect_refusal"):
                refusal_total += 1
                refusal_hits += state.get("final", "").startswith(INSUFFICIENT)
                if not state.get("final", "").startswith(INSUFFICIENT):
                    miss(case, "refusal", f"final={state.get('final', '')[:90]!r}")
        latencies.append(time.perf_counter() - start)
        per_case.append((latencies[-1], case["id"]))

    slow = ", ".join(f"{c} {t:.1f}s" for t, c in sorted(per_case, reverse=True)[:3])
    print(f"[latency] slowest: {slow}", file=sys.stderr)
    tps = f"{metrics.tokens / (metrics.nanos / 1e9):.1f}" if metrics.nanos else "n/a"
    row = {
        "model": model,
        "routing acc": pct(route_hits, len(cases)),
        "policy acc": pct(policy_hits, policy_total),
        "plan-valid %": pct(plan_valid, policy_total),
        "citation-valid %": pct(cite_hits, cite_total),
        "refusal %": pct(refusal_hits, refusal_total),
        "latency p50": pctl(latencies, 50),
        "latency p95": pctl(latencies, 95),
        "tokens/s": tps,
    }
    if judge:
        row["judge (1-5)"] = f"{statistics.mean(judge_scores):.1f}" if judge_scores else "n/a"
    return row


def _judge(question: str, answer: str) -> int:
    """Minimal LLM-judge. TODO: rubric-based scoring and agreement checks."""
    llm = get_chat_model(model=get_settings().judge_model)
    msgs = [
        SystemMessage(content='Rate the answer quality from 1 to 5. Reply {"score": int}.'),
        HumanMessage(content=f"Question: {question}\nAnswer: {answer}"),
    ]
    return invoke_structured(llm, msgs, JudgeScore).score


def to_markdown(rows: list[dict[str, str]]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=get_settings().llm_model)
    parser.add_argument("--judge", action="store_true", help="enable LLM-judge (needs JUDGE_MODEL)")
    args = parser.parse_args()
    cases = [json.loads(ln) for ln in DATASET.read_text().splitlines() if ln.strip()]
    rows = [
        asyncio.run(evaluate_model(m.strip(), cases, args.judge)) for m in args.models.split(",")
    ]
    print(hardware_line() + "\n")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
