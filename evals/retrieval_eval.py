"""Retrieval quality: hit@3 and MRR for vector, full-text and hybrid (RRF) retrieval.

Needs Postgres with the runbooks ingested (``make up && make ingest``) and Ollama for query
embeddings. One result file per run: evals/results/retrieval-<date>.json (+ .md).

hit@3: an expected chunk is among the top 3. MRR: mean of 1/rank of the first expected chunk
(0 when none is in the top 10). With several expected ids, any one counts.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from evalkit import fmt_rate, to_markdown

from infrapilot.config import get_settings
from infrapilot.llm import get_embeddings
from infrapilot.rag.retrieve import Mode, PgVectorRetriever

HERE = Path(__file__).parent
MODES: list[Mode] = ["vector", "fulltext", "hybrid"]


def first_rank(returned: list[str], expected: set[str]) -> int | None:
    return next((r for r, i in enumerate(returned, 1) if i in expected), None)


def score(ranks: list[int | None]) -> dict[str, Any]:
    hits = sum(r is not None and r <= 3 for r in ranks)
    return {
        "hit@3_k": hits,
        "n": len(ranks),
        "mrr": sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    s = get_settings()
    retriever = PgVectorRetriever(s.database_url, get_embeddings(s))
    queries = [json.loads(ln) for ln in (HERE / "retrieval.jsonl").read_text().splitlines() if ln]
    per_mode: dict[str, list[int | None]] = {m: [] for m in MODES}
    rows: list[dict[str, Any]] = []
    for q in queries:
        expected = set(q["expected_ids"])
        row: dict[str, Any] = {"id": q["id"], "query": q["query"], "kind": q["kind"]}
        for mode in MODES:
            got = [c.id for c in retriever.retrieve(q["query"], 10, mode=mode)]
            rank = first_rank(got, expected)
            per_mode[mode].append(rank)
            row[mode] = rank
        rows.append(row)
    summary = {m: score(r) for m, r in per_mode.items()}
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({q["kind"] for q in queries}):
        idx = [n for n, q in enumerate(queries) if q["kind"] == kind]
        by_kind[kind] = {m: score([per_mode[m][n] for n in idx]) for m in MODES}
    doc = {
        "date": args.date,
        "embed_model": s.embed_model,
        "queries": len(queries),
        "depth": 10,
        "summary": summary,
        "by_kind": by_kind,
        "rows": rows,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / f"retrieval-{args.date}.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False)
    )
    table = to_markdown(
        [
            {
                "retriever": m,
                "hit@3": fmt_rate(v["hit@3_k"], v["n"]),
                "MRR": f"{v['mrr']:.3f}",
            }
            for m, v in summary.items()
        ]
    )
    md = f"# Retrieval {args.date}\n\nembeddings: {s.embed_model}, n={len(queries)}\n\n{table}\n"
    (args.results_dir / f"retrieval-{args.date}.md").write_text(md)
    print(md)
    misses = [r for r in rows if r["hybrid"] is None or r["hybrid"] > 3]
    for r in misses:
        print(f"[hybrid miss] {r['id']} {r['query']!r} ranks v={r['vector']} f={r['fulltext']}")


if __name__ == "__main__":
    main()
