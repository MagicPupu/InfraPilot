"""Pure helpers for the eval harness: intervals, percentiles, aggregation, rendering.

No network, no Ollama: everything here is unit-tested.
"""

import json
import math
import random
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (two-sided, default 95%)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(k: int, n: int) -> str:
    """'92% [81-97] n=48'. The bracket is the 95% Wilson interval, in percent."""
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}] n={n}"


def percentile(values: list[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def bootstrap_ci(
    values: list[float], q: int, iters: int = 1000, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile bootstrap 95% interval for the q-th percentile (seeded, reproducible)."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(iters):
        sample = [rng.choice(values) for _ in values]
        p = percentile(sample, q)
        assert p is not None
        stats.append(p)
    stats.sort()
    return (stats[int(0.025 * iters)], stats[int(0.975 * iters) - 1])


def fmt_latency(values: list[float], q: int) -> str:
    p = percentile(values, q)
    if p is None:
        return "n/a"
    ci = bootstrap_ci(values, q)
    return f"{p:.1f}s [{ci[0]:.1f}-{ci[1]:.1f}]" if ci else f"{p:.1f}s"


def acceptable(expect: str | list[str]) -> list[str]:
    return [expect] if isinstance(expect, str) else list(expect)


# --- records ----------------------------------------------------------------


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def done_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    return {r["id"] for r in records}


def _count(records: list[dict[str, Any]], key: str) -> tuple[int, int]:
    vals = [r[key] for r in records if r.get(key) is not None]
    return sum(bool(v) for v in vals), len(vals)


METRICS = [
    ("routing acc", "route_ok"),
    ("policy acc", "policy_ok"),
    ("plan-valid", "plan_valid"),
    ("citation-valid", "citation_ok"),
    ("refusal rate", "refusal_ok"),
    ("adv as expected", "safe"),
    ("unsafe auto-run", "unsafe"),
]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of per-case records into metrics with Wilson intervals."""
    out: dict[str, Any] = {"n_cases": len(records)}
    for name, key in METRICS:
        k, n = _count(records, key)
        lo, hi = wilson(k, n)
        out[name] = {"k": k, "n": n, "lo": round(lo, 4), "hi": round(hi, 4)}
    lat = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    out["latency_s"] = {"n": len(lat), "p50": percentile(lat, 50), "p95": percentile(lat, 95)}
    tokens = sum(r.get("tokens", 0) for r in records)
    nanos = sum(r.get("nanos", 0) for r in records)
    out["tokens_per_s"] = round(tokens / (nanos / 1e9), 1) if nanos else None
    return out


def split_records(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return records if split == "all" else [r for r in records if r.get("split") == split]


def table_row(model: str, records: list[dict[str, Any]]) -> dict[str, str]:
    row = {"model": model}
    for name, key in METRICS:
        k, n = _count(records, key)
        row[name] = fmt_rate(k, n)
    lat = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    row["latency p50"] = fmt_latency(lat, 50)
    row["latency p95"] = fmt_latency(lat, 95)
    tokens = sum(r.get("tokens", 0) for r in records)
    nanos = sum(r.get("nanos", 0) for r in records)
    row["tokens/s"] = f"{tokens / (nanos / 1e9):.1f}" if nanos else "n/a"
    return row


def to_markdown(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    return "\n".join(lines)
