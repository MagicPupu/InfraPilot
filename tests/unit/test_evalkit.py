import json
from pathlib import Path

import pytest
from evalkit import (
    acceptable,
    append_record,
    bootstrap_ci,
    done_ids,
    fmt_rate,
    load_records,
    percentile,
    split_records,
    summarize,
    wilson,
)

DATASET = Path(__file__).parents[2] / "evals" / "dataset.jsonl"
ROUTES = {"question", "action"}
POLICIES = {"approve", "needs_approval", "refuse"}
CATEGORIES = {"routing", "policy", "plan", "citation", "no_evidence", "adversarial"}


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (5, 10, 0.2366, 0.7634),
        (45, 50, 0.7864, 0.9565),
    ],
)
def test_wilson_reference_values(k, n, lo, hi):
    got_lo, got_hi = wilson(k, n)
    assert got_lo == pytest.approx(lo, abs=1e-3) and got_hi == pytest.approx(hi, abs=1e-3)


def test_wilson_empty_sample_is_uninformative():
    assert wilson(0, 0) == (0.0, 1.0)


def test_fmt_rate_shows_n_and_interval():
    assert fmt_rate(9, 10) == "90% [60-98] n=10"
    assert fmt_rate(0, 0).startswith("n/a")


def test_percentile_and_bootstrap_are_deterministic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(values, 50) == pytest.approx(5.5)
    assert bootstrap_ci(values, 50) == bootstrap_ci(values, 50)
    assert percentile([], 50) is None and bootstrap_ci([1.0], 50) is None


def test_resume_skips_finished_cases(tmp_path):
    path = tmp_path / "m-2026-01-01.jsonl"
    assert load_records(path) == []
    append_record(path, {"id": "a"})
    append_record(path, {"id": "b"})
    assert done_ids(load_records(path)) == {"a", "b"}


def test_summarize_counts_only_applicable_cases():
    records = [
        {"id": "1", "split": "dev", "route_ok": True, "policy_ok": True, "latency_s": 1.0},
        {"id": "2", "split": "dev", "route_ok": False, "latency_s": 3.0},
        {"id": "3", "split": "heldout", "route_ok": True, "citation_ok": False, "latency_s": 2.0},
    ]
    s = summarize(split_records(records, "dev"))
    assert (s["routing acc"]["k"], s["routing acc"]["n"]) == (1, 2)
    assert (s["policy acc"]["k"], s["policy acc"]["n"]) == (1, 1)
    assert s["citation-valid"]["n"] == 0
    assert len(split_records(records, "all")) == 3


def test_acceptable_accepts_string_or_list():
    assert acceptable("refuse") == ["refuse"]
    assert acceptable(["refuse", "approve"]) == ["refuse", "approve"]


# --- dataset integrity -------------------------------------------------------


def _cases() -> list[dict]:
    return [json.loads(ln) for ln in DATASET.read_text().splitlines() if ln.strip()]


def test_dataset_is_large_enough_and_unique():
    cases = _cases()
    assert len(cases) >= 45
    assert len({c["id"] for c in cases}) == len(cases)


def test_dataset_fields_are_valid():
    for c in _cases():
        assert c["category"] in CATEGORIES, c["id"]
        assert c["split"] in {"dev", "heldout"}, c["id"]
        assert c["expect_route"] in ROUTES, c["id"]
        assert set(acceptable(c.get("expect_policy", "approve"))) <= POLICIES, c["id"]
        if "inject_chunk" in c:
            assert {"source", "heading_path", "text"} == set(c["inject_chunk"]), c["id"]
        if c["expect_route"] == "action":
            assert "expect_policy" in c, c["id"]
        else:
            assert c.get("expect_citation") or c.get("expect_refusal"), c["id"]


def test_dataset_has_adversarial_block_and_heldout_split():
    cases = _cases()
    adversarial = [c for c in cases if c["category"] == "adversarial"]
    assert len(adversarial) >= 10
    assert any("inject_chunk" in c for c in adversarial)
    heldout = [c for c in cases if c["split"] == "heldout"]
    assert 0.2 <= len(heldout) / len(cases) <= 0.35
    assert any(c["category"] == "adversarial" for c in heldout)
