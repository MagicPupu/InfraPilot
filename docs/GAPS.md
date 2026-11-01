# Gap audit

Audit of the repository at commit `39a1180`, against the interview-readiness task list.
Baseline: ruff clean, mypy clean, 45 tests pass (FakeLLM, no network).

The resolution table at the end records the state after the fixes. The sections in between are the
original audit and are left unchanged.

Status values: done, partial, missing.

## 1. Eval dataset

| Item | Status | Evidence | Planned fix |
|---|---|---|---|
| At least 45 cases | missing | `evals/dataset.jsonl` has 27 cases | Grow to 50 or more |
| Balance: routing, policy (allow, approval, refuse), plan validity, citation, refusal | partial | Policy and citation cases exist; no typos, ambiguous phrasing or mixed cases | Add typo and ambiguous routing cases, more refusals |
| Adversarial block (10+ cases) | missing | None in dataset; no injected runbook | Add 12+ cases: direct injection, indirect injection via a poisoned chunk, unlabeled delete, quota abuse, mixed legal and destructive, non-allowlisted tools |
| Dataset format and labelling doc | partial | Format only in the `run_evals.py` docstring | Write `evals/README.md` |
| Held-out split (about 30%) | missing | No `split` field | Add `split: dev or heldout`, assigned once, not used when tuning prompts |
| Metrics with n and Wilson 95% interval | missing | `pct()` prints a bare percentage | Add Wilson interval, report `k/n [lo, hi]` |

Constraint: indirect injection needs a retriever returning a poisoned chunk. A case-level
`inject_chunk` field with an in-memory retriever covers it without touching the database.

## 2. Retrieval

| Item | Status | Evidence | Planned fix |
|---|---|---|---|
| Real reranking | missing | `rerank()` is an identity stub (`rag/retrieve.py`) | Hybrid: pgvector cosine + Postgres full-text, merged with reciprocal rank fusion |
| Full-text index | missing | `rag/schema.sql` has only the HNSW index | Add a generated `tsvector` column and GIN index (idempotent migration) |
| Labelled retrieval set (25+) | missing | None | `evals/retrieval.jsonl`, expected chunk ids |
| hit@3 and MRR for vector, full-text, hybrid | missing | None | `evals/retrieval_eval.py`, results saved under `evals/results/` |

Chunk ids are `sha1(source|heading_path|idx)[:10]`, so labels can be written against stable ids
and validated against the ingested corpus.

## 3. Security and governance tests

| Item | Status | Evidence | Planned fix |
|---|---|---|---|
| Plan hash mismatch blocks executor | partial | `executor.authorize` checks it; `test_tampered_plan_after_policy_is_blocked` and `test_approval_for_other_plan_is_blocked` cover it | Add a graph-level test (state tampered between approval and execution) |
| Destructive action needs recorded approval | partial | `authorize` enforces; `test_destructive_without_approval_is_blocked` covers it at unit level | Add a graph-level test for a delete |
| Unlabeled container never touched | partial | `docker_tools.py` checks the label; tests cover only restart of an unmanaged container | Add tests for delete and scale on unlabeled containers |
| Instruction in retrieved chunk cannot add a tool call | missing | No test | Graph test: poisoned chunk on the question path, assert no tool call and no plan |
| Invalid output after one retry goes to `explain_failure` | partial | Router case tested (`test_invalid_router_output_routes_to_explain_failure`); planner case not | Add planner and answer cases |
| Postgres checkpointer used in default path | done | `cli.py` and `ui.py` use `AsyncPostgresSaver`; `build_graph` accepts any saver (InMemorySaver only in tests) | Document |
| Resume after the graph object is rebuilt | missing | Tests reuse the same graph object | Test with a rebuilt graph on the same saver (in-memory), plus a Postgres-backed variant that is skipped without a database |

## 4. Multi-model benchmark

| Item | Status | Evidence | Planned fix |
|---|---|---|---|
| Resumable runs (JSONL per case) | missing | `run_evals.py` keeps everything in memory | Append one line per case, skip finished ones |
| `evals/results/<model>-<date>.json` and markdown summary | missing | `evals/results/` does not exist | Implement |
| Dev and held-out reported separately | missing | | Implement |
| Hardware line | partial | `hardware_line()` lacks NUM_CTX and VRAM wording | Extend |
| Intervals in table | missing | | Wilson |
| Runs for qwen3:4b, qwen3:8b, llama3.1:8b | missing | Only `qwen3:4b` is installed; the existing README row has no saved result file | See "Blockers" |
| Findings section from real numbers | missing | | Write after runs |
| Langfuse score export (optional) | missing | | Implement behind `LANGFUSE_ENABLED` |

## 5. Demo assets

| Item | Status | Evidence | Planned fix |
|---|---|---|---|
| `scripts/demo.sh` | missing | No `scripts/` | Create |
| `docs/DEMO.md` and `docs/assets/` filenames | missing | One screenshot at `docs/img/langfuse-trace.png` | Create; commented links in README |

## 6. README and hygiene

| Item | Status | Evidence | Planned fix |
|---|---|---|---|
| Mermaid architecture | partial | Present; omits preflight node | Update |
| Quickstart verified from a clean clone | missing | Not verified | Test in a fresh venv (needs Docker for the DB steps) |
| Benchmark and retrieval tables | partial | One row, no saved result file, no intervals | Replace from saved results |
| Design decisions | done | README section and `docs/DECISIONS.md` | Add "why local-first" detail only if missing |
| Known limitations | partial | Lists the stub reranker and "benchmark not filled" (stale) | Update |
| EU AI Act notes | partial | `docs/EU_AI_ACT_NOTES.md` exists, 29 lines | Re-read for accuracy: "Art. 50 spirit" and "Art. 12" wording; add a not-a-compliance-claim line |
| Secrets | partial | No secrets found in tracked files; `.env` ignored | Re-scan before tag |
| `.env.example` complete | partial | Default `NUM_CTX=8192` and `LLM_MODEL=qwen3:8b` differ from the tested config; `OLLAMA_BASE_URL` uses `localhost` while the README advises 127.0.0.1 | Align |
| LICENSE | done | `LICENSE` | |
| CI | partial | `.github/workflows/ci.yml` runs ruff, format, mypy, pytest without Ollama; green status not checked | Check with `gh run list` |
| Badges | missing | | Add CI and license |
| CHANGELOG, tag `v0.1.0` | missing | No tags | Create locally |

## Blockers found during the audit

1. Docker Desktop is not running (`docker.exe ps` cannot reach the engine) and the `docker` CLI is
   not wired into this WSL distro. Everything that needs Postgres is blocked until it is started:
   retrieval evals, RAG eval cases, the Postgres checkpointer test, the clean-clone quickstart,
   the demo script.
2. Only `qwen3:4b` is installed. `qwen3:8b` and `llama3.1:8b` are about 5 GB each and do not fit
   fully in 6 GB of VRAM with a KV cache, so they will partly run on CPU and be slow.
3. Running 50+ cases times three models, including 8B models with CPU offload, may exceed 30
   minutes. This needs the owner's approval before it starts.

## Resolution (2026-11-01)

| Item | Result | Evidence |
|---|---|---|
| 1. Dataset | done | 66 cases, 17 adversarial, 18 held-out (27%); `evals/README.md`; Wilson intervals in every table |
| 2. Retrieval | done | RRF hybrid in `rag/retrieve.py`; 31 labelled queries; `evals/results/retrieval-2026-11-01.json`. Hybrid did not beat vector-only |
| 3. Governance tests | done | `tests/integration/test_governance.py`, `test_postgres.py`, `test_docker_tools.py`; two mutations of the code were confirmed to fail the tests |
| 4. Benchmark | partial | Harness resumable, JSON and Markdown outputs, Langfuse export verified. Only `qwen3:4b` was run; `qwen3:8b` and `llama3.1:8b` not run (owner decision: skip, too slow for the hardware) |
| 5. Demo assets | partial | `scripts/demo.sh` run end to end from a clean clone; `docs/DEMO.md` written. GIF and Streamlit screenshot are manual |
| 6. README and hygiene | done, except remote CI | README, EU notes, CHANGELOG, badges, `.env.example`, secret scan. The updated workflow (pgvector service) has not run on GitHub because nothing was pushed |
