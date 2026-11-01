# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [0.1.0] - 2026-11-01

First tagged release.

### Added
- LangGraph agent: router, RAG answer path with programmatic citation checks, action path with
  planner, deterministic policy guard, preflight target check, `interrupt` approval bound to a plan
  hash, MCP executor and verification.
- FastMCP sandbox server over the Docker SDK (`list`, `create`, `scale`, `restart`, `delete`),
  restricted to containers labelled `infrapilot.managed=true`.
- Schema-constrained output with validation and one retry; failures route to `explain_failure`.
- Postgres checkpointer (resume after restart) and optional self-hosted Langfuse tracing.
- Hybrid retrieval: pgvector cosine similarity and Postgres full-text search merged with reciprocal
  rank fusion; labelled retrieval set (31 queries) and `evals/retrieval_eval.py`.
- Eval harness: 66-case dataset (17 adversarial), dev/held-out split, Wilson 95% intervals,
  resumable per-case JSONL results, per-model JSON and Markdown outputs, optional Langfuse scores.
- Governance tests: plan-hash binding, approval requirement, unlabelled containers, indirect
  injection, retry and failure routing, resume with a rebuilt graph (in memory and on Postgres).
- `scripts/demo.sh` and `docs/DEMO.md`; Streamlit front-end; CI with a pgvector service.
- `docs/EU_AI_ACT_NOTES.md`: design notes, not a compliance claim.

### Changed
- Defaults aligned with the tested configuration (`qwen3:4b`, `NUM_CTX=4096`, `127.0.0.1` hosts).
- Screenshot moved to `docs/assets/`.

### Removed
- The identity `rerank()` stub.
