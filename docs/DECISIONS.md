# Design decisions

Short, dated records of choices made while scaffolding. Most recent last.

## 1. Fully local LLM stack (Ollama, no LiteLLM)
All inference and embeddings run through Ollama via `langchain-ollama`. `llm.py` is a thin
factory over `OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL` and `NUM_CTX`. No API key, cloud
provider or paid service is required. This replaces the original multi-provider (LiteLLM) design.

## 2. Schema-constrained JSON instead of native tool calling
Small local models are unreliable at function calling. Router, planner and grader ask Ollama for
JSON constrained by a Pydantic schema (`format=<json schema>`), validate the result, retry once with
the validation error appended, and otherwise route to `explain_failure`, which never executes.
`structured.py` implements this directly rather than through `with_structured_output`, so the retry
sees the exact validation error. Qwen3 reasoning is disabled because thinking tokens interfere with
constrained output.

## 3. Deterministic policy, deterministic executor
`policy.py` is plain Python: tool allowlist, per-argument schemas (`extra=forbid`), quotas, image
allowlist and naming rule. The LLM only plans and explains. The executor iterates over the approved
plan and calls MCP tools itself; it never asks the model to call anything.

## 4. Approval bound to a plan hash
`interrupt()` pauses the graph with a readable preview. The recorded approval stores the SHA-256 of
the exact calls that were shown. Before executing, the executor recomputes the hash, re-runs the
policy, and refuses if the plan changed, the policy refuses, or a required approval is missing or
bound to a different hash. Deletes and scale-to-zero are flagged destructive in the preview.

## 5. Policy outcomes
`approve` (read-only plans only), `needs_approval` (any mutation), `refuse` (any violation in any
call refuses the whole plan; more than 5 calls is refused).

## 6. MCP sandbox over Docker
FastMCP server over stdio, spawned by the client with `langchain-mcp-adapters`. Every container is
labeled `infrapilot.managed=true`; delete refuses unlabeled containers as a second line of defense
behind the policy. Tools return JSON strings with an `ok` flag instead of raising.
`scale_service` scales containers sharing the `infrapilot.service` label (Swarm is not used).

## 7. RAG
pgvector with `vector(768)`: the dimension of `nomic-embed-text` was verified against a live Ollama,
not assumed. Chunks follow markdown headings and carry `source`, `heading_path` and a stable id.
Citation faithfulness is checked in code: cited ids must be a non-empty subset of retrieved ids,
otherwise the answer is regenerated once, then replaced by "insufficient evidence". Retrieval is
hybrid (see 11).

## 8. Graph is async, nodes may be sync
The executor and verifier await MCP calls; other nodes are sync and run in LangGraph's thread pool.
The CLI uses `AsyncPostgresSaver`; tests use `InMemorySaver`.

## 9. Evals are deterministic and local-only
Routing accuracy, policy-outcome accuracy, plan validity, citation validity, refusal-when-no-evidence,
latency p50/p95 and tokens/s (from Ollama's own counters). No cost column; a hardware line is
auto-detected instead. An optional LLM-judge exists behind `--judge` and `JUDGE_MODEL` (a stub, not
used in reported numbers). Evals never run in CI. Rates carry a 95% Wilson interval and n; runs are
resumable (one JSONL line per case); cases are split into dev and held-out (see 12).

## 10. Typing
mypy is strict on `src` (`disallow_untyped_defs`), relaxed for tests. LangGraph's `add_node`
overloads reject partial TypedDict states, so node registration goes through one untyped alias in
`graph/build.py`.

## 11. Hybrid retrieval with reciprocal rank fusion, no reranking model
Vector search (pgvector cosine) and Postgres full-text search (generated `tsvector`, GIN index, OR
semantics) each return their top 10; RRF (k=60) merges the two rankings using ranks only, so the
incompatible score scales never need calibrating. No cross-encoder: the VRAM budget is already
taken by the chat and embedding models. Measured on 31 labelled queries it did not beat vector-only
(hit@3 90% versus 94%, overlapping intervals); see the README. It is kept because it is cheap and
covers exact-token queries, which the labelled set barely exercises.

## 12. Dev and held-out eval splits
The first 27 eval cases were used while tuning prompts, so they are all `dev`. Cases added later were
assigned alternately to `dev` and `held-out` within each category before any model ran on them. Only
`dev` misses may justify a prompt change. The adversarial labels are strict on purpose (a harmless
substitution by the model counts as a miss), and a separate "unsafe auto-run" metric measures the
property that matters for safety.
