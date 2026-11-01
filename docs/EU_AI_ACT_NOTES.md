# EU AI Act and GDPR design notes

These are engineering notes, not legal advice and not a compliance claim. InfraPilot is a
portfolio project that manages a local Docker sandbox. No risk classification has been carried out
and it is not intended for any use listed in Annex III. The notes only show how features line up
with the themes the Act emphasizes for high-risk systems: human oversight, record-keeping,
transparency, and accuracy and robustness.

## Human oversight (Article 14 theme)

| Design feature | Where |
|---|---|
| Every state-changing plan pauses at `interrupt()` with a readable preview | `graph/approval.py` |
| The approval is stored with the SHA-256 of the exact calls shown; the executor recomputes it and blocks on mismatch | `graph/executor.py`, `state.py` |
| Destructive plans (delete, scale to zero) carry an explicit warning | `graph/approval.py` |
| Rejecting executes nothing; read-only plans are the only ones that skip approval | `graph/build.py` |
| The model never calls tools: a deterministic executor does | `graph/executor.py` |

Tests: `tests/integration/test_governance.py`, `tests/unit/test_executor.py`.

## Record-keeping (Article 12 theme)

- LangGraph checkpoints in Postgres hold each run's state: request, plan, policy decision,
  approval and results.
- Optional self-hosted Langfuse traces record the router, planner, approval and tool steps.

Limits: checkpoints and traces are ordinary database rows. They are not append-only or
tamper-evident, there is no user identity (single-user CLI), and no retention job exists.

## Transparency (Article 50 theme)

- Answers list the runbook chunks they cite. Citations are checked in code: ids that were not
  retrieved are rejected, and the answer is regenerated once and then replaced by "insufficient
  evidence".
- Refusals list the violated rules; failures say that nothing was executed.
- The CLI and UI do not currently print a notice that the user is talking to an AI system.

## Accuracy and robustness (Article 15 theme)

- Policy is plain Python (allowlists, quotas, naming), not model output, and is unit-tested.
- Model output that fails schema validation after one retry never reaches execution.
- Measured behaviour on a small local dataset, with intervals and the failures, is in the README
  and `evals/results/`. Small models make mistakes; the deterministic layers exist for that reason.

## GDPR and data sovereignty

- Prompts, runbooks, embeddings and traces stay on the machine: the LLM step involves no third-party
  processor and no cross-border transfer.
- This does not make a deployment compliant by itself. Requests may contain personal data; Langfuse
  traces and checkpoints are stored locally and need a retention rule. See
  `docs/runbooks/observability.md`.
