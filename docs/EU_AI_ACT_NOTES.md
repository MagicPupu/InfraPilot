# EU AI Act and GDPR notes

Engineering notes, not legal advice. InfraPilot is a portfolio project that manages a local
sandbox; it is not a high-risk system under Annex III. These notes show how the design maps onto
the principles the Act emphasizes.

## Human oversight (Art. 14 spirit)
- Every state-changing plan pauses at `interrupt()` and shows a readable preview.
- The approval is bound to the plan hash; any change voids it.
- Destructive operations carry an explicit warning.
- The reviewer can always reject; rejection executes nothing.

## Record-keeping and logging (Art. 12 spirit)
- LangGraph checkpoints persist each run's state, including plan, policy decision and approval.
- Optional self-hosted Langfuse traces record router, planner, policy, approval and tool results.

## Transparency (Art. 50 spirit)
- Users interact with an AI system and the CLI says so; answers list their sources.
- Answers without verified citations are replaced by "insufficient evidence".

## Robustness and accuracy
- Policy is deterministic code, not model output.
- Model output that fails schema validation twice never reaches execution.

## GDPR and data sovereignty
- Everything runs locally: prompts, runbooks, embeddings and traces never leave the machine.
- No third-party processor is involved, which removes transfer and sub-processor questions.
- Requests may contain personal data; Langfuse traces and checkpoints are stored locally and should
  follow your retention rules. See `docs/runbooks/observability.md`.
