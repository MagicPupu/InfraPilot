# Troubleshooting

## A request was refused
The refusal message lists each violated rule: unknown tool, name without the `ip-` prefix, image not allowlisted, or a limit above the quota. Fix the request and retry.

## The agent says there is insufficient evidence
The runbooks do not cover the question. Rephrase it, or add a runbook and run the ingestion again.

## A plan was not executed
Check whether the approval was rejected or whether the model produced output that failed schema validation twice. In both cases nothing is executed.
