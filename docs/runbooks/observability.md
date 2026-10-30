# Observability

## Traces
When tracing is enabled, every run is recorded in the self-hosted Langfuse instance: router decision, plan, policy outcome, approval and tool results.

## Retention
Traces stay on the local machine. They contain user requests, so treat them as internal data.

## Enabling
Set `LANGFUSE_ENABLED=true` in the environment file and start the stack with `make up`.
