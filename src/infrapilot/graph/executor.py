"""Deterministic executor: runs the approved plan; the LLM is never involved."""

from collections.abc import Awaitable, Callable
from typing import Any

from infrapilot.graph.policy import TOOLS, evaluate_plan
from infrapilot.mcp_client import ToolInvoker
from infrapilot.state import Approval, InfraState, Plan, PolicyDecision, plan_hash


def authorize(state: InfraState) -> str | None:
    """Return a reason to block execution, or None if the plan may run."""
    plan = Plan.model_validate(state["plan"])
    policy = PolicyDecision.model_validate(state["policy"])
    digest = plan_hash(plan.calls)
    if digest != policy.plan_hash:
        return "plan was modified after the policy check"
    if evaluate_plan(plan).outcome == "refuse" or policy.outcome == "refuse":
        return "plan is refused by policy"
    if policy.outcome == "needs_approval":
        raw = state.get("approval") or {}
        if not raw:
            return "no recorded approval"
        approval = Approval.model_validate(raw)
        if not approval.approved or approval.plan_hash != digest:
            return "approval missing or bound to a different plan"
    return None


def make_executor(
    invoker: ToolInvoker,
) -> Callable[[InfraState], Awaitable[dict[str, Any]]]:
    async def executor(state: InfraState) -> dict[str, Any]:
        blocked = authorize(state)
        if blocked:
            return {"error": f"execution blocked: {blocked}"}
        results: list[dict[str, Any]] = []
        for call in Plan.model_validate(state["plan"]).calls:
            if call.tool not in TOOLS:
                return {"error": f"execution blocked: tool '{call.tool}' not in plan allowlist"}
            outcome = await invoker.call(call.tool, call.args)
            results.append({"tool": call.tool, "args": call.args, **outcome})
            if not outcome.get("ok"):
                break
        return {"results": results}

    return executor
