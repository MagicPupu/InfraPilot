from typing import Any

from langgraph.types import interrupt

from infrapilot.state import Approval, InfraState, Plan, PolicyDecision


def render_preview(plan: Plan, policy: PolicyDecision) -> str:
    lines = [f"Plan: {plan.rationale}", ""]
    for i, call in enumerate(plan.calls, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
        lines.append(f"  {i}. {call.tool}({args})")
    lines += ["", f"Policy: {'; '.join(policy.reasons)}"]
    if policy.destructive:
        lines.append("WARNING: this plan is destructive and cannot be undone.")
    lines.append(f"Plan hash: {policy.plan_hash[:12]}")
    return "\n".join(lines)


def approval(state: InfraState) -> dict[str, Any]:
    """Pause for a human decision. Must stay side-effect free: it re-runs on resume."""
    plan = Plan.model_validate(state["plan"])
    policy = PolicyDecision.model_validate(state["policy"])
    answer = interrupt({"preview": render_preview(plan, policy), "plan_hash": policy.plan_hash})
    approved = bool(answer.get("approved")) if isinstance(answer, dict) else False
    return {"approval": Approval(approved=approved, plan_hash=policy.plan_hash).model_dump()}
