from typing import Any

from infrapilot.state import InfraState


def explain_failure(state: InfraState) -> dict[str, Any]:
    """Safe terminal node: explains what went wrong, never executes anything."""
    return {
        "final": f"I could not complete this request safely. Nothing was executed.\nReason: {state.get('error', 'unknown error')}"
    }


def explain_refusal(state: InfraState) -> dict[str, Any]:
    reasons = state.get("policy", {}).get("reasons", [])
    bullets = "\n".join(f"- {r}" for r in reasons)
    return {"final": f"This request was refused by policy. Nothing was executed.\n{bullets}"}


def rejected(state: InfraState) -> dict[str, Any]:
    return {"final": "Plan rejected by the reviewer. Nothing was executed."}
