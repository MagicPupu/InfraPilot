"""Pre-approval check: targets of restart/delete/scale must exist.

Stops a plan that names a non-existent container *before* a human is asked to
approve it. Read-only (only calls ``list_containers``); the policy guard stays
the sole authority on what is allowed.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from infrapilot.mcp_client import ToolInvoker
from infrapilot.state import InfraState, Plan

_NEEDS_EXISTING = {"restart_container", "delete_container", "scale_service"}


def make_preflight(invoker: ToolInvoker) -> Callable[[InfraState], Awaitable[dict[str, Any]]]:
    async def preflight(state: InfraState) -> dict[str, Any]:
        calls = Plan.model_validate(state["plan"]).calls
        if not any(c.tool in _NEEDS_EXISTING for c in calls):
            return {}
        listing = await invoker.call("list_containers", {})
        if not listing.get("ok"):
            return {"error": f"could not check targets: {listing.get('error')}"}
        containers = listing["containers"]
        names = {c["name"] for c in containers}
        services = {c.get("service") for c in containers}
        for call in calls:  # replay the plan so earlier creates satisfy later calls
            name = call.args.get("name")
            if call.tool == "create_container":
                names.add(name)
                services.add(name)
            elif call.tool in ("restart_container", "delete_container") and name not in names:
                return {"error": _missing(name, sorted(names))}
            elif call.tool == "scale_service" and name not in services:
                return {"error": _missing(name, sorted(s for s in services if s))}
        return {}

    return preflight


def _missing(name: Any, existing: list[str]) -> str:
    have = ", ".join(existing) if existing else "none"
    return (
        f"'{name}' does not exist in the sandbox (managed containers: {have}). "
        "Create it first or give the exact container name."
    )
