from collections.abc import Awaitable, Callable
from typing import Any

from infrapilot.mcp_client import ToolInvoker
from infrapilot.state import InfraState


def _check(call: dict[str, Any], containers: list[dict[str, Any]]) -> str | None:
    names = {c["name"] for c in containers}
    args = call["args"]
    match call["tool"]:
        case "create_container" if args["name"] not in names:
            return f"{args['name']} not found after creation"
        case "delete_container" if args["name"] in names:
            return f"{args['name']} still present after deletion"
        case "scale_service":
            count = sum(1 for c in containers if c.get("service") == args["name"])
            if count != args["replicas"]:
                return f"{args['name']} has {count} replicas, expected {args['replicas']}"
    return None


def make_verify(invoker: ToolInvoker) -> Callable[[InfraState], Awaitable[dict[str, Any]]]:
    async def verify(state: InfraState) -> dict[str, Any]:
        results = state.get("results", [])
        problems = [f"{r['tool']} failed: {r.get('error')}" for r in results if not r.get("ok")]
        if not problems:
            listing = await invoker.call("list_containers", {})
            if not listing.get("ok"):
                problems.append(f"could not verify: {listing.get('error')}")
            else:
                for r in results:
                    if problem := _check(r, listing["containers"]):
                        problems.append(problem)
        ok = not problems
        lines = "\n".join(
            f"- {r['tool']}({', '.join(f'{k}={v!r}' for k, v in r['args'].items())}): "
            + ("ok" if r.get("ok") else f"FAILED ({r.get('error')})")
            for r in results
        )
        summary = (
            f"Executed {len(results)} call(s) and verified the resulting state.\n{lines}"
            if ok
            else "Execution finished with problems: " + "; ".join(problems)
        )
        if ok and results and results[-1]["tool"] == "list_containers":
            summary += "\n" + "\n".join(
                f"- {c['name']} ({c['image']}, {c['status']})" for c in results[-1]["containers"]
            )
        return {"verification": {"ok": ok, "problems": problems}, "final": summary}

    return verify
