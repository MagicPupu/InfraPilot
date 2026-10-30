from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from infrapilot.graph.policy import TOOLS
from infrapilot.state import InfraState, Plan
from infrapilot.structured import StructuredOutputError, invoke_structured

_TOOL_LIST = "\n".join(f"- {spec.signature}" for spec, _ in TOOLS.values())

SYSTEM = f"""You plan infrastructure changes as an ordered list of tool calls.
Available tools:
{_TOOL_LIST}
Rules: container names start with "ip-"; mem_mb is an integer in MB; cpus is a number.
Only plan what the user asked for. If the request cannot be done with these tools,
still return your best plan using the closest tool name; a separate policy check decides.
Reply with JSON: {{"rationale": str, "calls": [{{"tool": str, "args": object}}]}}.

Examples:
"Start redis as ip-cache with 128 MB and half a CPU" ->
{{"rationale": "Create one redis container.", "calls": [{{"tool": "create_container",
"args": {{"image": "redis:7", "name": "ip-cache", "mem_mb": 128, "cpus": 0.5}}}}]}}
"Scale ip-web to 3 replicas" ->
{{"rationale": "Scale the service.", "calls": [{{"tool": "scale_service",
"args": {{"name": "ip-web", "replicas": 3}}}}]}}
"Show me what is running" ->
{{"rationale": "List managed containers.", "calls": [{{"tool": "list_containers", "args": {{}}}}]}}"""


def make_planner(llm: BaseChatModel) -> Callable[[InfraState], dict[str, Any]]:
    def planner(state: InfraState) -> dict[str, Any]:
        messages = [SystemMessage(content=SYSTEM), HumanMessage(content=state["request"])]
        try:
            plan = invoke_structured(llm, messages, Plan)
        except StructuredOutputError as exc:
            return {"error": str(exc)}
        return {
            "plan": plan.model_dump(),
            "policy": {},
            "approval": {},
            "results": [],
            "verification": {},
        }

    return planner
