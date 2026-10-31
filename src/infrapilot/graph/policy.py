"""Deterministic policy guard. Pure Python: the LLM never decides what is allowed."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from infrapilot.state import InfraState, Plan, PlannedCall, PolicyDecision, plan_hash

MAX_CALLS_PER_PLAN = 5
MAX_MEM_MB = 1024
MAX_CPUS = 2.0
MAX_REPLICAS = 5
ALLOWED_IMAGES = frozenset({"nginx", "redis", "postgres", "alpine", "busybox", "python"})
NAME_PATTERN = r"^ip-[a-z0-9][a-z0-9-]{1,40}$"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListContainersArgs(_Args):
    pass


class CreateContainerArgs(_Args):
    image: str
    name: str = Field(pattern=NAME_PATTERN)
    mem_mb: int = Field(ge=16, le=MAX_MEM_MB)
    cpus: float = Field(gt=0, le=MAX_CPUS)


class ScaleServiceArgs(_Args):
    name: str = Field(pattern=NAME_PATTERN)
    replicas: int = Field(ge=0, le=MAX_REPLICAS)


class RestartContainerArgs(_Args):
    name: str = Field(pattern=NAME_PATTERN)


class DeleteContainerArgs(_Args):
    name: str = Field(pattern=NAME_PATTERN)


class ToolSpec(BaseModel):
    name: str
    mutating: bool
    destructive: bool
    signature: str


TOOLS: dict[str, tuple[ToolSpec, type[_Args]]] = {
    "list_containers": (
        ToolSpec(
            name="list_containers",
            mutating=False,
            destructive=False,
            signature="list_containers()",
        ),
        ListContainersArgs,
    ),
    "create_container": (
        ToolSpec(
            name="create_container",
            mutating=True,
            destructive=False,
            signature="create_container(image, name, mem_mb, cpus)",
        ),
        CreateContainerArgs,
    ),
    "scale_service": (
        ToolSpec(
            name="scale_service",
            mutating=True,
            destructive=False,
            signature="scale_service(name, replicas)",
        ),
        ScaleServiceArgs,
    ),
    "restart_container": (
        ToolSpec(
            name="restart_container",
            mutating=True,
            destructive=False,
            signature="restart_container(name)",
        ),
        RestartContainerArgs,
    ),
    "delete_container": (
        ToolSpec(
            name="delete_container",
            mutating=True,
            destructive=True,
            signature="delete_container(name)",
        ),
        DeleteContainerArgs,
    ),
}


def _check_call(call: PlannedCall) -> tuple[list[str], bool, bool]:
    """Return (violations, mutating, destructive) for one call."""
    entry = TOOLS.get(call.tool)
    if entry is None:
        return [f"unknown tool '{call.tool}'"], False, False
    spec, args_model = entry
    violations: list[str] = []
    try:
        args = args_model.model_validate(call.args)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "args"
            violations.append(f"{call.tool}: {loc}: {err['msg']}")
        return violations, spec.mutating, spec.destructive
    if isinstance(args, CreateContainerArgs) and args.image.split(":")[0] not in ALLOWED_IMAGES:
        violations.append(f"{call.tool}: image '{args.image}' is not allowlisted")
    # Destructive means data loss: deletes and scaling a service to zero.
    destructive = spec.destructive or (isinstance(args, ScaleServiceArgs) and args.replicas == 0)
    return violations, spec.mutating, destructive


def evaluate_plan(plan: Plan) -> PolicyDecision:
    digest = plan_hash(plan.calls)
    if len(plan.calls) > MAX_CALLS_PER_PLAN:
        return PolicyDecision(
            outcome="refuse",
            reasons=[f"plan has {len(plan.calls)} calls, limit is {MAX_CALLS_PER_PLAN}"],
            plan_hash=digest,
        )
    reasons: list[str] = []
    mutating = destructive = False
    for call in plan.calls:
        violations, is_mut, is_destr = _check_call(call)
        reasons += violations
        mutating |= is_mut
        destructive |= is_destr
    if reasons:
        return PolicyDecision(outcome="refuse", reasons=reasons, plan_hash=digest)
    if mutating:
        note = (
            "destructive operation requires approval"
            if destructive
            else ("mutating operation requires approval")
        )
        return PolicyDecision(
            outcome="needs_approval",
            reasons=[note],
            destructive=destructive,
            plan_hash=digest,
        )
    return PolicyDecision(outcome="approve", reasons=["read-only plan"], plan_hash=digest)


def policy_guard(state: InfraState) -> dict[str, Any]:
    return {"policy": evaluate_plan(Plan.model_validate(state["plan"])).model_dump()}
