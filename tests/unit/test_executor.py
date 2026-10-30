import asyncio

from infrapilot.graph.executor import authorize, make_executor
from infrapilot.graph.policy import evaluate_plan
from infrapilot.state import Approval, Plan, PlannedCall
from tests.fakes import FakeInvoker

DELETE = Plan(rationale="r", calls=[PlannedCall(tool="delete_container", args={"name": "ip-web"})])


def state_for(plan: Plan, approved: bool | None, hash_override: str | None = None) -> dict:
    policy = evaluate_plan(plan)
    s: dict = {"plan": plan.model_dump(), "policy": policy.model_dump(), "approval": {}}
    if approved is not None:
        s["approval"] = Approval(
            approved=approved, plan_hash=hash_override or policy.plan_hash
        ).model_dump()
    return s


def test_destructive_without_approval_is_blocked():
    assert authorize(state_for(DELETE, None)) == "no recorded approval"


def test_rejected_approval_is_blocked():
    assert authorize(state_for(DELETE, False)) is not None


def test_approval_for_other_plan_is_blocked():
    assert authorize(state_for(DELETE, True, hash_override="deadbeef")) is not None


def test_tampered_plan_after_policy_is_blocked():
    s = state_for(DELETE, True)
    s["plan"]["calls"][0]["args"]["name"] = "ip-other"
    assert authorize(s) == "plan was modified after the policy check"


def test_valid_approval_executes_and_invalid_never_calls_tools():
    invoker = FakeInvoker()
    execute = make_executor(invoker)
    blocked = asyncio.run(execute(state_for(DELETE, None)))
    assert "error" in blocked and not invoker.calls
    ok = asyncio.run(execute(state_for(DELETE, True)))
    assert ok["results"][0]["ok"] and invoker.calls
