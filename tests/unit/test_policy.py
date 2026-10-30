import pytest

from infrapilot.graph.policy import evaluate_plan
from infrapilot.state import Plan, PlannedCall, plan_hash


def plan(*calls: tuple[str, dict]) -> Plan:
    return Plan(rationale="t", calls=[PlannedCall(tool=t, args=a) for t, a in calls])


GOOD_CREATE = (
    "create_container",
    {"image": "nginx:1.27", "name": "ip-web", "mem_mb": 256, "cpus": 0.5},
)


def test_read_only_is_auto_approved():
    assert evaluate_plan(plan(("list_containers", {}))).outcome == "approve"


def test_mutation_needs_approval():
    d = evaluate_plan(plan(GOOD_CREATE))
    assert d.outcome == "needs_approval" and not d.destructive


def test_delete_is_destructive_and_needs_approval():
    d = evaluate_plan(plan(("delete_container", {"name": "ip-web"})))
    assert d.outcome == "needs_approval" and d.destructive


def test_scale_to_zero_is_destructive():
    d = evaluate_plan(plan(("scale_service", {"name": "ip-web", "replicas": 0})))
    assert d.destructive


@pytest.mark.parametrize(
    "call",
    [
        ("drop_database", {}),
        (
            "create_container",
            {"image": "evil/miner", "name": "ip-x", "mem_mb": 64, "cpus": 1},
        ),
        (
            "create_container",
            {"image": "nginx", "name": "ip-x", "mem_mb": 4096, "cpus": 1},
        ),
        (
            "create_container",
            {"image": "nginx", "name": "ip-x", "mem_mb": 64, "cpus": 8},
        ),
        (
            "create_container",
            {"image": "nginx", "name": "web", "mem_mb": 64, "cpus": 1},
        ),
        (
            "create_container",
            {
                "image": "nginx",
                "name": "ip-x",
                "mem_mb": 64,
                "cpus": 1,
                "privileged": True,
            },
        ),
        ("scale_service", {"name": "ip-web", "replicas": 50}),
        ("delete_container", {"name": "production-db"}),
    ],
)
def test_violations_are_refused(call):
    d = evaluate_plan(plan(call))
    assert d.outcome == "refuse" and d.reasons


def test_one_bad_call_refuses_whole_plan():
    d = evaluate_plan(plan(GOOD_CREATE, ("nope", {})))
    assert d.outcome == "refuse"


def test_too_many_calls_refused():
    assert evaluate_plan(plan(*[("list_containers", {})] * 6)).outcome == "refuse"


def test_hash_is_stable_and_sensitive():
    a, b = plan(GOOD_CREATE), plan(GOOD_CREATE)
    assert plan_hash(a.calls) == plan_hash(b.calls) == evaluate_plan(a).plan_hash
    changed = plan(("create_container", {**GOOD_CREATE[1], "mem_mb": 512}))
    assert plan_hash(changed.calls) != plan_hash(a.calls)
