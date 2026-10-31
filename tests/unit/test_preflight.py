import asyncio

from infrapilot.graph.preflight import make_preflight
from tests.fakes import FakeInvoker


def run(calls: list[dict], existing: tuple[str, ...] = ()) -> dict:
    inv = FakeInvoker()
    for n in existing:
        inv.containers[n] = {"name": n, "image": "nginx", "status": "running", "service": n}
    state = {"plan": {"rationale": "t", "calls": calls}}
    return asyncio.run(make_preflight(inv)(state))  # type: ignore[arg-type]


def test_restart_of_missing_container_is_stopped_before_approval():
    out = run([{"tool": "restart_container", "args": {"name": "ip-nginx"}}], existing=("ip-web",))
    assert "does not exist" in out["error"] and "ip-web" in out["error"]


def test_restart_of_existing_container_passes():
    assert run([{"tool": "restart_container", "args": {"name": "ip-web"}}], ("ip-web",)) == {}


def test_create_then_restart_in_same_plan_passes():
    calls = [
        {
            "tool": "create_container",
            "args": {"image": "nginx", "name": "ip-a", "mem_mb": 64, "cpus": 0.5},
        },
        {"tool": "restart_container", "args": {"name": "ip-a"}},
    ]
    assert run(calls) == {}


def test_scale_of_unknown_service_is_stopped():
    assert "error" in run([{"tool": "scale_service", "args": {"name": "ip-x", "replicas": 2}}])
