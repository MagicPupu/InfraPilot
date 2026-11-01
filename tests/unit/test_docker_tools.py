from unittest.mock import MagicMock

from docker.errors import NotFound

from infrapilot.mcp_server.docker_tools import MANAGED_LABEL, DockerTools


def tools_with(container: MagicMock | None) -> DockerTools:
    client = MagicMock()
    if container is None:
        client.containers.get.side_effect = NotFound("nope")
    else:
        client.containers.get.return_value = container
    return DockerTools(client)


def test_restart_managed_container():
    c = MagicMock(labels={MANAGED_LABEL: "true"}, status="running")
    assert tools_with(c).restart_container("ip-web")["ok"] is True
    c.restart.assert_called_once()


def test_restart_refuses_unmanaged_container():
    c = MagicMock(labels={})
    result = tools_with(c).restart_container("postgres-prod")
    assert result["ok"] is False
    c.restart.assert_not_called()


def test_restart_unknown_container():
    assert tools_with(None).restart_container("ip-ghost")["ok"] is False


# --- unlabeled containers are never touched ---------------------------------


def test_delete_refuses_unlabeled_container():
    c = MagicMock(labels={}, status="running")
    result = tools_with(c).delete_container("ip-web")
    assert result["ok"] is False and "not managed" in result["error"]
    c.remove.assert_not_called()


def test_delete_refuses_container_with_wrong_label_value():
    c = MagicMock(labels={MANAGED_LABEL: "false"})
    assert tools_with(c).delete_container("ip-web")["ok"] is False
    c.remove.assert_not_called()


def test_restart_unlabeled_container_is_not_restarted():
    c = MagicMock(labels={})
    tools_with(c).restart_container("ip-web")
    c.restart.assert_not_called()


def test_listing_and_scaling_only_query_managed_containers():
    client = MagicMock()
    client.containers.list.return_value = []
    tools = DockerTools(client)
    tools.list_containers()
    assert tools.scale_service("ip-web", 0)["ok"] is False  # nothing managed: nothing removed
    for call in client.containers.list.call_args_list:
        assert f"{MANAGED_LABEL}=true" in call.kwargs["filters"]["label"]


def test_create_does_not_overwrite_an_existing_unlabeled_container():
    client = MagicMock()
    client.containers.get.return_value = MagicMock(labels={})  # same name, not ours
    result = DockerTools(client).create_container("nginx:1.27", "ip-web", 128, 0.5)
    assert result["ok"] is False
    client.containers.run.assert_not_called()
