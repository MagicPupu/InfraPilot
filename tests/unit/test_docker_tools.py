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
