"""FastMCP server exposing the sandbox tools over stdio.

Run with: ``python -m infrapilot.mcp_server.server``
"""

import json

from mcp.server.fastmcp import FastMCP

from infrapilot.mcp_server.docker_tools import DockerTools

mcp = FastMCP("infrapilot-sandbox")
_tools = DockerTools()


@mcp.tool()
def list_containers() -> str:
    """List containers managed by InfraPilot."""
    return json.dumps(_tools.list_containers())


@mcp.tool()
def create_container(image: str, name: str, mem_mb: int, cpus: float) -> str:
    """Create and start a managed container with memory (MB) and CPU limits."""
    return json.dumps(_tools.create_container(image, name, mem_mb, cpus))


@mcp.tool()
def scale_service(name: str, replicas: int) -> str:
    """Scale a managed service to the given number of replicas."""
    return json.dumps(_tools.scale_service(name, replicas))


@mcp.tool()
def restart_container(name: str) -> str:
    """Restart a managed container. Refuses containers without the managed label."""
    return json.dumps(_tools.restart_container(name))


@mcp.tool()
def delete_container(name: str) -> str:
    """Delete a managed container. Refuses containers without the managed label."""
    return json.dumps(_tools.delete_container(name))


if __name__ == "__main__":
    mcp.run()
