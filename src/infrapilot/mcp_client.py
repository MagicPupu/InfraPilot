"""Client side of the MCP sandbox server (langchain-mcp-adapters)."""

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

from langchain_mcp_adapters.client import MultiServerMCPClient


class ToolInvoker(Protocol):
    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]: ...


def _parse(raw: Any) -> dict[str, Any]:
    """Adapters return a string or a list of content blocks depending on version."""
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"unparseable tool output: {raw!r}"}
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": "unexpected output"}


class McpToolInvoker:
    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = tools

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool not in self._tools:
            return {"ok": False, "error": f"tool '{tool}' not exposed by MCP server"}
        return _parse(await self._tools[tool].ainvoke(args))


@asynccontextmanager
async def open_sandbox_invoker() -> AsyncIterator[McpToolInvoker]:
    client = MultiServerMCPClient(
        {
            "sandbox": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "infrapilot.mcp_server.server"],
            }
        }
    )
    async with client.session("sandbox") as session:
        from langchain_mcp_adapters.tools import load_mcp_tools

        tools = await load_mcp_tools(session)
        yield McpToolInvoker({t.name: t for t in tools})
