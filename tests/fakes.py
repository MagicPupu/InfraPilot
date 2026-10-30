"""Test doubles. No network, no Ollama, no Docker."""

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from infrapilot.state import Chunk


def scripted_llm(*outputs: Any) -> GenericFakeChatModel:
    """A FakeLLM that replays the given outputs (dicts are JSON-encoded) in call order."""
    texts = [o if isinstance(o, str) else json.dumps(o) for o in outputs]
    return GenericFakeChatModel(messages=iter(AIMessage(content=t) for t in texts))


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        self.queries.append(query)
        return self.chunks


class FakeInvoker:
    """In-memory stand-in for the MCP sandbox server."""

    def __init__(self) -> None:
        self.containers: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, args))
        if tool == "list_containers":
            return {"ok": True, "containers": list(self.containers.values())}
        if tool == "create_container":
            self.containers[args["name"]] = {
                "name": args["name"],
                "image": args["image"],
                "status": "running",
                "service": args["name"],
            }
            return {"ok": True, "name": args["name"]}
        if tool == "delete_container":
            self.containers.pop(args["name"], None)
            return {"ok": True, "deleted": args["name"]}
        return {"ok": False, "error": f"unsupported in fake: {tool}"}
