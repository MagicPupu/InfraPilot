import asyncio
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import typer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel

from infrapilot.config import get_settings
from infrapilot.graph.build import Deps, build_graph
from infrapilot.llm import get_chat_model, get_embeddings
from infrapilot.mcp_client import open_sandbox_invoker
from infrapilot.observability import get_callbacks
from infrapilot.rag.ingest import ingest_directory
from infrapilot.rag.retrieve import PgVectorRetriever

app = typer.Typer(
    help="InfraPilot: a local, policy-gated infrastructure copilot.", no_args_is_help=True
)
console = Console()


async def _ask(request: str, auto_approve: bool) -> None:
    s = get_settings()
    async with AsyncPostgresSaver.from_conn_string(s.database_url) as saver:
        await saver.setup()
        async with open_sandbox_invoker() as invoker:
            deps = Deps(
                get_chat_model(s), PgVectorRetriever(s.database_url, get_embeddings(s)), invoker
            )
            graph = build_graph(deps, saver)
            config: Any = {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "callbacks": get_callbacks(s),
            }
            result = await graph.ainvoke({"request": request}, config)
            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                console.print(
                    Panel(payload["preview"], title="Approval required", border_style="yellow")
                )
                approved = auto_approve or typer.confirm("Approve this plan?", default=False)
                result = await graph.ainvoke(Command(resume={"approved": approved}), config)
            console.print(Panel(result.get("final", "(no answer)"), title="InfraPilot"))


@app.command()
def ask(
    request: str = typer.Argument(..., help="Natural-language request."),
    yes: bool = typer.Option(False, "--yes", help="Auto-approve (policy still applies)."),
) -> None:
    """Ask a question or request an infrastructure change."""
    asyncio.run(_ask(request, yes))


@app.command()
def ingest(
    directory: Path = typer.Argument(Path("docs/runbooks"), exists=True, file_okay=False),
) -> None:
    """Chunk, embed and store markdown runbooks in pgvector."""
    s = get_settings()
    count = ingest_directory(directory, s.database_url, get_embeddings(s))
    console.print(f"Ingested {count} chunks from {directory}")


@app.command(
    name="eval", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def eval_(ctx: typer.Context) -> None:
    """Run the local deterministic evals (forwards extra args, e.g. --models a,b)."""
    script = Path(__file__).resolve().parents[2] / "evals" / "run_evals.py"
    raise typer.Exit(subprocess.run([sys.executable, str(script), *ctx.args]).returncode)
