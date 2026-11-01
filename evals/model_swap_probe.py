"""Measure what OLLAMA_MAX_LOADED_MODELS=1 costs: alternate chat and embedding calls and time them.

Output is saved by hand to evals/results/model-swap-<date>.txt. Run with the same settings as the
evals (NUM_CTX, keep_alive) and nothing else using the GPU.
"""

import subprocess
import time
from pathlib import Path

from infrapilot.config import get_settings
from infrapilot.llm import get_chat_model, get_embeddings


def resident() -> str:
    lines = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.splitlines()
    return ",".join(ln.split()[0] for ln in lines[1:]) or "none"


def timed(fn: object, *args: str) -> float:
    start = time.perf_counter()
    fn(*args)  # type: ignore[operator]
    return time.perf_counter() - start


def main() -> None:
    s = get_settings()
    llm, emb = get_chat_model(s), get_embeddings(s)
    print(f"chat={s.llm_model} embed={s.embed_model} num_ctx={s.num_ctx}")
    server = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True, text=True)
    pid = server.stdout.split()[0] if server.stdout.split() else ""
    if pid:
        env = Path(f"/proc/{pid}/environ").read_text().split("\0")
        print("server env:", [e for e in env if e.startswith("OLLAMA_MAX_LOADED")])
    for i in range(1, 4):
        a = timed(llm.invoke, "ping")
        print(f"round {i}: chat call {a:5.1f}s  resident after: {resident()}")
        b = timed(emb.embed_query, "memory limits")
        print(f"round {i}: embed call {b:5.1f}s  resident after: {resident()}")


if __name__ == "__main__":
    main()
