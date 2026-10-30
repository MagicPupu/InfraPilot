from pathlib import Path

from infrapilot.graph.rag_nodes import invalid_citations
from infrapilot.rag.ingest import MAX_CHUNK_CHARS, load_chunks, parse_markdown

DOC = """# Limits

Intro text.

## Memory

Max is 1024 MB.

```
# not a heading
```

## CPU

Max is 2 cores.
"""


def test_heading_paths_and_ids():
    chunks = parse_markdown("limits.md", DOC)
    paths = {c.heading_path for c in chunks}
    assert {"Limits", "Limits > Memory", "Limits > CPU"} <= paths
    assert not any("not a heading" in p for p in paths)
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks == parse_markdown("limits.md", DOC)  # ids are stable


def test_long_sections_are_split():
    body = "\n\n".join("word " * 100 for _ in range(10))
    chunks = parse_markdown("big.md", f"# Big\n\n{body}")
    assert len(chunks) > 1 and all(len(c.text) <= MAX_CHUNK_CHARS + 600 for c in chunks)


def test_sample_runbooks_parse():
    chunks = load_chunks(Path("docs/runbooks"))
    assert len(chunks) >= 9 and all(c.text for c in chunks)


def test_citation_check():
    ids = {"a", "b"}
    assert invalid_citations(["a"], ids) == []
    assert invalid_citations(["a", "zzz"], ids) == ["zzz"]
    assert invalid_citations([], ids) == ["<no citation>"]
