"""Postgres fixtures. Tests run in a throwaway schema so the real runbook index is untouched."""

import uuid
from collections.abc import Iterator

import psycopg
import pytest

from infrapilot.config import get_settings


@pytest.fixture
def pg_dsn() -> Iterator[str]:
    base = get_settings().database_url
    try:
        admin = psycopg.connect(base, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("pgvector database not reachable (make up)")
    schema = f"test_{uuid.uuid4().hex[:10]}"
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in base else "?"
    try:
        yield f"{base}{sep}options=-csearch_path%3D{schema},public"
    finally:
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()
