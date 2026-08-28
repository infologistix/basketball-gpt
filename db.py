from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import Connection
from psycopg import sql


DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://nba:nba@localhost:5432/nba",
)
DEFAULT_SCHEMA = os.getenv("NBA_DB_SCHEMA", "bronze")
DEFAULT_SCHEMAS = os.getenv("NBA_DB_SCHEMAS", DEFAULT_SCHEMA)
SCHEMA_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_schema_list(value: str | None) -> list[str]:
    """Parse, validate, and deduplicate a comma-separated schema list."""
    raw = value or DEFAULT_SCHEMAS
    schemas = [schema.strip() for schema in raw.split(",") if schema.strip()]
    schemas = schemas or [DEFAULT_SCHEMA]
    invalid = [schema for schema in schemas if not SCHEMA_NAME_PATTERN.fullmatch(schema)]
    if invalid:
        raise ValueError(f"Invalid PostgreSQL schema name(s): {', '.join(invalid)}")
    return list(dict.fromkeys(schemas))


def get_schemas() -> list[str]:
    """Return the configured PostgreSQL schemas in search-path order."""
    return parse_schema_list(os.getenv("NBA_DB_SCHEMAS") or os.getenv("NBA_DB_SCHEMA") or DEFAULT_SCHEMAS)


def get_default_schema() -> str:
    """Return the first configured schema used for unqualified table prompts."""
    return get_schemas()[0]


def get_database_url() -> str:
    """Return the active PostgreSQL connection URL."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_db_label() -> str:
    """Return a display-safe database label for the Streamlit sidebar."""
    schemas = ",".join(get_schemas())
    return f"{safe_database_url(get_database_url())} schemas={schemas}"


def get_database_name() -> str:
    """Return just the database name, for a compact sidebar status line."""
    try:
        path = urlsplit(get_database_url()).path
    except ValueError:
        return "unknown"
    return path.lstrip("/") or "unknown"


def safe_database_url(database_url: str) -> str:
    """Return a display-safe database URL with any password removed."""
    try:
        parsed = urlsplit(database_url)
    except ValueError:
        return "(invalid database url)"

    if not parsed.password:
        return database_url
    username = parsed.username or ""
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    auth = f"{username}:***@" if username else ""
    return urlunsplit((parsed.scheme, f"{auth}{hostname}{port}", parsed.path, parsed.query, parsed.fragment))


@contextmanager
def connect(database_url: str | None = None) -> Iterator[Connection]:
    """Open a PostgreSQL connection with the configured schema search path."""
    schemas = get_schemas()
    conn = psycopg.connect(database_url or get_database_url())
    try:
        # Search path keeps legacy unqualified table prompts working while the
        # query engine still prefers explicit schema-qualified table names.
        identifiers = [sql.Identifier(schema) for schema in schemas] + [sql.Identifier("public")]
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.SQL(", ").join(identifiers)))
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
