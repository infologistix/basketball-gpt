from __future__ import annotations

import os
import sys

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


def parse_schemas(value: str) -> list[str]:
    """Parse the comma-separated schema list to verify."""
    schemas = [schema.strip() for schema in value.split(",") if schema.strip()]
    if not schemas:
        raise SystemExit("Set SCHEMAS to at least one schema name.")
    return schemas


def schema_counts(database_url: str, schemas: list[str]) -> dict[tuple[str, str], int]:
    """Return row counts for every base table in the selected schemas."""
    counts: dict[tuple[str, str], int] = {}
    with psycopg.connect(database_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema = ANY(%s) AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name
                """,
                (schemas,),
            )
            for row in cur.fetchall():
                table_schema = row["table_schema"]
                table_name = row["table_name"]
                cur.execute(
                    sql.SQL("SELECT count(*) AS row_count FROM {}.{}").format(
                        sql.Identifier(table_schema),
                        sql.Identifier(table_name),
                    )
                )
                counts[(table_schema, table_name)] = cur.fetchone()["row_count"]
    return counts


def main() -> int:
    """Compare source and target table counts and print CSV-style results."""
    source_url = os.environ.get("SOURCE_DATABASE_URL")
    target_url = os.environ.get("DATABASE_URL")
    if not source_url:
        raise SystemExit("Set SOURCE_DATABASE_URL.")
    if not target_url:
        raise SystemExit("Set DATABASE_URL.")

    schemas = parse_schemas(os.environ.get("SCHEMAS", "silver,gold"))
    source_counts = schema_counts(source_url, schemas)
    target_counts = schema_counts(target_url, schemas)
    keys = sorted(set(source_counts) | set(target_counts))

    print("schema,table,source_rows,local_rows,status")
    mismatches = 0
    for schema, table in keys:
        source_rows = source_counts.get((schema, table))
        local_rows = target_counts.get((schema, table))
        status = "OK" if source_rows == local_rows else "MISMATCH"
        if status != "OK":
            mismatches += 1
        print(f"{schema},{table},{source_rows},{local_rows},{status}")

    print(f"total_tables={len(keys)} mismatches={mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
