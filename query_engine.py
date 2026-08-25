from __future__ import annotations

import os
import re
import json
import urllib.error
import urllib.request
from decimal import Decimal
from typing import Any

import google.generativeai as genai
from psycopg import sql as pg_sql
from psycopg import Error as PostgresError
from psycopg.rows import dict_row

from db import connect, get_database_url, get_default_schema, get_schemas
from lightweight_rag import format_rejected_context, format_retrieved_context

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/api")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-coder:480b-cloud")
# In-cluster LiteLLM proxy (OpenAI-compatible), same pattern as the sibling "datachat" app.
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://litellm.litellm.svc.cluster.local:4000/v1")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3.6-35b-a3b-coder")
MAX_QUERY_ROWS = int(os.getenv("MAX_QUERY_ROWS", "5000"))

SCHEMA_PROMPT_TEMPLATE = """
You are writing PostgreSQL SELECT queries for a basketball analytics database.

Schemas: {schemas}

Domain notes:
- Table prefixes indicate competitions: b_el = EuroLeague, b_ec = EuroCup, b_cl = Champions League, b_bbl = Basketball Bundesliga.
- boxscore tables are usually best for player/team game totals and rankings.
- playbyplay tables are usually best for event sequences, shot/action timing, and possession-level questions.
- player_info tables are usually best for player lookup and roster attributes.
- wurfposition means shot position/location.

Table relationships:
{relationships}

Natural-language aliases:
- points, score, total points, scored = pts
- player, athlete = player_name
- team, club = player_team when using boxscore/player rows
- opponent = opponent when that column exists
- minutes played = minutes
- game, match = game_id or link depending on available columns
- image/logo = image_url or image columns when present
- shot location, shot position = wurfposition tables

Available tables and columns:
{tables}

Rules:
- Return exactly one PostgreSQL SELECT statement.
- Use only tables from these schemas: {schemas}. Do not query the public schema unless metadata requires information_schema.
- Prefer schema-qualified table names, for example {default_schema}.table_name.
- Do not include markdown, comments, explanations, semicolons, INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, DETACH, PRAGMA, or CREATE.
- Prefer explicit joins and readable aliases.
- Add a LIMIT when returning row lists.
- For broad sample/list requests, use LIMIT 100 or less.
- Only prepare chart-style grouped/ranked data when the intent is draw. For table/schema/sample requests, return normal row or metadata results and do not force aggregation.
- For charts that use the minutes column, convert MM:SS text to decimal minutes with split_part(minutes, ':', 1)::numeric + split_part(minutes, ':', 2)::numeric / 60.0.
- For pts vs minutes scatter charts, filter minutes with minutes ~ '^[0-9]{{1,3}}:[0-9]{{2}}$' and exclude minutes = '00:00' unless the user asks for zero-minute rows.
- For top/ranking charts over positive statistics such as pts, prefer meaningful non-zero rows. If sorting ascending by an aggregate like SUM(pts), add a HAVING clause that removes zero totals unless the user explicitly asks for zero-value rows.
""".strip()

UNSAFE_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|pragma|create|replace|vacuum|reindex)\b",
    re.IGNORECASE,
)
SCHEMA_COLUMNS_SQL = """
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = ANY(%s)
ORDER BY table_schema, table_name, ordinal_position
""".strip()
LIST_CONFIGURED_TABLES_SQL = """
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema = ANY(%s) AND table_type = 'BASE TABLE'
ORDER BY table_schema, table_name
LIMIT %s
""".strip()
COLUMNS_FOR_TABLE_SQL = """
SELECT table_schema, table_name, column_name, data_type, ordinal_position
FROM information_schema.columns
WHERE table_schema = %s AND table_name = %s
ORDER BY ordinal_position
""".strip()
COLUMN_COUNT_FOR_TABLE_SQL = """
SELECT table_schema, table_name, count(*)::integer AS column_count
FROM information_schema.columns
WHERE table_schema = %s AND table_name = %s
GROUP BY table_schema, table_name
""".strip()


class QueryEngineError(Exception):
    pass


class MissingApiKeyError(QueryEngineError):
    pass


class LlmProviderError(QueryEngineError):
    pass


class UnsafeSqlError(QueryEngineError):
    pass


def get_db_path() -> str:
    """Return the configured database URL for legacy call sites."""
    return get_database_url()


def configure_gemini(api_key: str | None) -> None:
    """Configure the Gemini client or raise when no key is available."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise MissingApiKeyError("Gemini API key is required. Add it in the sidebar or set GEMINI_API_KEY.")
    genai.configure(api_key=key)


def _gemini_model(model_name: str | None = None) -> genai.GenerativeModel:
    """Create a Gemini model instance using the configured default model."""
    return genai.GenerativeModel(model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))


def _ollama_generate(
    prompt: str,
    system: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Call an Ollama-compatible generate endpoint and return plain text."""
    root = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).rstrip("/")
    endpoint = f"{root}/generate" if root.endswith("/api") else f"{root}/api/generate"
    key = api_key or os.getenv("OLLAMA_API_KEY")

    if "ollama.com" in root and not key:
        raise MissingApiKeyError("Ollama cloud requires an API key. Local Ollama does not.")

    payload = {
        "model": model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LlmProviderError(f"Ollama request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LlmProviderError("Ollama returned invalid JSON.") from exc

    if "error" in data:
        raise LlmProviderError(f"Ollama error: {data['error']}")
    return str(data.get("response", "")).strip()


def _openai_generate(
    prompt: str,
    system: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Call an OpenAI-compatible chat endpoint and return plain text.

    Used for the in-cluster LiteLLM proxy, which speaks the OpenAI protocol rather
    than Ollama's. Base URL already includes the version prefix, e.g.
    http://litellm.litellm.svc.cluster.local:4000/v1
    """
    root = (base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)).rstrip("/")
    endpoint = f"{root}/chat/completions"
    key = api_key or os.getenv("OPENAI_API_KEY")

    if not key:
        raise MissingApiKeyError(
            "An API key is required. Add it in the sidebar or set OPENAI_API_KEY "
            "(for LiteLLM this is a virtual key)."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        "messages": messages,
        "stream": False,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LlmProviderError(f"OpenAI-compatible request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LlmProviderError("OpenAI-compatible endpoint returned invalid JSON.") from exc

    if "error" in data:
        raise LlmProviderError(f"OpenAI-compatible error: {data['error']}")

    choices = data.get("choices") or []
    if not choices:
        raise LlmProviderError("OpenAI-compatible endpoint returned no choices.")
    return str(choices[0].get("message", {}).get("content", "")).strip()


def generate_text(
    prompt: str,
    system: str | None = None,
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Generate text with the selected LLM provider."""
    selected = (provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
    if selected == "gemini":
        configure_gemini(gemini_api_key)
        parts = [prompt] if system is None else [system, prompt]
        response = _gemini_model(model_name).generate_content(parts)
        return (response.text or "").strip()
    if selected == "ollama":
        return _ollama_generate(
            prompt=prompt,
            system=system,
            model_name=model_name,
            base_url=ollama_base_url,
            api_key=ollama_api_key,
        )
    if selected == "openai":
        return _openai_generate(
            prompt=prompt,
            system=system,
            model_name=model_name,
            base_url=openai_base_url,
            api_key=openai_api_key,
        )
    raise LlmProviderError(f"Unsupported LLM provider: {selected}")


def extract_sql(text: str) -> str:
    """Extract a single SQL statement from raw model output."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = cleaned.rstrip(";").strip()
    return cleaned


def validate_sql(sql: str) -> str:
    """Validate that SQL is a single read-only SELECT statement."""
    normalized = sql.strip()
    if not normalized:
        raise UnsafeSqlError("The LLM returned an empty SQL statement.")
    if not re.match(r"^select\b", normalized, flags=re.IGNORECASE):
        raise UnsafeSqlError("Only SELECT statements are allowed.")
    if ";" in normalized:
        raise UnsafeSqlError("Multiple SQL statements are not allowed.")
    if UNSAFE_SQL_PATTERN.search(normalized):
        raise UnsafeSqlError("The generated SQL contains a blocked keyword.")
    return normalized


def normalize_value(value: Any) -> Any:
    """Convert database-native values into UI-friendly Python values."""
    if isinstance(value, Decimal):
        return float(value)
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize every value in a database result row."""
    return {key: normalize_value(value) for key, value in row.items()}


def improve_sql_for_question(sql: str, question: str) -> str:
    """Apply deterministic SQL improvements for known prompt patterns."""
    normalized_question = question.lower()
    improved = sql
    if (
        "ascending" in normalized_question
        and "top" in normalized_question
        and "pts" in normalized_question
        and "sum(pts)" in improved.lower()
        and "having" not in improved.lower()
        and "zero" not in normalized_question
    ):
        improved = re.sub(
            r"(\bGROUP\s+BY\b.+?)(\bORDER\s+BY\b)",
            r"\1 HAVING SUM(pts) > 0 \2",
            improved,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return validate_sql(improved)


def excluded_zero_totals(sql: str, question: str) -> bool:
    """Return whether ascending top-point SQL excluded zero totals."""
    normalized_question = question.lower()
    normalized_sql = sql.lower()
    return (
        "ascending" in normalized_question
        and "top" in normalized_question
        and "sum(pts)" in normalized_sql
        and "having sum(pts) > 0" in normalized_sql
        and "zero" not in normalized_question
    )


def classify_question(question: str) -> str:
    """Classify a prompt into the app's supported response intents."""
    normalized = question.lower()
    if has_draw_visualization_intent(question):
        return "draw"
    if "sql" in normalized and any(word in normalized for word in ("debug", "fix", "explain", "why", "error")):
        return "debug_sql"
    if any(phrase in normalized for phrase in ("list me all tables", "list all tables", "show tables")):
        return "schema"
    if any(word in normalized for word in ("schema", "column", "columns", "tables")):
        return "schema"
    if any(
        phrase in normalized
        for phrase in (
            "sample row",
            "sample rows",
            "show rows",
            "show me rows",
            "preview",
            "first rows",
            "raw rows",
        )
    ):
        return "table"
    return "answer"


def has_draw_visualization_intent(question: str | None) -> bool:
    """Return whether a prompt explicitly asks for a visualization."""
    if not question:
        return False
    normalized = question.lower()
    draw_words = ("draw", "draws", "drawing", "draw me")
    visualization_words = (
        "chart",
        "diagram",
        "visualization",
        "visualize",
        "plot",
        "graph",
        "bar chart",
        "line chart",
        "scatter",
        "scatter plot",
        "histogram",
    )
    return any(word in normalized for word in draw_words + visualization_words)


def get_schema_prompt(db_path: str | None = None) -> str:
    """Build the LLM system prompt from live database schema metadata."""
    schemas = get_schemas()
    default_schema = get_default_schema()
    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(SCHEMA_COLUMNS_SQL, (schemas,))
                rows = [dict(row) for row in cur.fetchall()]
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise QueryEngineError(f"PostgreSQL error while reading schema: {message}") from exc

    if not rows:
        table_text = "(No tables found.)"
    else:
        table_columns: dict[str, list[str]] = {}
        for row in rows:
            qualified_name = f"{row['table_schema']}.{row['table_name']}"
            table_columns.setdefault(qualified_name, []).append(
                f"{row['column_name']} {row['data_type']}"
            )
        table_text = "\n".join(
            f"- {table}({', '.join(columns)})"
            for table, columns in table_columns.items()
        )

    relationships = build_relationship_notes(rows)
    return SCHEMA_PROMPT_TEMPLATE.format(
        schemas=", ".join(schemas),
        default_schema=default_schema,
        relationships=relationships,
        tables=table_text,
    )


def build_relationship_notes(rows: list[dict[str, Any]]) -> str:
    """Build domain relationship notes from discovered table names."""
    if not rows:
        return "- No table relationship hints are available because the schema is empty."

    table_columns: dict[str, set[str]] = {}
    for row in rows:
        table_columns.setdefault(f"{row['table_schema']}.{row['table_name']}", set()).add(row["column_name"])

    competition_names = {
        "b_el": "EuroLeague",
        "b_ec": "EuroCup",
        "b_cl": "Champions League",
        "b_bbl": "Basketball Bundesliga",
    }
    notes = []
    for prefix, competition in competition_names.items():
        family = sorted(table for table in table_columns if table.split(".", 1)[-1].startswith(f"{prefix}_"))
        if not family:
            continue
        notes.append(f"- {prefix}_* tables belong to {competition}: {', '.join(family)}.")

    notes.extend(
        [
            "- Within the same competition prefix, boxscore and playbyplay tables can often be compared by game/link-style columns when those columns exist.",
            "- Use player_name for player-level grouping. Use player_team for team-level grouping in boxscore-style rows.",
            "- Use player_info tables for roster/player metadata, boxscore tables for statistics, playbyplay tables for event logs, and wurfposition tables for shot locations.",
        ]
    )
    return "\n".join(notes)


def intent_instruction(intent: str) -> str:
    """Return prompt guidance for a classified question intent."""
    instructions = {
        "draw": "Intent: draw. Return chart-ready data: usually one label column and one or two numeric/date columns. Use GROUP BY/ORDER BY/LIMIT when useful.",
        "table": "Intent: table. Return raw rows for inspection. Do not aggregate unless the user explicitly asks for counts or totals. Always use a reasonable LIMIT.",
        "schema": "Intent: schema. Return metadata such as table names, columns, data types, or row counts.",
        "debug_sql": "Intent: debug_sql. Return SQL that helps inspect or validate the issue using read-only SELECTs.",
        "answer": "Intent: answer. Return the smallest result set needed to answer the question clearly.",
    }
    return instructions.get(intent, instructions["answer"])


def generate_sql(
    question: str,
    db_path: str | None = None,
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Generate, extract, validate, and lightly improve SQL for a question."""
    intent = classify_question(question)
    retrieved_context = format_retrieved_context(question)
    rejected_context = format_rejected_context(question)
    prompt_parts = [intent_instruction(intent)]
    if retrieved_context:
        prompt_parts.append(retrieved_context)
    if rejected_context:
        prompt_parts.append(rejected_context)
    prompt_parts.append(f"Question: {question}")
    response = generate_text(
        prompt="\n\n".join(prompt_parts),
        system=get_schema_prompt(db_path=db_path),
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    sql = extract_sql(response)
    return improve_sql_for_question(validate_sql(sql), question)


def repair_sql(
    question: str,
    failed_sql: str,
    error: str,
    db_path: str | None = None,
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Ask the LLM to repair a failed SQL statement using the DB error."""
    intent = classify_question(question)
    retrieved_context = format_retrieved_context(question)
    rejected_context = format_rejected_context(question)
    prompt = f"""
The previous PostgreSQL query failed. Rewrite it as one valid, safe SELECT statement.

{intent_instruction(intent)}

{retrieved_context}

{rejected_context}

Question:
{question}

Failed SQL:
{failed_sql}

PostgreSQL error:
{error}
""".strip()
    response = generate_text(
        prompt=prompt,
        system=get_schema_prompt(db_path=db_path),
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    sql = extract_sql(response)
    return validate_sql(sql)


def execute_sql(sql: str, db_path: str | None = None) -> list[dict[str, Any]]:
    """Execute validated SQL and return normalized rows with a row cap."""
    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(MAX_QUERY_ROWS + 1)
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        if "connection" in message.lower():
            raise QueryEngineError("NBA Postgres database was not reachable. Start local Postgres first.") from exc
        raise QueryEngineError(f"PostgreSQL error: {message}") from exc
    if len(rows) > MAX_QUERY_ROWS:
        raise QueryEngineError(
            f"The query returned more than {MAX_QUERY_ROWS:,} rows. "
            "Ask a narrower question or include a LIMIT."
        )
    return [normalize_row(dict(row)) for row in rows]


def answer_schema_question(question: str, db_path: str | None = None) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Answer simple table-list questions without calling the LLM."""
    normalized = question.lower()
    column_answer = answer_column_metadata_question(question, db_path=db_path)
    if column_answer is not None:
        return column_answer

    asks_for_tables = "table" in normalized and any(
        phrase in normalized
        for phrase in ("list", "show", "what", "which", "all")
    )
    if "first table" not in normalized and not asks_for_tables:
        return None

    schemas = get_schemas()
    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(LIST_CONFIGURED_TABLES_SQL, (schemas, 1 if "first table" in normalized else 10000))
                rows = cur.fetchall()
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise QueryEngineError(f"PostgreSQL error: {message}") from exc

    rows = [dict(row) for row in rows]
    if not rows:
        return (
            f"I could not find any tables in the configured PostgreSQL schemas: `{', '.join(schemas)}`.",
            f"SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = ANY({schemas!r}) ORDER BY table_schema, table_name",
            rows,
        )

    rendered_sql = (
        "SELECT table_schema, table_name\n"
        "FROM information_schema.tables\n"
        f"WHERE table_schema = ANY(ARRAY{schemas!r}) AND table_type = 'BASE TABLE'\n"
        "ORDER BY table_schema, table_name"
    )
    if "first table" not in normalized:
        table_list = ", ".join(f"`{row['table_schema']}.{row['table_name']}`" for row in rows)
        return f"The configured schemas contain these tables: {table_list}.", rendered_sql, rows

    table_name = f"{rows[0]['table_schema']}.{rows[0]['table_name']}"
    answer = (
        f"The first table in the configured PostgreSQL schemas is `{table_name}`. "
        f"`{table_name}` is a table name, not one specific team. "
        "If you meant the first row inside that table, ask: "
        "`Which team is the first row in the teams table?`"
    )
    return answer, rendered_sql, rows


def answer_column_metadata_question(
    question: str,
    db_path: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Answer specific column-count and column-list metadata questions."""
    normalized = question.lower()
    if "column" not in normalized and "columns" not in normalized:
        return None

    table_reference = table_name_from_question(question)
    if not table_reference:
        return None

    resolved_table = resolve_table_reference(table_reference, db_path=db_path)
    if not resolved_table:
        return None

    table_schema, table_name = resolved_table
    wants_count = any(phrase in normalized for phrase in ("how many", "count", "number of"))
    query_sql = COLUMN_COUNT_FOR_TABLE_SQL if wants_count else COLUMNS_FOR_TABLE_SQL

    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query_sql, (table_schema, table_name))
                rows = [dict(row) for row in cur.fetchall()]
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise QueryEngineError(f"PostgreSQL error: {message}") from exc

    rendered_sql = (
        query_sql.replace("%s", f"'{table_schema}'", 1)
        .replace("%s", f"'{table_name}'", 1)
    )
    qualified_name = f"{table_schema}.{table_name}"
    if not rows:
        return f"I could not find any columns for `{qualified_name}`.", rendered_sql, rows

    if wants_count:
        count = rows[0]["column_count"]
        return f"`{qualified_name}` has {count:,} columns.", rendered_sql, rows

    return f"Here are the columns in `{qualified_name}`.", rendered_sql, rows


def summarize_answer(
    question: str,
    sql: str,
    rows: list[dict[str, Any]],
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Summarize SQL result rows into a user-facing answer."""
    if not rows:
        return "No data found for that question. Try asking about a narrower table, date range, team, or player."

    intent = classify_question(question)
    if intent == "draw":
        return chart_answer(question, sql, rows)

    if intent in {"table", "schema"}:
        return table_answer(question, sql, rows)

    prompt = f"""
Answer the user's basketball analytics question using only the SQL result rows below.

Question:
{question}

SQL:
{sql}

Rows:
{rows[:100]}

Give a concise, formatted answer. Mention if the result is limited by returned rows or a SQL LIMIT.
""".strip()
    response = generate_text(
        prompt=prompt,
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    return response or "I found matching rows, but the LLM returned an empty answer."


def is_chart_request(question: str) -> bool:
    """Return whether a question requests a chart."""
    return has_draw_visualization_intent(question)


def table_answer(question: str, sql: str, rows: list[dict[str, Any]]) -> str:
    """Create a concise answer for table and schema inspection results."""
    row_count = len(rows)
    columns = list(rows[0].keys()) if rows else []
    table_name = referenced_table_name(sql)
    target = f" from `{table_name}`" if table_name else ""
    if "column" in question.lower() or "schema" in question.lower():
        return f"Here are the schema details{target}: {row_count} rows returned."
    return f"Here are {row_count} rows{target}. The result has {len(columns)} columns."


def referenced_table_name(sql: str) -> str | None:
    """Extract the first referenced table name from a SQL FROM clause."""
    match = re.search(
        r'\bfrom\s+(?:"?([a-zA-Z_][\w]*)"?\.)?"?([a-zA-Z_][\w]*)"?',
        sql,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    schema, table = match.groups()
    return f"{schema}.{table}" if schema else table


def table_name_from_question(question: str) -> str | None:
    """Extract a schema-qualified or unqualified table name from a prompt."""
    for pattern in (
        r"\b(?:from|in\s+table|table|in)\s+((?:[a-zA-Z_][\w]*\.)?[a-zA-Z_][\w]*)",
        r"\b((?:[a-zA-Z_][\w]*\.)[a-zA-Z_][\w]*)",
    ):
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def resolve_table_reference(table_reference: str, db_path: str | None = None) -> tuple[str, str] | None:
    """Resolve a prompt table reference against configured schemas."""
    schemas = get_schemas()
    if "." in table_reference:
        table_schema, table_name = table_reference.split(".", 1)
        if table_schema not in schemas:
            return None
        where_clause = "table_schema = %s AND table_name = %s"
        params: tuple[Any, ...] = (table_schema, table_name)
    else:
        where_clause = "table_schema = ANY(%s) AND table_name = %s"
        params = (schemas, table_reference)

    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE {where_clause}
                      AND table_type = 'BASE TABLE'
                    ORDER BY array_position(%s, table_schema), table_name
                    LIMIT 1
                    """,
                    (*params, schemas),
                )
                row = cur.fetchone()
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise QueryEngineError(f"PostgreSQL error: {message}") from exc

    return (row["table_schema"], row["table_name"]) if row else None


def answer_known_chart_question(question: str, db_path: str | None = None) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Answer deterministic chart prompts that need domain-specific SQL."""
    normalized = question.lower()
    if classify_question(question) != "draw":
        return None
    if "scatter" not in normalized or "pts" not in normalized or "minutes" not in normalized:
        return None

    requested_table = table_name_from_question(question)
    if not requested_table or not re.fullmatch(r"(?:[a-zA-Z_][\w]*\.)?[a-zA-Z_][\w]*", requested_table):
        return None

    if "." in requested_table:
        schema, table = requested_table.split(".", 1)
    else:
        schema = get_default_schema()
        table = requested_table
    if schema not in get_schemas():
        return None
    sql = pg_sql.SQL(
        """
        SELECT
            split_part(minutes, ':', 1)::numeric + split_part(minutes, ':', 2)::numeric / 60.0 AS minutes,
            pts,
            player_name,
            player_team
        FROM {}.{}
        WHERE minutes ~ '^[0-9]{{1,3}}:[0-9]{{2}}$'
          AND pts IS NOT NULL
          AND minutes <> '00:00'
        LIMIT 500
        """
    ).format(pg_sql.Identifier(schema), pg_sql.Identifier(table)).as_string()
    sql = "\n".join(line.strip() for line in sql.strip().splitlines())
    rows = execute_sql(validate_sql(sql), db_path=db_path)
    answer = (
        f"Here is the chart-ready result with {len(rows)} rows. "
        "I converted `minutes` from MM:SS text into decimal minutes and excluded `00:00` rows."
    )
    return answer, sql, rows


def chart_answer(question: str, sql: str, rows: list[dict[str, Any]]) -> str:
    """Create a concise answer for chart-ready result sets."""
    row_count = len(rows)
    limited = "limit" in sql.lower()
    notes = []
    if limited:
        notes.append("The SQL result is limited.")
    if excluded_zero_totals(sql, question):
        notes.append("Zero point totals were excluded so the ascending top chart starts with positive scorers.")
    suffix = " " + " ".join(notes) if notes else ""
    return f"Here is the chart-ready result with {row_count} rows.{suffix}"


def answer_question(
    question: str,
    db_path: str | None = None,
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Answer a user question by routing, generating SQL, and summarizing rows."""
    schema_answer = answer_schema_question(question, db_path=db_path)
    if schema_answer is not None:
        return schema_answer

    known_chart_answer = answer_known_chart_question(question, db_path=db_path)
    if known_chart_answer is not None:
        return known_chart_answer

    sql = generate_sql(
        question,
        db_path=db_path,
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    try:
        rows = execute_sql(sql, db_path=db_path)
    except QueryEngineError as exc:
        repaired_sql = repair_sql(
            question,
            failed_sql=sql,
            error=str(exc),
            db_path=db_path,
            provider=provider,
            gemini_api_key=gemini_api_key,
            ollama_api_key=ollama_api_key,
            model_name=model_name,
            ollama_base_url=ollama_base_url,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )
        rows = execute_sql(repaired_sql, db_path=db_path)
        sql = repaired_sql
    answer = summarize_answer(
        question,
        sql,
        rows,
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    return answer, sql, rows
