from __future__ import annotations

import os
import re
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from psycopg import sql
from psycopg import Error as PostgresError
from psycopg.rows import dict_row

from db import connect, get_db_label, get_schemas
from lightweight_rag import load_entries, load_rejected_entries, rag_enabled, save_bad_example, save_good_example
from query_engine import MissingApiKeyError, QueryEngineError, UnsafeSqlError, answer_question, classify_question


load_dotenv()

st.set_page_config(page_title="BasketballGPT", layout="wide")

AUTO_CREATE_SCHEMAS = os.getenv("NBA_AUTO_CREATE_SCHEMAS", "false").lower() in {"1", "true", "yes"}

STARTER_QUESTIONS = [
    "List me all tables",
    "Show 20 sample rows from bronze.b_el_boxscore",
    "Draw a bar chart of total pts by player_name from b_el_boxscore, top 20, sort descending",
    "Which players have the most rows in b_el_boxscore?",
    "Show me the columns in b_el_playbyplay",
]


def inject_styles() -> None:
    """Inject CSS that keeps chat, SQL, and table output readable."""
    st.markdown(
        """
        <style>
        [data-testid="stChatMessage"],
        [data-testid="stChatMessageContent"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stExpander"],
        [data-testid="stCodeBlock"] {
            max-width: 100%;
            min-width: 0;
        }

        [data-testid="stChatMessageContent"] p,
        [data-testid="stMarkdownContainer"] p {
            overflow-wrap: anywhere;
            word-break: normal;
            white-space: normal;
        }

        [data-testid="stCodeBlock"] {
            overflow-x: hidden;
        }

        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code {
            white-space: pre-wrap !important;
            overflow-wrap: anywhere;
            word-break: break-word !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_database() -> None:
    """Optionally create configured schemas for local development setups."""
    if not AUTO_CREATE_SCHEMAS:
        return
    with connect() as conn:
        for schema in get_schemas():
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))


@st.cache_data(ttl=60)
def schema_overview(schemas: list[str]) -> list[dict[str, Any]]:
    """Return table counts and column metadata for configured schemas."""
    with connect() as conn:
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
            tables = [(row["table_schema"], row["table_name"]) for row in cur.fetchall()]

            overview = []
            for table_schema, table in tables:
                cur.execute(
                    sql.SQL("SELECT count(*) AS row_count FROM {}.{}").format(
                        sql.Identifier(table_schema),
                        sql.Identifier(table),
                    )
                )
                row_count = cur.fetchone()["row_count"]
                cur.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table_schema, table),
                )
                columns = [dict(row) for row in cur.fetchall()]
                overview.append(
                    {
                        "schema": table_schema,
                        "table_name": table,
                        "row_count": row_count,
                        "columns": columns,
                    }
                )
            return overview


def configured_table_count() -> str:
    """Return the number of base tables visible in configured schemas."""
    schemas = get_schemas()
    try:
        with connect() as conn:
            row = conn.execute(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = ANY(%s)
                """,
                (schemas,),
            ).fetchone()
    except PostgresError:
        return "Unknown"
    return str(row[0] if row else 0)


def render_schema_browser() -> None:
    """Render the sidebar schema browser and selected-table column details."""
    schemas = get_schemas()
    try:
        overview = schema_overview(schemas)
    except PostgresError as exc:
        st.warning(f"Could not read schema metadata: {exc}")
        return

    if not overview:
        st.info(f"No tables found in `{', '.join(schemas)}`.")
        return

    table_rows = [
        {
            "table": f"{item['schema']}.{item['table_name']}",
            "rows": item["row_count"],
            "cols": len(item["columns"]),
        }
        for item in overview
    ]
    st.dataframe(
        table_rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            # "medium" pushed rows/cols off-screen in a narrow sidebar - table
            # names are long (e.g. bronze.b_cl_player_info_test) but TextColumn
            # already ellipsis-truncates on overflow, so "small" here is safe.
            "table": st.column_config.TextColumn("table", width="small"),
            "rows": st.column_config.NumberColumn("rows", width="small"),
            "cols": st.column_config.NumberColumn("cols", width="small"),
        },
    )

    table_options = [f"{item['schema']}.{item['table_name']}" for item in overview]
    selected_table = st.selectbox(
        "Inspect table",
        options=table_options,
    )
    selected = next(item for item in overview if f"{item['schema']}.{item['table_name']}" == selected_table)
    st.dataframe(selected["columns"], hide_index=True, use_container_width=True)


def sidebar() -> dict[str, str | None]:
    """Render sidebar controls and return the selected LLM configuration."""
    with st.sidebar:
        st.header("BasketballGPT")
        st.caption("Basketball analytics across bronze, silver, and gold.")
        st.caption(f"Database: `{get_db_label()}`")
        st.caption(f"Configured tables: {configured_table_count()}")

        with st.expander("Schema Browser", expanded=True):
            render_schema_browser()

        st.divider()
        st.header("SQL Memory")
        status = "enabled" if rag_enabled() else "disabled"
        st.caption(f"Lightweight RAG: `{status}`")
        st.caption(f"Good examples: `{len(load_entries())}`")
        st.caption(f"Rejected examples: `{len(load_rejected_entries())}`")

        st.divider()
        st.header("LLM")
        default_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        provider_options = ["ollama", "gemini", "openai"]
        default_index = provider_options.index(default_provider) if default_provider in provider_options else 0
        provider = st.selectbox(
            "Provider",
            options=provider_options,
            index=default_index,
        )

        empty_llm_config = {
            "gemini_api_key": None,
            "ollama_api_key": None,
            "ollama_base_url": None,
            "openai_api_key": None,
            "openai_base_url": None,
        }

        if provider == "ollama":
            ollama_base_url = st.text_input(
                "Ollama base URL",
                value=st.session_state.get(
                    "ollama_base_url",
                    os.getenv("OLLAMA_BASE_URL", "https://ollama.com/api"),
                ),
            )
            ollama_model = st.text_input(
                "Ollama model",
                value=st.session_state.get("ollama_model", os.getenv("OLLAMA_MODEL", "qwen3-coder:480b-cloud")),
            )
            ollama_api_key = st.text_input(
                "Ollama API key",
                # Not pre-filled from OLLAMA_API_KEY: this value is rendered to the browser, and on
                # a shared deployment that would leak the platform's key to anyone who opens the
                # page. Leave blank to use the server-side env var (still applied in query_engine.py).
                value=st.session_state.get("ollama_api_key", ""),
                type="password",
                help="Only needed for https://ollama.com/api cloud access. Leave blank to use the server-configured key.",
            )
            st.session_state["ollama_base_url"] = ollama_base_url
            st.session_state["ollama_model"] = ollama_model
            st.session_state["ollama_api_key"] = ollama_api_key
            st.caption("Ollama cloud requires an API key. Local Ollama uses http://host.docker.internal:11434 and no key.")
            return {
                **empty_llm_config,
                "provider": provider,
                "model_name": ollama_model or None,
                "ollama_base_url": ollama_base_url or None,
                "ollama_api_key": ollama_api_key or None,
            }

        if provider == "openai":
            openai_base_url = st.text_input(
                "OpenAI-compatible base URL",
                value=st.session_state.get(
                    "openai_base_url",
                    os.getenv("OPENAI_BASE_URL", "http://litellm.litellm.svc.cluster.local:4000/v1"),
                ),
                help="The in-cluster LiteLLM proxy by default (same one datachat uses).",
            )
            openai_model = st.text_input(
                "OpenAI-compatible model",
                value=st.session_state.get("openai_model", os.getenv("OPENAI_MODEL", "qwen3.6-35b-a3b-coder")),
            )
            openai_api_key = st.text_input(
                "OpenAI-compatible API key",
                # Not pre-filled from OPENAI_API_KEY — see the note on the Ollama key field above.
                value=st.session_state.get("openai_api_key", ""),
                type="password",
                help="A LiteLLM virtual key. Leave blank to use the server-configured key.",
            )
            st.session_state["openai_base_url"] = openai_base_url
            st.session_state["openai_model"] = openai_model
            st.session_state["openai_api_key"] = openai_api_key
            st.caption("Keys are kept in Streamlit session state and are not written to disk.")
            return {
                **empty_llm_config,
                "provider": provider,
                "model_name": openai_model or None,
                "openai_base_url": openai_base_url or None,
                "openai_api_key": openai_api_key or None,
            }

        gemini_api_key = st.text_input(
            "Gemini API key",
            # Not pre-filled from GEMINI_API_KEY — see the note on the Ollama key field above.
            value=st.session_state.get("gemini_api_key", ""),
            type="password",
            help="Leave blank to use the server-configured key.",
        )
        gemini_model = st.text_input(
            "Gemini model",
            value=st.session_state.get("gemini_model", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")),
        )
        st.session_state["gemini_api_key"] = gemini_api_key
        st.session_state["gemini_model"] = gemini_model
        st.caption("Keys are kept in Streamlit session state and are not written to disk.")
        return {
            **empty_llm_config,
            "provider": provider,
            "model_name": gemini_model or None,
            "gemini_api_key": gemini_api_key or None,
        }


def render_chat() -> None:
    """Replay chat history with SQL, result summaries, and optional charts."""
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sql"):
                with st.expander("Generated SQL"):
                    st.code(message["sql"], language="sql")
                    if message.get("rows"):
                        st.dataframe(message["rows"], use_container_width=True)
                render_result_summary(message.get("rows"), message.get("intent"))
                render_result_chart(message.get("rows"), message.get("sql"), message.get("question"))
                render_feedback_controls(message, key_prefix=f"history_{index}")


def render_feedback_controls(message: dict[str, Any], key_prefix: str) -> None:
    """Render buttons that save good and bad SQL feedback to memory files."""
    question = message.get("question")
    sql = message.get("sql")
    if not question or not sql:
        return

    columns = st.columns([1, 1, 5])
    if columns[0].button("Save good", key=f"{key_prefix}_save_good", help="Save this question and SQL as a reusable example."):
        saved, detail = save_good_example(question=question, sql=sql, answer=message.get("content"))
        if saved:
            st.success("Saved as a good SQL example.")
        else:
            st.info(detail)
    if columns[1].button("Mark bad", key=f"{key_prefix}_mark_bad", help="Save this SQL as a rejected pattern for similar questions."):
        saved, detail = save_bad_example(question=question, sql=sql)
        if saved:
            st.warning("Saved as a rejected SQL example.")
        else:
            st.info(detail)


def render_starter_questions() -> str | None:
    """Render starter question buttons and return the clicked question."""
    st.caption("Starter questions")
    cols = st.columns(3)
    for index, question in enumerate(STARTER_QUESTIONS):
        if cols[index % len(cols)].button(question, use_container_width=True):
            return question
    return None


def render_result_chart(
    rows: list[dict[str, Any]] | None,
    generated_sql: str | None = None,
    question: str | None = None,
) -> None:
    """Render chart tabs when the user explicitly asks to draw a chart."""
    if not rows:
        return
    if not should_show_visualizations(question, generated_sql):
        return

    df = pd.DataFrame(rows)
    if df.empty or len(df.columns) < 2:
        return

    df = prepare_chart_dataframe(df)
    numeric_columns = list(df.select_dtypes(include="number").columns)
    if not numeric_columns:
        return

    with st.expander("Visualizations", expanded=True):
        chart_options = prioritize_chart_options(available_chart_options(df), question)
        if not chart_options:
            st.caption("No compatible chart found for these rows.")
            return

        tabs = st.tabs(chart_options)
        for tab, chart_name in zip(tabs, chart_options, strict=True):
            with tab:
                if chart_name == "Bar":
                    render_bar_chart(df, numeric_columns, generated_sql)
                elif chart_name == "Line":
                    render_line_chart(df, numeric_columns)
                elif chart_name == "Scatter":
                    render_scatter_chart(df, numeric_columns, question)
                elif chart_name == "Histogram":
                    render_histogram(df, numeric_columns)


def should_show_visualizations(question: str | None, _generated_sql: str | None) -> bool:
    """Return whether a result should show visualization controls."""
    return classify_question(question or "") == "draw"


def render_result_summary(rows: list[dict[str, Any]] | None, intent: str | None = None) -> None:
    """Render lightweight profiling information for returned SQL rows."""
    if not rows:
        return

    df = pd.DataFrame(rows)
    if df.empty:
        return

    summary_df = prepare_chart_dataframe(df)
    numeric_columns = list(summary_df.select_dtypes(include="number").columns)
    date_columns = list(summary_df.select_dtypes(include="datetime").columns)
    text_columns = [column for column in summary_df.columns if column not in numeric_columns + date_columns]

    with st.expander("Result summary", expanded=classify_summary_expanded(intent)):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Rows returned", f"{len(summary_df):,}")
        metric_cols[1].metric("Columns", f"{len(summary_df.columns):,}")
        metric_cols[2].metric("Numeric columns", f"{len(numeric_columns):,}")
        metric_cols[3].metric("Text/date columns", f"{len(text_columns) + len(date_columns):,}")

        column_profile = []
        for column in summary_df.columns:
            series = summary_df[column]
            column_profile.append(
                {
                    "column": column,
                    "type": str(series.dtype),
                    "nulls": int(series.isna().sum()),
                    "non_nulls": int(series.notna().sum()),
                    "distinct_in_result": int(series.nunique(dropna=True)),
                }
            )
        st.dataframe(column_profile, hide_index=True, use_container_width=True)

        if numeric_columns:
            numeric_summary = (
                summary_df[numeric_columns]
                .agg(["min", "max", "mean"])
                .round(2)
                .transpose()
                .reset_index()
                .rename(columns={"index": "column"})
            )
            st.dataframe(numeric_summary, hide_index=True, use_container_width=True)


def classify_summary_expanded(intent: str | None) -> bool:
    """Return whether the result summary should be expanded by default."""
    return intent in {"table", "schema"}


def prepare_chart_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert chart-friendly text columns to numeric or datetime types."""
    chart_df = df.copy()
    for column in chart_df.columns:
        if is_date_like_column(column):
            parsed = pd.to_datetime(chart_df[column], errors="coerce")
            if parsed.notna().sum() >= max(1, len(parsed) // 2):
                chart_df[column] = parsed
            continue

        if is_duration_like_column(column, chart_df[column]):
            converted_minutes = parse_duration_minutes(chart_df[column])
            if converted_minutes.notna().sum() >= max(1, len(converted_minutes) // 2):
                chart_df[column] = converted_minutes
                continue

        converted = pd.to_numeric(chart_df[column], errors="coerce")
        if converted.notna().sum() >= max(1, len(converted) // 2):
            chart_df[column] = converted
    return chart_df


def is_date_like_column(column: str) -> bool:
    """Return whether a column name likely contains date or time values."""
    lowered = column.lower()
    return any(part in lowered for part in ("date", "time", "timestamp", "created_at", "updated_at"))


def is_duration_like_column(column: str, series: pd.Series) -> bool:
    """Return whether a column likely stores basketball duration values."""
    lowered = column.lower()
    if any(part in lowered for part in ("minute", "minutes", "min")):
        return True
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    return sample.str.match(r"^\d{1,3}:\d{2}$").mean() >= 0.5


def parse_duration_minutes(series: pd.Series) -> pd.Series:
    """Convert MM:SS duration strings into decimal minutes."""
    # Basketball boxscore minutes are commonly stored as MM:SS text. Convert
    # them to decimal minutes for quantitative charts without mutating the DB.
    text = series.astype(str).str.strip()
    parts = text.str.extract(r"^(?P<minutes>\d{1,3}):(?P<seconds>\d{2})$")
    whole_minutes = pd.to_numeric(parts["minutes"], errors="coerce")
    seconds = pd.to_numeric(parts["seconds"], errors="coerce")
    parsed = whole_minutes + (seconds / 60)
    numeric = pd.to_numeric(series, errors="coerce")
    return parsed.fillna(numeric)


def available_chart_options(df: pd.DataFrame) -> list[str]:
    """Return compatible chart types for the dataframe's column types."""
    numeric_columns = list(df.select_dtypes(include="number").columns)
    date_columns = list(df.select_dtypes(include="datetime").columns)
    categorical_columns = [column for column in df.columns if column not in numeric_columns + date_columns]

    options = []
    if categorical_columns and numeric_columns:
        options.append("Bar")
    if date_columns and numeric_columns:
        options.append("Line")
    if len(numeric_columns) >= 2:
        options.append("Scatter")
    if numeric_columns:
        options.append("Histogram")
    return options


def prioritize_chart_options(options: list[str], question: str | None) -> list[str]:
    """Move the chart type requested in the prompt to the first tab."""
    requested = requested_chart_name(question)
    if requested in options:
        return [requested] + [option for option in options if option != requested]
    return options


def requested_chart_name(question: str | None) -> str | None:
    """Infer the chart type requested by the user's prompt."""
    if not question:
        return None
    lowered = question.lower()
    if "scatter" in lowered:
        return "Scatter"
    if "line" in lowered:
        return "Line"
    if "histogram" in lowered:
        return "Histogram"
    if "bar" in lowered:
        return "Bar"
    return None


def render_bar_chart(df: pd.DataFrame, numeric_columns: list[str], generated_sql: str | None) -> None:
    """Render a horizontal bar chart with stable sorting and tooltips."""
    label_columns = [column for column in df.columns if column not in numeric_columns]
    label_column = st.selectbox("Label", label_columns or list(df.columns), key=f"bar_label_{id(df)}")
    value_column = st.selectbox("Value", numeric_columns, key=f"bar_value_{id(df)}")
    sort_ascending = bar_chart_sort_ascending(generated_sql)
    chart_df = df[[label_column, value_column]].dropna()
    if should_sort_bar_chart(generated_sql, value_column):
        chart_df = chart_df.sort_values(
            value_column,
            ascending=sort_ascending,
        )
    chart_df = chart_df.head(25)
    if chart_df.empty:
        st.caption("No rows available for a bar chart.")
        return

    sort_order = "ascending" if sort_ascending else "descending"
    title = f"{humanize_column(value_column)} by {humanize_column(label_column)}"
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_column}:Q", title=humanize_column(value_column)),
            y=alt.Y(
                f"{label_column}:N",
                sort=alt.EncodingSortField(field=value_column, order=sort_order),
                title=humanize_column(label_column),
                axis=alt.Axis(labelLimit=260),
            ),
            tooltip=[
                alt.Tooltip(f"{label_column}:N", title=humanize_column(label_column)),
                alt.Tooltip(f"{value_column}:Q", title=humanize_column(value_column)),
            ],
        )
        .properties(title=title)
        .properties(height=max(320, min(700, 28 * len(chart_df))))
    )
    st.altair_chart(chart, use_container_width=True)


def humanize_column(column: str) -> str:
    """Format a database column name for display in chart labels."""
    return column.replace("_", " ").strip().title()


def should_sort_bar_chart(generated_sql: str | None, value_column: str) -> bool:
    """Return whether the chart should sort locally rather than trust SQL order."""
    if not generated_sql:
        return True
    return value_column.lower() not in generated_sql.lower() or "order by" not in generated_sql.lower()


def bar_chart_sort_ascending(generated_sql: str | None) -> bool:
    """Infer bar chart sort direction from the generated SQL."""
    if not generated_sql:
        return False
    lowered = generated_sql.lower()
    if "order by" in lowered and " asc" in lowered:
        return True
    return False


def render_line_chart(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    """Render a line chart for date/time series results."""
    date_columns = list(df.select_dtypes(include="datetime").columns)
    date_column = st.selectbox("Date/time", date_columns, key=f"line_date_{id(df)}")
    value_column = st.selectbox("Value", numeric_columns, key=f"line_value_{id(df)}")
    chart_df = df[[date_column, value_column]].dropna().sort_values(date_column).head(500)
    if chart_df.empty:
        st.caption("No rows available for a line chart.")
        return
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{date_column}:T", title=humanize_column(date_column)),
            y=alt.Y(f"{value_column}:Q", title=humanize_column(value_column)),
            tooltip=[
                alt.Tooltip(f"{date_column}:T", title=humanize_column(date_column)),
                alt.Tooltip(f"{value_column}:Q", title=humanize_column(value_column)),
            ],
        )
        .properties(title=f"{humanize_column(value_column)} Over {humanize_column(date_column)}")
    )
    st.altair_chart(chart, use_container_width=True)


def render_scatter_chart(df: pd.DataFrame, numeric_columns: list[str], question: str | None = None) -> None:
    """Render a scatter chart with prompt-aware default axes."""
    preferred_x, preferred_y = preferred_scatter_columns(question, numeric_columns)
    x_index = numeric_columns.index(preferred_x) if preferred_x in numeric_columns else 0
    x_column = st.selectbox("X", numeric_columns, index=x_index, key=f"scatter_x_{id(df)}")
    y_candidates = [column for column in numeric_columns if column != x_column] or numeric_columns
    y_index = y_candidates.index(preferred_y) if preferred_y in y_candidates else 0
    y_column = st.selectbox("Y", y_candidates, index=y_index, key=f"scatter_y_{id(df)}")
    chart_df = df[[x_column, y_column]].dropna().head(1000)
    if chart_df.empty:
        st.caption("No rows available for a scatter chart.")
        return
    chart = (
        alt.Chart(chart_df)
        .mark_circle(size=70, opacity=0.7)
        .encode(
            x=alt.X(f"{x_column}:Q", title=humanize_column(x_column)),
            y=alt.Y(f"{y_column}:Q", title=humanize_column(y_column)),
            tooltip=[
                alt.Tooltip(f"{x_column}:Q", title=humanize_column(x_column)),
                alt.Tooltip(f"{y_column}:Q", title=humanize_column(y_column)),
            ],
        )
        .properties(title=f"{humanize_column(y_column)} vs {humanize_column(x_column)}")
    )
    st.altair_chart(chart, use_container_width=True)


def preferred_scatter_columns(question: str | None, numeric_columns: list[str]) -> tuple[str | None, str | None]:
    """Infer scatter chart x/y defaults from prompts like 'pts vs minutes'."""
    if not question:
        return None, None
    lowered = question.lower()
    match = re.search(r"\b([\w_]+)\s+vs\.?\s+([\w_]+)", lowered)
    if not match:
        return None, None
    y_requested, x_requested = match.groups()
    x_column = match_column_name(x_requested, numeric_columns)
    y_column = match_column_name(y_requested, numeric_columns)
    return x_column, y_column


def match_column_name(requested: str, columns: list[str]) -> str | None:
    """Match a requested metric or alias to an available dataframe column."""
    normalized = requested.lower()
    aliases = {
        "points": "pts",
        "point": "pts",
        "score": "pts",
        "minutes": "minutes",
        "minute": "minutes",
        "mins": "minutes",
    }
    target = aliases.get(normalized, normalized)
    for column in columns:
        if column.lower() == target:
            return column
    for column in columns:
        if target in column.lower():
            return column
    return None


def render_histogram(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    """Render a histogram for one numeric result column."""
    value_column = st.selectbox("Value", numeric_columns, key=f"hist_value_{id(df)}")
    chart_df = df[[value_column]].dropna().head(5000)
    if chart_df.empty:
        st.caption("No rows available for a histogram.")
        return

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_column}:Q", bin=alt.Bin(maxbins=30), title=humanize_column(value_column)),
            y=alt.Y("count():Q", title="Rows"),
            tooltip=[
                alt.Tooltip(f"{value_column}:Q", bin=True, title=humanize_column(value_column)),
                alt.Tooltip("count():Q", title="Rows"),
            ],
        )
        .properties(title=f"Distribution of {humanize_column(value_column)}")
    )
    st.altair_chart(chart, use_container_width=True)


def main() -> None:
    """Run the Streamlit basketball analytics chat application."""
    ensure_database()
    inject_styles()
    llm_config = sidebar()

    st.title("BasketballGPT")
    st.write("Ask questions across bronze, silver, and gold basketball analytics data.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to BasketballGPT. Ask questions like `List me all tables`, "
                    "`Show sample rows from silver.s_el_player_stats_prospiel`, "
                    "or `Draw a bar chart of total pts by player_name from bronze.b_el_boxscore`."
                ),
            }
        ]

    starter_question = render_starter_questions()
    render_chat()

    typed_question = st.chat_input("Ask about players, teams, games, or stats")
    question = starter_question or typed_question
    if not question:
        return
    intent = classify_question(question)

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL and querying PostgreSQL..."):
            _rows = None
            try:
                answer, sql, _rows = answer_question(question, **llm_config)
            except MissingApiKeyError as exc:
                answer = str(exc)
                sql = None
                st.error(answer)
            except UnsafeSqlError as exc:
                answer = f"I rejected the generated SQL: {exc}"
                sql = None
                st.error(answer)
            except QueryEngineError as exc:
                answer = str(exc)
                sql = None
                st.error(answer)
            except Exception as exc:
                answer = f"LLM or database query failed: {exc}"
                sql = None
                st.error(answer)
            else:
                st.markdown(answer)
                with st.expander("Generated SQL"):
                    st.code(sql, language="sql")
                    if _rows:
                        st.dataframe(_rows, use_container_width=True)
                render_result_summary(_rows, intent)
                render_result_chart(_rows, sql, question)
                render_feedback_controls(
                    {
                        "question": question,
                        "sql": sql,
                        "content": answer,
                    },
                    key_prefix="current_answer",
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sql": sql,
            "rows": _rows if sql else None,
            "question": question,
            "intent": intent,
        }
    )


if __name__ == "__main__":
    main()
