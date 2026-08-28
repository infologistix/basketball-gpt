from __future__ import annotations

import os
import re
from typing import Any
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from psycopg import sql
from psycopg import Error as PostgresError
from psycopg.rows import dict_row

from db import connect, get_database_name, get_db_label, get_schemas
from lightweight_rag import load_entries, load_rejected_entries, rag_enabled, save_bad_example, save_good_example
from query_engine import MissingApiKeyError, QueryEngineError, UnsafeSqlError, answer_question, classify_question


load_dotenv()

st.set_page_config(page_title="BasketballGPT", page_icon="🏀", layout="wide")

AUTO_CREATE_SCHEMAS = os.getenv("NBA_AUTO_CREATE_SCHEMAS", "false").lower() in {"1", "true", "yes"}

# Six, not five, so the three-column grid fills two even rows instead of leaving
# a gap. Kept to a similar length each: the old set mixed a three-word label with
# a full sentence, so one button wrapped to two lines and the row went ragged.
# German, like the rest of the interface - that is the language questions are
# actually asked in here.
STARTER_QUESTIONS = [
    "Welche Tabellen gibt es in bronze?",
    "Meiste Rebounds in der BBL-Saison 2025-2026",
    "Welche BBL-Spieler kommen aus Deutschland?",
    "Punkte pro BBL-Team als Balkendiagramm",
    "20 Beispielzeilen aus bronze.b_el_boxscore",
    "Beste Dreipunktequote der EuroLeague 2025-2026",
]


# Chart palette, kept in step with .streamlit/config.toml. Vega's defaults are a
# light blue on a white plot area, which is why untouched charts looked pasted
# onto the page rather than part of it.
CHART_FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
CHART_ACCENT = "#E07A46"
CHART_TEXT = "#EFEAE3"
CHART_MUTED = "#A69F95"
CHART_GRID = "#2B261F"


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Apply the app's palette and typography to a chart.

    Called once per chart right before rendering, so all four chart types share
    one definition instead of repeating configure_* calls in each builder.
    """
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelFont=CHART_FONT,
            titleFont=CHART_FONT,
            labelColor=CHART_MUTED,
            titleColor=CHART_MUTED,
            labelFontSize=13,
            titleFontSize=13,
            titleFontWeight="normal",
            titlePadding=10,
            gridColor=CHART_GRID,
            domainColor=CHART_GRID,
            tickColor=CHART_GRID,
        )
        .configure_title(
            font=CHART_FONT,
            fontSize=16,
            fontWeight=500,
            color=CHART_TEXT,
            anchor="start",
            offset=14,
        )
        .configure_legend(
            labelFont=CHART_FONT,
            titleFont=CHART_FONT,
            labelColor=CHART_MUTED,
            titleColor=CHART_MUTED,
        )
    )


# Only the tab captions are translated. The keys stay English because they are
# also what requested_chart_name() returns and what prioritize_chart_options()
# matches on - renaming those would mean touching the matching logic too.
CHART_TAB_LABELS = {
    "Bar": "Balken",
    "Line": "Linie",
    "Scatter": "Streuung",
    "Histogram": "Histogramm",
}


def inject_styles() -> None:
    """Inject CSS that keeps chat, SQL, and table output readable."""
    st.markdown(
        """
        <style>
        /* config.toml's [theme] font key only accepts sans serif / serif /
           monospace, so the typeface has to come through here. IBM Plex pairs a
           sans with a mono of matching metrics, which is what table names and
           SQL need sitting next to prose. */
        @import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap");

        /* Streamlit sizes nearly everything in rem off the root, so raising the
           root size once scales body text, captions, inputs, buttons and their
           spacing together instead of chasing individual selectors. 16px is the
           browser default; IBM Plex also runs slightly smaller on the x-height
           than Streamlit's stock Source Sans, which made the switch read as a
           size drop. */
        html {
            font-size: 17.5px;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
        }

        /* The dataframe grid renders to canvas and does not inherit the root
           size, so it needs its own step to stay in proportion. */
        [data-testid="stDataFrame"] {
            font-size: 0.92rem;
        }

        code, pre, [data-testid="stCodeBlock"] * {
            font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, monospace !important;
        }

        /* Without tabular figures the numeric columns of every result table
           jitter as rows scroll past. */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            font-variant-numeric: tabular-nums;
        }

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
        return "?"
    return str(row[0] if row else 0)


def render_schema_browser() -> None:
    """Render the sidebar schema browser and selected-table column details."""
    schemas = get_schemas()
    try:
        overview = schema_overview(schemas)
    except PostgresError as exc:
        st.warning(f"Schema-Metadaten nicht lesbar: {exc}")
        return

    if not overview:
        st.info(f"Keine Tabellen in {', '.join(schemas)} gefunden.")
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
            # "small" on the table column truncated the only value that
            # identifies a row - the sidebar showed "silver.s_cl_p…" while rows
            # and cols kept full width. The counts are the columns that can
            # afford to be narrow, so the space goes to the name instead.
            "table": st.column_config.TextColumn("Tabelle", width="medium"),
            "rows": st.column_config.NumberColumn("Zeilen", width="small"),
            "cols": st.column_config.NumberColumn("Sp.", width="small"),
        },
    )

    table_options = [f"{item['schema']}.{item['table_name']}" for item in overview]
    selected_table = st.selectbox(
        "Tabelle ansehen",
        options=table_options,
    )
    selected = next(item for item in overview if f"{item['schema']}.{item['table_name']}" == selected_table)
    st.dataframe(selected["columns"], hide_index=True, use_container_width=True)


def sidebar() -> dict[str, str | None]:
    """Render sidebar controls and return the selected LLM configuration."""
    with st.sidebar:
        # st.header() gives no way to put a mark next to the title, so the
        # sidebar heading is markup. Drawn rather than an emoji so it inherits
        # the accent colour and stays crisp at any zoom.
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;margin:0 0 2px">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                   stroke="#E07A46" stroke-width="1.6" stroke-linecap="round">
                <circle cx="12" cy="12" r="9"></circle>
                <path d="M3 12h18M12 3v18"></path>
                <path d="M5.6 5.6c3.5 3 3.5 9.8 0 12.8M18.4 5.6c-3.5 3-3.5 9.8 0 12.8"></path>
              </svg>
              <span style="font-size:1.35rem;font-weight:600;letter-spacing:-.2px">BasketballGPT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Basketball-Analytics über bronze, silver und gold.")

        # The full connection string used to sit here, rendered as four lines of
        # inline code - the loudest element on the page for a value nobody reads
        # day to day. What matters at a glance is the database name and how many
        # tables are reachable; the URL moves to the bottom of the sidebar.
        st.caption(f"{get_database_name()} · {configured_table_count()} Tabellen")

        with st.expander("Schema", expanded=True):
            render_schema_browser()

        st.divider()
        st.header("SQL-Gedächtnis")
        # Backticks here rendered as inline code, i.e. in the theme's code
        # colour - an accidental highlight on a plain count. Plain text instead,
        # and the two counts side by side so they read as a pair.
        good_column, rejected_column = st.columns(2)
        good_column.metric("Gut", len(load_entries()))
        rejected_column.metric("Abgelehnt", len(load_rejected_entries()))
        if not rag_enabled():
            st.caption("Retrieval ist deaktiviert (SQL_RAG_ENABLED)")

        with st.expander("Verbindungsdetails"):
            st.code(get_db_label(), language=None)

        st.divider()
        st.header("Modell")
        default_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        provider_options = ["ollama", "gemini", "openai"]
        # "openai" stays the internal key - it is the LLM_PROVIDER env value, the
        # branch conditions below and the openai_* config fields. Only the label
        # changes: this provider talks to the in-cluster KubeSpectra LiteLLM
        # proxy, which is merely OpenAI-compatible, not OpenAI.
        provider_labels = {"openai": "KubeSpectra"}
        default_index = provider_options.index(default_provider) if default_provider in provider_options else 0
        provider = st.selectbox(
            "Anbieter",
            options=provider_options,
            index=default_index,
            format_func=lambda value: provider_labels.get(value, value),
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
                "Ollama Basis-URL",
                value=st.session_state.get(
                    "ollama_base_url",
                    os.getenv("OLLAMA_BASE_URL", "https://ollama.com/api"),
                ),
            )
            ollama_model = st.text_input(
                "Ollama Modell",
                value=st.session_state.get("ollama_model", os.getenv("OLLAMA_MODEL", "qwen3-coder:480b-cloud")),
            )
            ollama_api_key = st.text_input(
                "Ollama API-Key",
                # Not pre-filled from OLLAMA_API_KEY: this value is rendered to the browser, and on
                # a shared deployment that would leak the platform's key to anyone who opens the
                # page. Leave blank to use the server-side env var (still applied in query_engine.py).
                value=st.session_state.get("ollama_api_key", ""),
                type="password",
                help="Nur für den Cloud-Zugang https://ollama.com/api nötig. Leer lassen, um den serverseitigen Key zu nutzen.",
            )
            st.session_state["ollama_base_url"] = ollama_base_url
            st.session_state["ollama_model"] = ollama_model
            st.session_state["ollama_api_key"] = ollama_api_key
            st.caption("Ollama Cloud braucht einen API-Key. Lokales Ollama nutzt http://host.docker.internal:11434 ohne Key.")
            return {
                **empty_llm_config,
                "provider": provider,
                "model_name": ollama_model or None,
                "ollama_base_url": ollama_base_url or None,
                "ollama_api_key": ollama_api_key or None,
            }

        if provider == "openai":
            openai_base_url = st.text_input(
                "KubeSpectra Basis-URL",
                value=st.session_state.get(
                    "openai_base_url",
                    os.getenv("OPENAI_BASE_URL", "http://litellm.litellm.svc.cluster.local:4000/v1"),
                ),
                help="Standardmäßig der Cluster-interne LiteLLM-Proxy (derselbe, den datachat nutzt).",
            )
            openai_model = st.text_input(
                "KubeSpectra Modell",
                value=st.session_state.get("openai_model", os.getenv("OPENAI_MODEL", "qwen3.6-35b-a3b-coder")),
            )
            openai_api_key = st.text_input(
                "KubeSpectra API-Key",
                # Not pre-filled from OPENAI_API_KEY — see the note on the Ollama key field above.
                value=st.session_state.get("openai_api_key", ""),
                type="password",
                help="Ein virtueller LiteLLM-Key. Leer lassen, um den serverseitigen Key zu nutzen.",
            )
            st.session_state["openai_base_url"] = openai_base_url
            st.session_state["openai_model"] = openai_model
            st.session_state["openai_api_key"] = openai_api_key
            st.caption("Keys liegen nur in der Streamlit-Session und werden nicht auf Platte geschrieben.")
            return {
                **empty_llm_config,
                "provider": provider,
                "model_name": openai_model or None,
                "openai_base_url": openai_base_url or None,
                "openai_api_key": openai_api_key or None,
            }

        gemini_api_key = st.text_input(
            "Gemini API-Key",
            # Not pre-filled from GEMINI_API_KEY — see the note on the Ollama key field above.
            value=st.session_state.get("gemini_api_key", ""),
            type="password",
            help="Leer lassen, um den serverseitigen Key zu nutzen.",
        )
        gemini_model = st.text_input(
            "Gemini Modell",
            value=st.session_state.get("gemini_model", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")),
        )
        st.session_state["gemini_api_key"] = gemini_api_key
        st.session_state["gemini_model"] = gemini_model
        st.caption("Keys liegen nur in der Streamlit-Session und werden nicht auf Platte geschrieben.")
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
                with st.expander("Erzeugtes SQL"):
                    st.code(message["sql"], language="sql")
                    if message.get("rows"):
                        st.dataframe(message["rows"], use_container_width=True)
                render_result_summary(message.get("rows"), message.get("intent"))
                render_result_chart(message.get("rows"), message.get("sql"), message.get("question"))
                render_feedback_controls(
                    message,
                    key_prefix=message.get("feedback_id") or f"history_{index}",
                )


def render_feedback_controls(message: dict[str, Any], key_prefix: str) -> None:
    """Render buttons that save good and bad SQL feedback to memory files."""
    question = message.get("question")
    sql = message.get("sql")
    if not question or not sql:
        return

    reason_key = f"{key_prefix}_ask_reason"

    columns = st.columns([1, 1, 5])
    if columns[0].button("Gut", key=f"{key_prefix}_save_good", help="Frage und SQL als wiederverwendbares Beispiel speichern."):
        saved, detail = save_good_example(question=question, sql=sql, answer=message.get("content"))
        if saved:
            st.success("Als gutes Beispiel gespeichert.")
        else:
            st.info(detail)
    if columns[1].button("Falsch", key=f"{key_prefix}_mark_bad", help="Dieses SQL als Anti-Muster für ähnliche Fragen speichern."):
        st.session_state[reason_key] = True

    if not st.session_state.get(reason_key):
        return

    # format_rejected_context() puts this reason straight into later prompts, so
    # the default "Marked as bad by user." teaches the model nothing: it learns
    # to avoid the pattern without learning how to fix it. Require a real one.
    with st.form(key=f"{key_prefix}_reason_form"):
        reason = st.text_input(
            "Warum ist das falsch?",
            placeholder="z. B. fehlendes DISTINCT ON – der Join fächert auf und verdoppelt jede Summe",
        )
        form_columns = st.columns([1, 1, 5])
        confirmed = form_columns[0].form_submit_button("Speichern")
        cancelled = form_columns[1].form_submit_button("Abbrechen")

    if cancelled:
        st.session_state[reason_key] = False
        return
    if not confirmed:
        return
    if not reason.strip():
        st.error("Bitte zuerst einen Grund angeben.")
        return

    saved, detail = save_bad_example(question=question, sql=sql, reason=reason.strip())
    st.session_state[reason_key] = False
    if saved:
        st.warning("Als abgelehntes Beispiel gespeichert.")
    else:
        st.info(detail)


def render_starter_questions() -> str | None:
    """Render starter question buttons and return the clicked question."""
    st.caption("Beispielfragen")
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

    with st.expander("Visualisierungen", expanded=True):
        chart_options = prioritize_chart_options(available_chart_options(df), question)
        if not chart_options:
            st.caption("Kein passendes Diagramm für diese Zeilen.")
            return

        tabs = st.tabs([CHART_TAB_LABELS.get(name, name) for name in chart_options])
        for tab, chart_name in zip(tabs, chart_options, strict=True):
            with tab:
                if chart_name == "Bar":
                    render_bar_chart(df, numeric_columns, generated_sql)
                elif chart_name == "Line":
                    render_line_chart(df, numeric_columns, generated_sql)
                elif chart_name == "Scatter":
                    render_scatter_chart(df, numeric_columns, question)
                elif chart_name == "Histogram":
                    render_histogram(df, numeric_columns, generated_sql)


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

    with st.expander("Zusammenfassung", expanded=classify_summary_expanded(intent)):
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
    if "scatter" in lowered or "streu" in lowered:
        return "Scatter"
    if "line" in lowered or "linien" in lowered or "verlauf" in lowered:
        return "Line"
    if "histogram" in lowered:
        return "Histogram"
    if "bar" in lowered or "balken" in lowered:
        return "Bar"
    return None


def render_bar_chart(df: pd.DataFrame, numeric_columns: list[str], generated_sql: str | None) -> None:
    """Render a horizontal bar chart with stable sorting and tooltips."""
    label_columns = [column for column in df.columns if column not in numeric_columns]
    label_column = st.selectbox("Beschriftung", label_columns or list(df.columns), key=f"bar_label_{id(df)}")
    value_column = st.selectbox(
        "Wert", numeric_columns, index=default_value_index(numeric_columns, generated_sql), key=f"bar_value_{id(df)}"
    )
    sort_ascending = bar_chart_sort_ascending(generated_sql)
    chart_df = df[[label_column, value_column]].dropna()
    if should_sort_bar_chart(generated_sql, value_column):
        chart_df = chart_df.sort_values(
            value_column,
            ascending=sort_ascending,
        )
    chart_df = chart_df.head(25)
    if chart_df.empty:
        st.caption("Keine Zeilen für ein Balkendiagramm.")
        return

    sort_order = "ascending" if sort_ascending else "descending"
    title = f"{humanize_column(value_column)} nach {humanize_column(label_column)}"
    chart = (
        alt.Chart(chart_df)
        .mark_bar(color=CHART_ACCENT, cornerRadiusEnd=2)
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
    st.altair_chart(style_chart(chart), use_container_width=True)


def humanize_column(column: str) -> str:
    """Format a database column name for display in chart labels."""
    return column.replace("_", " ").strip().title()


def default_value_index(numeric_columns: list[str], generated_sql: str | None) -> int:
    """Return the index of the ORDER BY column, so the value selectbox opens on the
    metric the query actually ranked by - not the first numeric column returned.

    st.selectbox without an explicit index defaults to position 0, i.e. the leftmost
    numeric column in the SELECT list. For a query like
    "SELECT ..., ftm AS ft_made, ROUND(...) AS ft_pct ... ORDER BY ft_pct DESC",
    that silently opens on ft_made - a raw count - while the chart's own title still
    reads "ft_pct nach player_name", making the bar lengths and sort order answer a
    different question than the one asked. Verified against live data: a "best free
    throw percentage, min 50 attempts" question correctly selected the right 10
    players via ft_pct, then displayed and sorted them by ft_made instead - the true
    #1 by percentage (91.0%, 61 makes) rendered near the bottom, while a lower-ranked
    player with more raw makes (89.1%, 139 makes) rendered at the top.
    """
    if not generated_sql:
        return 0
    match = re.search(r"order\s+by\s+([a-zA-Z_]\w*)", generated_sql, re.IGNORECASE)
    if not match:
        return 0
    target = match.group(1).lower()
    for index, column in enumerate(numeric_columns):
        if column.lower() == target:
            return index
    return 0


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


def render_line_chart(df: pd.DataFrame, numeric_columns: list[str], generated_sql: str | None = None) -> None:
    """Render a line chart for date/time series results."""
    date_columns = list(df.select_dtypes(include="datetime").columns)
    date_column = st.selectbox("Datum/Zeit", date_columns, key=f"line_date_{id(df)}")
    value_column = st.selectbox(
        "Wert", numeric_columns, index=default_value_index(numeric_columns, generated_sql), key=f"line_value_{id(df)}"
    )
    chart_df = df[[date_column, value_column]].dropna().sort_values(date_column).head(500)
    if chart_df.empty:
        st.caption("Keine Zeilen für ein Liniendiagramm.")
        return
    chart = (
        alt.Chart(chart_df)
        .mark_line(color=CHART_ACCENT, strokeWidth=2, point=alt.OverlayMarkDef(color=CHART_ACCENT, size=45))
        .encode(
            x=alt.X(f"{date_column}:T", title=humanize_column(date_column)),
            y=alt.Y(f"{value_column}:Q", title=humanize_column(value_column)),
            tooltip=[
                alt.Tooltip(f"{date_column}:T", title=humanize_column(date_column)),
                alt.Tooltip(f"{value_column}:Q", title=humanize_column(value_column)),
            ],
        )
        .properties(title=f"{humanize_column(value_column)} über {humanize_column(date_column)}")
    )
    st.altair_chart(style_chart(chart), use_container_width=True)


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
        st.caption("Keine Zeilen für ein Streudiagramm.")
        return
    chart = (
        alt.Chart(chart_df)
        .mark_circle(color=CHART_ACCENT, size=70, opacity=0.55)
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
    st.altair_chart(style_chart(chart), use_container_width=True)


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


def render_histogram(df: pd.DataFrame, numeric_columns: list[str], generated_sql: str | None = None) -> None:
    """Render a histogram for one numeric result column."""
    value_column = st.selectbox(
        "Wert", numeric_columns, index=default_value_index(numeric_columns, generated_sql), key=f"hist_value_{id(df)}"
    )
    chart_df = df[[value_column]].dropna().head(5000)
    if chart_df.empty:
        st.caption("Keine Zeilen für ein Histogramm.")
        return

    chart = (
        alt.Chart(chart_df)
        .mark_bar(color=CHART_ACCENT, cornerRadiusEnd=2)
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
    st.altair_chart(style_chart(chart), use_container_width=True)


def main() -> None:
    """Run the Streamlit basketball analytics chat application."""
    ensure_database()
    inject_styles()
    llm_config = sidebar()

    st.title("BasketballGPT")
    st.write("Stell Fragen zu den Basketball-Daten in bronze, silver und gold.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                # The three examples used to be repeated here as inline code,
                # which broke the sentence into blocks and duplicated what the
                # starter buttons above already show.
                "content": (
                    "Frag nach Spielern, Teams, Spielen oder Statistiken aus BBL, "
                    "EuroLeague, EuroCup und Champions League. "
                    "Die Buttons oben sind ein Startpunkt."
                ),
            }
        ]

    starter_question = render_starter_questions()
    render_chat()

    typed_question = st.chat_input("Frag nach Spielern, Teams, Spielen oder Statistiken")
    question = starter_question or typed_question
    if not question:
        return
    intent = classify_question(question)

    # One id per answer, minted before the answer is rendered and stored with it.
    # The feedback buttons used to be keyed "current_answer" while live and
    # "history_<index>" once replayed, so the first click landed on a widget the
    # rerun then destroyed - the event was dropped and the button needed a second
    # press. A stable key makes both render passes the same widget.
    feedback_id = f"fb_{uuid4().hex[:12]}"

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Erzeuge SQL und frage PostgreSQL ab …"):
            _rows = None
            try:
                answer, sql, _rows = answer_question(question, **llm_config)
            except MissingApiKeyError as exc:
                answer = str(exc)
                sql = None
                st.error(answer)
            except UnsafeSqlError as exc:
                answer = f"Ich habe das erzeugte SQL abgelehnt: {exc}"
                sql = None
                st.error(answer)
            except QueryEngineError as exc:
                answer = str(exc)
                sql = None
                st.error(answer)
            except Exception as exc:
                answer = f"LLM- oder Datenbankabfrage fehlgeschlagen: {exc}"
                sql = None
                st.error(answer)
            else:
                st.markdown(answer)
                with st.expander("Erzeugtes SQL"):
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
                    key_prefix=feedback_id,
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sql": sql,
            "rows": _rows if sql else None,
            "question": question,
            "intent": intent,
            "feedback_id": feedback_id,
        }
    )


if __name__ == "__main__":
    main()
