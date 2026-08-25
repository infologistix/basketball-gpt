from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_KNOWLEDGE_PATH = Path(__file__).parent / "knowledge" / "sql_examples.json"
DEFAULT_REJECTED_PATH = Path(__file__).parent / "knowledge" / "rejected_sql_examples.json"
TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+")
SQL_TABLE_PATTERN = re.compile(
    r"\b(?:from|join)\s+(?:(?:only)\s+)?\"?([a-zA-Z_][\w]*)\"?\.\"?([a-zA-Z_][\w]*)\"?",
    re.IGNORECASE,
)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "me",
    "of",
    "or",
    "show",
    "table",
    "the",
    "to",
    "with",
}


def knowledge_path() -> Path:
    """Return the configured JSON knowledge base path."""
    return Path(os.getenv("SQL_RAG_KNOWLEDGE_PATH", DEFAULT_KNOWLEDGE_PATH))


def rejected_path() -> Path:
    """Return the configured rejected-example JSON path."""
    return Path(os.getenv("SQL_RAG_REJECTED_PATH", DEFAULT_REJECTED_PATH))


def rag_enabled() -> bool:
    """Return whether lightweight SQL retrieval is enabled."""
    return os.getenv("SQL_RAG_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def max_examples() -> int:
    """Return the maximum number of retrieved examples to inject."""
    raw_value = os.getenv("SQL_RAG_MAX_EXAMPLES", "4")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 4


@lru_cache(maxsize=4)
def load_entries(path: str | None = None) -> list[dict[str, Any]]:
    """Load curated SQL examples and documentation snippets from JSON."""
    source = Path(path) if path else knowledge_path()
    return read_json_entries(source)


@lru_cache(maxsize=4)
def load_rejected_entries(path: str | None = None) -> list[dict[str, Any]]:
    """Load rejected SQL examples from JSON."""
    source = Path(path) if path else rejected_path()
    return read_json_entries(source)


def read_json_entries(source: Path) -> list[dict[str, Any]]:
    """Read a JSON list of dictionary entries from disk."""
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def tokenize(text: str | None) -> list[str]:
    """Tokenize text for simple local retrieval scoring."""
    if not text:
        return []
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def entry_text(entry: dict[str, Any]) -> str:
    """Return searchable text for a knowledge-base entry."""
    parts = [
        str(entry.get("question", "")),
        str(entry.get("sql", "")),
        str(entry.get("doc", "")),
        " ".join(str(tag) for tag in entry.get("tags", []) if tag),
        " ".join(str(table) for table in entry.get("tables", []) if table),
    ]
    return "\n".join(parts)


def score_entry(question_tokens: set[str], entry: dict[str, Any]) -> float:
    """Score one entry against a user question using token overlap."""
    if not question_tokens:
        return 0.0

    tokens = tokenize(entry_text(entry))
    if not tokens:
        return 0.0

    entry_tokens = set(tokens)
    overlap = question_tokens & entry_tokens
    if not overlap:
        return 0.0

    rare_token_boost = sum(1.0 + math.log(1 + len(token)) / 4 for token in overlap)
    coverage = len(overlap) / max(1, len(question_tokens))
    table_boost = sum(2.5 for table in entry.get("tables", []) if str(table).lower() in question_tokens)
    tag_boost = sum(0.75 for tag in entry.get("tags", []) if str(tag).lower() in question_tokens)
    return rare_token_boost + coverage + table_boost + tag_boost


def retrieve_examples(question: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Return the most relevant curated examples for a user question."""
    if not rag_enabled():
        return []
    max_results = max_examples() if limit is None else limit
    if max_results <= 0:
        return []

    question_tokens = set(tokenize(question))
    scored_entries = [
        (score_entry(question_tokens, entry), entry)
        for entry in load_entries()
    ]
    ranked = [entry for score, entry in sorted(scored_entries, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:max_results]


def retrieve_rejected_examples(question: str, limit: int = 2) -> list[dict[str, Any]]:
    """Return relevant rejected examples to warn the model away from them."""
    if not rag_enabled() or limit <= 0:
        return []
    question_tokens = set(tokenize(question))
    scored_entries = [
        (score_entry(question_tokens, entry), entry)
        for entry in load_rejected_entries()
    ]
    ranked = [entry for score, entry in sorted(scored_entries, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:limit]


def format_retrieved_context(question: str, limit: int | None = None) -> str:
    """Format retrieved examples for prompt injection."""
    examples = retrieve_examples(question, limit=limit)
    if not examples:
        return ""

    blocks = [
        "Relevant retrieved examples and notes:",
        "Use these as patterns when they match the question. Do not copy a table name blindly if the user asks for another competition/schema.",
    ]
    for index, example in enumerate(examples, start=1):
        blocks.append(f"\nExample {index}:")
        if example.get("question"):
            blocks.append(f"Question: {example['question']}")
        if example.get("doc"):
            blocks.append(f"Note: {example['doc']}")
        if example.get("sql"):
            blocks.append("SQL pattern:")
            blocks.append(str(example["sql"]).strip())
    return "\n".join(blocks)


def format_rejected_context(question: str, limit: int = 2) -> str:
    """Format rejected examples as anti-pattern guidance for the prompt."""
    examples = retrieve_rejected_examples(question, limit=limit)
    if not examples:
        return ""

    blocks = ["Relevant rejected examples:", "Avoid repeating these SQL patterns if they match the current question."]
    for index, example in enumerate(examples, start=1):
        blocks.append(f"\nRejected example {index}:")
        if example.get("question"):
            blocks.append(f"Question: {example['question']}")
        if example.get("reason"):
            blocks.append(f"Reason: {example['reason']}")
        if example.get("sql"):
            blocks.append("Rejected SQL:")
            blocks.append(str(example["sql"]).strip())
    return "\n".join(blocks)


def save_good_example(question: str, sql: str, answer: str | None = None) -> tuple[bool, str]:
    """Persist a user-approved question/SQL example for future retrieval."""
    entry = build_feedback_entry(
        question=question,
        sql=sql,
        tags=["saved", "good", *infer_tags(question)],
        doc="Saved from user feedback as a good SQL example.",
    )
    if answer:
        entry["answer"] = answer
    return append_json_entry(knowledge_path(), entry)


def save_bad_example(question: str, sql: str, reason: str | None = None) -> tuple[bool, str]:
    """Persist a user-rejected question/SQL example as an anti-pattern."""
    entry = build_feedback_entry(
        question=question,
        sql=sql,
        tags=["saved", "bad", *infer_tags(question)],
        doc="Saved from user feedback as a rejected SQL example.",
    )
    entry["reason"] = reason or "Marked as bad by user."
    return append_json_entry(rejected_path(), entry)


def build_feedback_entry(question: str, sql: str, tags: list[str], doc: str) -> dict[str, Any]:
    """Build a JSON-serializable feedback entry."""
    return {
        "question": question.strip(),
        "sql": sql.strip(),
        "tables": extract_sql_tables(sql),
        "tags": list(dict.fromkeys(tag for tag in tags if tag)),
        "doc": doc,
        "source": "streamlit_feedback",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def append_json_entry(path: Path, entry: dict[str, Any]) -> tuple[bool, str]:
    """Append an entry to a JSON list using an atomic file replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = read_json_entries(path)
    duplicate = any(
        existing.get("question") == entry.get("question")
        and existing.get("sql") == entry.get("sql")
        for existing in entries
    )
    if duplicate:
        return False, "This exact question and SQL are already saved."

    entries.append(entry)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(path)
    load_entries.cache_clear()
    load_rejected_entries.cache_clear()
    return True, f"Saved to {path}."


def extract_sql_tables(sql: str) -> list[str]:
    """Extract schema-qualified table names from FROM and JOIN clauses."""
    tables = [f"{schema}.{table}" for schema, table in SQL_TABLE_PATTERN.findall(sql or "")]
    return list(dict.fromkeys(tables))


def infer_tags(question: str) -> list[str]:
    """Infer simple retrieval tags from a user question."""
    lowered = question.lower()
    tags = []
    candidates = {
        "draw": ("draw", "chart", "graph", "plot", "scatter", "histogram"),
        "schema": ("schema", "column", "columns", "table", "tables"),
        "sample": ("sample", "preview", "rows"),
        "points": ("point", "points", "pts", "score"),
        "euroleague": ("euroleague", "el"),
        "eurocup": ("eurocup", "ec"),
        "bbl": ("bbl", "bundesliga"),
        "champions_league": ("champions league", "cl"),
        "silver": ("silver",),
        "gold": ("gold",),
        "bronze": ("bronze",),
    }
    for tag, words in candidates.items():
        if any(word in lowered for word in words):
            tags.append(tag)
    return tags
