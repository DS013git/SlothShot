"""Safe keyword search over the local FTS5 index."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """One ranked screenshot match returned by a search mode."""

    image_id: int
    path: str
    score: float
    snippet: str


def sanitize_query(query: str) -> tuple[str, ...]:
    """Keep only searchable Unicode words, safely removing FTS operators.

    The tokenizer is deliberately small: terms are always quoted in the FTS
    expression, so user input can never become a column filter or FTS operator.
    """
    return tuple(re.findall(r"[^\W_]+", query, flags=re.UNICODE)[:16])


def _build_fts_query(terms: tuple[str, ...], *, operator: str) -> str:
    """Quote terms and add FTS5's suffix wildcard for prefix matching."""
    if operator not in {"AND", "OR"}:
        raise ValueError("FTS term operator must be AND or OR")
    quoted = [f'"{term}"*' for term in terms]
    return f" {operator} ".join(quoted)


def search_keyword(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
) -> list[SearchResult]:
    """Search exact terms and prefixes with BM25, falling back to OR if needed.

    AND gives short, precise queries their expected behavior. If that has no
    hits, an OR retry lets longer natural-language queries still return useful
    candidates instead of an empty page. Lower BM25 scores rank first.
    """
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    terms = sanitize_query(query)
    if not terms:
        return []

    results = _run_search(connection, _build_fts_query(terms, operator="AND"), limit)
    if results or len(terms) == 1:
        return results
    return _run_search(connection, _build_fts_query(terms, operator="OR"), limit)


def _run_search(
    connection: sqlite3.Connection,
    fts_query: str,
    limit: int,
) -> list[SearchResult]:
    """Run one parameterized FTS query and convert rows to plain results."""
    rows = connection.execute(
        """SELECT images.id, images.path, bm25(images_fts) AS score,
                  snippet(images_fts, 1, '[', ']', ' … ', 18) AS snippet
           FROM images_fts
           JOIN images ON images.id = images_fts.rowid
           WHERE images_fts MATCH ?
           ORDER BY score ASC, images.id ASC
           LIMIT ?""",
        (fts_query, limit),
    ).fetchall()
    return [
        SearchResult(
            image_id=int(row["id"]),
            path=str(row["path"]),
            score=float(row["score"]),
            snippet=str(row["snippet"]),
        )
        for row in rows
    ]
