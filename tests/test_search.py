"""Keyword query sanitization, ranking, and prefix matching tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from slothshot.db import connect_database
from slothshot.search import sanitize_query, search_keyword


def _add_image(connection: sqlite3.Connection, path: str, text: str) -> int:
    cursor = connection.execute(
        """INSERT INTO images (
               path, sha256, width, height, ocr_text, avg_confidence,
               file_mtime, indexed_at, status
           ) VALUES (?, ?, 400, 200, ?, 0.9, 1.0,
                     '2026-10-21T00:00:00+00:00', 'indexed')""",
        (path, path, text),
    )
    return int(cursor.lastrowid)


def test_sanitizer_discards_fts_syntax_and_keeps_unicode_words() -> None:
    assert sanitize_query('"OR" (docker-port) AND 银行 123 !') == (
        "OR", "docker", "port", "AND", "银行", "123"
    )
    assert sanitize_query("--- ... !!!") == ()


@pytest.mark.parametrize(
    "query",
    [
        '" OR ',
        "NEAR(foo bar)",
        "title:secret",
        "'broken quote",
        "foo OR bar NOT baz",
        "⚠️ !!! % ; DROP TABLE images; --",
    ],
)
def test_user_fts_syntax_is_treated_as_plain_terms(tmp_path: Path, query: str) -> None:
    connection = connect_database(tmp_path / "search.sqlite3")
    try:
        _add_image(connection, "C:/shots/one.png", "foo bar docker address")
        results = search_keyword(connection, query)
        assert isinstance(results, list)
        assert connection.execute("SELECT count(*) FROM images").fetchone()[0] == 1
    finally:
        connection.close()


def test_keyword_search_prefers_all_terms_and_marks_matching_snippet(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "search.sqlite3")
    try:
        expected_id = _add_image(
            connection, "C:/shots/docker.png", "docker port address already in use"
        )
        _add_image(connection, "C:/shots/port.png", "port configuration")

        results = search_keyword(connection, "docker port")

        assert [result.image_id for result in results] == [expected_id]
        assert "[docker]" in results[0].snippet
        assert "[port]" in results[0].snippet
        assert results[0].score > 0
    finally:
        connection.close()


def test_prefix_search_and_or_fallback_return_useful_results(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "search.sqlite3")
    try:
        prefix_id = _add_image(connection, "C:/shots/docker.png", "docker container")
        port_id = _add_image(connection, "C:/shots/port.png", "port already busy")

        prefix_results = search_keyword(connection, "dock")
        fallback_results = search_keyword(connection, "docker port error")

        assert prefix_results[0].image_id == prefix_id
        assert {result.image_id for result in fallback_results} == {prefix_id, port_id}
    finally:
        connection.close()


def test_empty_query_and_invalid_limit_are_handled(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "search.sqlite3")
    try:
        assert search_keyword(connection, "... !") == []
        with pytest.raises(ValueError, match="positive"):
            search_keyword(connection, "words", limit=0)
    finally:
        connection.close()
