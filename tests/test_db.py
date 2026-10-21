"""SQLite schema and trigger behavior tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from slothshot.db import SCHEMA_VERSION, connect_database


def _insert_image(connection: sqlite3.Connection, path: str = "C:/shots/one.png") -> int:
    cursor = connection.execute(
        """INSERT INTO images (
               path, sha256, width, height, ocr_text, avg_confidence,
               file_mtime, indexed_at, status
           ) VALUES (?, 'hash', 100, 50, 'address already in use', 0.9, 1.0,
                     '2026-10-21T00:00:00+00:00', 'indexed')""",
        (path,),
    )
    return int(cursor.lastrowid)


def test_database_uses_wal_and_creates_versioned_schema(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "db" / "screenshots.sqlite3")
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(images)")}
        assert {
            "id", "path", "sha256", "width", "height", "ocr_text",
            "avg_confidence", "file_mtime", "indexed_at", "status",
        } <= columns
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'embeddings'"
        ).fetchone() is not None
    finally:
        connection.close()


def test_fts_triggers_keep_text_and_path_synchronized(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "screenshots.sqlite3")
    try:
        image_id = _insert_image(connection)
        assert connection.execute(
            "SELECT rowid FROM images_fts WHERE images_fts MATCH 'address'"
        ).fetchone()[0] == image_id

        connection.execute(
            "UPDATE images SET ocr_text = 'docker port error', "
            "path = 'C:/shots/two.png' WHERE id = ?",
            (image_id,),
        )
        assert connection.execute(
            "SELECT rowid FROM images_fts WHERE images_fts MATCH 'docker'"
        ).fetchone()[0] == image_id
        assert connection.execute(
            "SELECT rowid FROM images_fts WHERE images_fts MATCH 'two'"
        ).fetchone()[0] == image_id
        assert connection.execute(
            "SELECT rowid FROM images_fts WHERE images_fts MATCH 'address'"
        ).fetchone() is None

        connection.execute("DELETE FROM images WHERE id = ?", (image_id,))
        assert connection.execute(
            "SELECT rowid FROM images_fts WHERE images_fts MATCH 'docker'"
        ).fetchone() is None
    finally:
        connection.close()


def test_embeddings_cascade_when_an_image_is_deleted(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "screenshots.sqlite3")
    try:
        image_id = _insert_image(connection)
        connection.execute(
            "INSERT INTO embeddings(image_id, chunk_index, model_name, embedding) "
            "VALUES (?, 0, 'test-model', X'0000')",
            (image_id,),
        )
        connection.execute("DELETE FROM images WHERE id = ?", (image_id,))
        assert connection.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    finally:
        connection.close()


def test_newer_schema_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 999")
    connection.close()

    with pytest.raises(RuntimeError, match="newer than supported"):
        connect_database(database)
