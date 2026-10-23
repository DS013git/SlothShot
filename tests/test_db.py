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
            "id",
            "path",
            "sha256",
            "width",
            "height",
            "ocr_text",
            "avg_confidence",
            "file_mtime",
            "indexed_at",
            "status",
            "name_state",
        } <= columns
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'embeddings'"
            ).fetchone()
            is not None
        )
        image_id = _insert_image(connection)
        assert (
            connection.execute(
                "SELECT name_state FROM images WHERE id = ?", (image_id,)
            ).fetchone()[0]
            == "pending"
        )
    finally:
        connection.close()


def test_fts_triggers_keep_text_and_path_synchronized(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "screenshots.sqlite3")
    try:
        image_id = _insert_image(connection)
        assert (
            connection.execute(
                "SELECT rowid FROM images_fts WHERE images_fts MATCH 'address'"
            ).fetchone()[0]
            == image_id
        )

        connection.execute(
            "UPDATE images SET ocr_text = 'docker port error', "
            "path = 'C:/shots/two.png' WHERE id = ?",
            (image_id,),
        )
        assert (
            connection.execute(
                "SELECT rowid FROM images_fts WHERE images_fts MATCH 'docker'"
            ).fetchone()[0]
            == image_id
        )
        assert (
            connection.execute(
                "SELECT rowid FROM images_fts WHERE images_fts MATCH 'two'"
            ).fetchone()[0]
            == image_id
        )
        assert (
            connection.execute(
                "SELECT rowid FROM images_fts WHERE images_fts MATCH 'address'"
            ).fetchone()
            is None
        )

        connection.execute("DELETE FROM images WHERE id = ?", (image_id,))
        assert (
            connection.execute(
                "SELECT rowid FROM images_fts WHERE images_fts MATCH 'docker'"
            ).fetchone()
            is None
        )
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


def test_phase_two_database_migrates_chunk_text_column(tmp_path: Path) -> None:
    database = tmp_path / "phase-two.sqlite3"
    raw = sqlite3.connect(database)
    raw.executescript(
        """
        CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);
        CREATE TABLE embeddings (
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (image_id, chunk_index, model_name)
        );
        PRAGMA user_version = 1;
        """
    )
    raw.close()

    connection = connect_database(database)
    try:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(embeddings)")}
        assert "chunk_text" in columns
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        connection.close()


def test_schema_two_migration_preserves_rows_and_marks_names_done(tmp_path: Path) -> None:
    database = tmp_path / "schema-two.sqlite3"
    raw = sqlite3.connect(database)
    raw.executescript(
        """
        CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);
        CREATE TABLE embeddings (
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            chunk_text TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (image_id, chunk_index, model_name)
        );
        INSERT INTO images VALUES (7, 'C:/shots/keep.png');
        INSERT INTO embeddings VALUES (7, 0, 'model', X'0102', 'kept chunk');
        PRAGMA user_version = 2;
        """
    )
    raw.close()

    connection = connect_database(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert tuple(connection.execute("SELECT id, path, name_state FROM images").fetchone()) == (
            7,
            "C:/shots/keep.png",
            "done",
        )
        assert tuple(
            connection.execute("SELECT image_id, embedding, chunk_text FROM embeddings").fetchone()
        ) == (7, b"\x01\x02", "kept chunk")
    finally:
        connection.close()


def test_schema_one_migration_runs_both_steps_and_preserves_data(tmp_path: Path) -> None:
    database = tmp_path / "schema-one.sqlite3"
    raw = sqlite3.connect(database)
    raw.executescript(
        """
        CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);
        CREATE TABLE embeddings (
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (image_id, chunk_index, model_name)
        );
        INSERT INTO images VALUES (9, 'C:/shots/legacy.png');
        INSERT INTO embeddings VALUES (9, 0, 'model', X'0304');
        PRAGMA user_version = 1;
        """
    )
    raw.close()

    connection = connect_database(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert (
            connection.execute("SELECT name_state FROM images WHERE id = 9").fetchone()[0] == "done"
        )
        assert (
            connection.execute("SELECT chunk_text FROM embeddings WHERE image_id = 9").fetchone()[0]
            == ""
        )
        assert (
            connection.execute("SELECT embedding FROM embeddings WHERE image_id = 9").fetchone()[0]
            == b"\x03\x04"
        )
    finally:
        connection.close()
