"""SQLite storage for screenshot metadata, full-text search, and embeddings."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("slothshot.db")
SCHEMA_VERSION = 1


def connect_database(path: Path | str) -> sqlite3.Connection:
    """Open the local database and ensure its versioned schema exists.

    Each worker should open its own connection. WAL mode allows readers to keep
    using a stable snapshot while an indexer commits one image at a time.
    """
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    initialize_schema(connection)
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the current schema or reject a database newer than this app."""
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {version} is newer than supported version "
            f"{SCHEMA_VERSION}. Upgrade SlothShot before opening this database."
        )
    if version == SCHEMA_VERSION:
        return

    # FTS5 is an external-content index. Triggers keep it synchronized inside
    # the same SQLite transaction as each image row change.
    connection.executescript(
        """
        BEGIN;
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL,
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            ocr_text TEXT NOT NULL DEFAULT '',
            avg_confidence REAL NOT NULL DEFAULT 0.0,
            file_mtime REAL NOT NULL,
            indexed_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('indexed', 'error'))
        );
        CREATE INDEX IF NOT EXISTS images_sha256_idx ON images(sha256);
        CREATE INDEX IF NOT EXISTS images_indexed_at_idx ON images(indexed_at);

        CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
            path,
            ocr_text,
            content='images',
            content_rowid='id',
            tokenize='unicode61'
        );
        CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
            INSERT INTO images_fts(rowid, path, ocr_text)
            VALUES (new.id, new.path, new.ocr_text);
        END;
        CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
            INSERT INTO images_fts(images_fts, rowid, path, ocr_text)
            VALUES ('delete', old.id, old.path, old.ocr_text);
        END;
        CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE OF path, ocr_text ON images BEGIN
            INSERT INTO images_fts(images_fts, rowid, path, ocr_text)
            VALUES ('delete', old.id, old.path, old.ocr_text);
            INSERT INTO images_fts(rowid, path, ocr_text)
            VALUES (new.id, new.path, new.ocr_text);
        END;

        CREATE TABLE IF NOT EXISTS embeddings (
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (image_id, chunk_index, model_name)
        );
        INSERT INTO images_fts(images_fts) VALUES ('rebuild');
        PRAGMA user_version = 1;
        COMMIT;
        """
    )
    logger.info("Initialized SlothShot database schema version %d", SCHEMA_VERSION)


def image_by_path(connection: sqlite3.Connection, path: Path | str) -> sqlite3.Row | None:
    """Look up a single image using its normalized absolute path."""
    normalized = str(Path(path).resolve())
    return connection.execute("SELECT * FROM images WHERE path = ?", (normalized,)).fetchone()


def image_by_hash(connection: sqlite3.Connection, sha256: str) -> sqlite3.Row | None:
    """Return any previously indexed image with the same content hash."""
    return connection.execute(
        "SELECT * FROM images WHERE sha256 = ? ORDER BY id LIMIT 1", (sha256,)
    ).fetchone()
