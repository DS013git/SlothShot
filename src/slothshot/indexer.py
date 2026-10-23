"""Hash-aware folder scanning and one-image-at-a-time indexing."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from PIL import Image, UnidentifiedImageError

from slothshot.db import image_by_hash, image_by_path
from slothshot.embed import (
    EmbeddingModel,
    reuse_embeddings_for_hash,
    save_image_embeddings,
)

logger = logging.getLogger("slothshot.indexer")
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IGNORED_SUFFIXES = {".part", ".partial", ".tmp", ".temp"}
HASH_CHUNK_SIZE = 1024 * 1024


class ImageReader(Protocol):
    """Small interface implemented by OCRReader and test doubles."""

    def read(self, image_path: Path) -> object: ...


@dataclass(frozen=True)
class IndexSummary:
    """Counts describing the result of a complete folder scan."""

    found: int = 0
    indexed: int = 0
    reused: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large screenshots use little memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Read dimensions without retaining decoded pixels in memory."""
    with Image.open(path) as image:
        return image.size


def _candidate_images(folder: Path) -> list[Path]:
    """Return supported images recursively in deterministic path order."""
    paths = [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and path.suffix.lower() not in IGNORED_SUFFIXES
    ]
    return sorted(paths, key=lambda path: str(path).casefold())


def _write_image(
    connection: sqlite3.Connection,
    *,
    path: Path,
    sha256: str,
    file_mtime: float,
    width: int,
    height: int,
    ocr_text: str,
    confidence: float,
    status: str,
) -> None:
    """Insert or update one image row and commit it as its own unit of work."""
    normalized = str(path.resolve())
    indexed_at = datetime.now(UTC).isoformat(timespec="seconds")
    with connection:
        connection.execute(
            """INSERT INTO images (
                   path, sha256, width, height, ocr_text, avg_confidence,
                   file_mtime, indexed_at, status, name_state
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
               ON CONFLICT(path) DO UPDATE SET
                   sha256 = excluded.sha256,
                   width = excluded.width,
                   height = excluded.height,
                   ocr_text = excluded.ocr_text,
                   avg_confidence = excluded.avg_confidence,
                   file_mtime = excluded.file_mtime,
                   indexed_at = excluded.indexed_at,
                   status = excluded.status,
                   name_state = 'pending'""",
            (
                normalized,
                sha256,
                width,
                height,
                ocr_text,
                confidence,
                file_mtime,
                indexed_at,
                status,
            ),
        )


def _delete_embeddings_for_path(connection: sqlite3.Connection, path: Path) -> None:
    """Drop stale vectors when the pixels at an existing path change."""
    row = image_by_path(connection, path)
    if row is not None:
        with connection:
            connection.execute("DELETE FROM embeddings WHERE image_id = ?", (row["id"],))


def index_image(connection: sqlite3.Connection, path: Path, reader: ImageReader) -> str:
    """Index one file, returning ``indexed``, ``reused``, ``skipped``, or ``failed``.

    OCR runs before opening a database write transaction. A process interruption
    can therefore lose at most the current image; earlier images stay committed.
    """
    path = path.resolve()
    try:
        stat = path.stat()
        sha256 = sha256_file(path)
    except OSError:
        logger.exception("Could not read or hash %s", path)
        return "failed"

    existing = image_by_path(connection, path)
    if existing is not None and existing["sha256"] == sha256:
        # File timestamps can change without the pixels changing. Keep metadata
        # current while avoiding an expensive OCR pass.
        with connection:
            connection.execute(
                "UPDATE images SET file_mtime = ? WHERE id = ?", (stat.st_mtime, existing["id"])
            )
        return "skipped"

    duplicate = image_by_hash(connection, sha256)
    if duplicate is not None:
        if existing is not None:
            _delete_embeddings_for_path(connection, path)
        _write_image(
            connection,
            path=path,
            sha256=sha256,
            file_mtime=stat.st_mtime,
            width=duplicate["width"],
            height=duplicate["height"],
            ocr_text=duplicate["ocr_text"],
            confidence=duplicate["avg_confidence"],
            status=duplicate["status"],
        )
        return "reused"

    try:
        width, height = _image_dimensions(path)
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        logger.exception("Image could not be decoded: %s", path)
        width, height = 0, 0

    result = reader.read(path)
    if existing is not None:
        _delete_embeddings_for_path(connection, path)
    text = str(getattr(result, "text", ""))
    confidence = float(getattr(result, "average_confidence", 0.0))
    error = getattr(result, "error", None)
    _write_image(
        connection,
        path=path,
        sha256=sha256,
        file_mtime=stat.st_mtime,
        width=width,
        height=height,
        ocr_text=text,
        confidence=confidence,
        status="error" if error else "indexed",
    )
    if error:
        logger.warning("Stored OCR error for %s: %s", path, error)
        return "failed"
    return "indexed"


def scan_folder(
    folder: Path,
    connection: sqlite3.Connection,
    reader: ImageReader,
    embedder: EmbeddingModel | None = None,
    on_indexed: Callable[[int, Path, str], None] | None = None,
) -> IndexSummary:
    """Index supported images under ``folder`` and remove stale entries.

    Cleanup happens only after discovery and processing complete. If scanning
    raises midway, already committed image work remains safe and no rows are
    removed based on an incomplete view of the folder.
    """
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Screenshot folder does not exist or is not a directory: {root}")
    paths = _candidate_images(root)
    logger.info("Found %d supported images under %s", len(paths), root)

    counts = {"indexed": 0, "reused": 0, "skipped": 0, "failed": 0}
    discovered = {str(path.resolve()) for path in paths}
    newly_indexed: list[Path] = []
    for path in paths:
        outcome = index_image(connection, path, reader)
        counts[outcome] += 1
        if outcome in {"indexed", "reused"}:
            newly_indexed.append(path.resolve())
        if embedder is not None and outcome != "failed":
            row = image_by_path(connection, path)
            if row is not None and row["status"] == "indexed":
                try:
                    if outcome == "reused":
                        reuse_embeddings_for_hash(
                            connection, image_id=int(row["id"]), sha256=str(row["sha256"])
                        )
                    save_image_embeddings(
                        connection,
                        image_id=int(row["id"]),
                        text=str(row["ocr_text"]),
                        model=embedder,
                    )
                except sqlite3.IntegrityError:
                    logger.exception("Could not save embeddings for %s", path)
                    counts["failed"] += 1

    # Limit cleanup to descendants of this root; indexing another directory
    # must never remove records owned by a different scan.
    prefix = str(root).rstrip("\\/") + path_separator(root)
    stale = connection.execute(
        "SELECT id, path FROM images WHERE path = ? OR path LIKE ? ESCAPE '\\'",
        (str(root), _like_escape(prefix) + "%"),
    ).fetchall()
    removed = 0
    for row in stale:
        stored_path = row["path"]
        if stored_path not in discovered:
            with connection:
                connection.execute("DELETE FROM images WHERE id = ?", (row["id"],))
            removed += 1

    if on_indexed is not None:
        for path in newly_indexed:
            row = image_by_path(connection, path)
            if row is not None and row["status"] == "indexed":
                on_indexed(int(row["id"]), path, str(row["ocr_text"] or ""))

    summary = IndexSummary(
        found=len(paths),
        indexed=counts["indexed"],
        reused=counts["reused"],
        skipped=counts["skipped"],
        failed=counts["failed"],
        removed=removed,
    )
    logger.info("Index scan finished: %s", summary)
    return summary


def path_separator(path: Path) -> str:
    """Return the separator used by the host path flavor."""
    return "\\" if "\\" in str(path) else "/"


def _like_escape(value: str) -> str:
    """Escape SQL LIKE metacharacters in a path prefix."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
