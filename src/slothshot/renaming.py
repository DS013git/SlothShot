"""Shared, claim-safe vision renaming for indexing and the file watcher."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from slothshot.db import connect_database
from slothshot.naming import (
    filename_from_description,
    rename_indexed_image,
    unique_filename,
)
from slothshot.vision import OllamaUnavailableError, VisionModelError, describe_screenshot

logger = logging.getLogger("slothshot.renaming")
CLAIM_STALE_SECONDS = 600
RENAME_RETRIES = 3
RENAME_RETRY_DELAY_SECONDS = 0.1


@dataclass(frozen=True)
class RenameSummary:
    renamed: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    unavailable: bool = False
    pending: int = 0
    locked: int = 0
    model_errors: int = 0
    last_error: str | None = None


def _rename_with_retries(
    database_path: Path,
    image_id: int,
    path: Path,
    new_name: str,
    root: Path,
    before_move: Callable[[Path, Path], None] | None,
) -> None:
    """Retry transient Windows sharing violations before leaving a name pending."""
    for attempt in range(RENAME_RETRIES + 1):
        try:
            rename_indexed_image(
                database_path,
                image_id,
                path,
                new_name,
                root,
                before_move=before_move,
            )
            return
        except PermissionError:
            if attempt == RENAME_RETRIES:
                raise
            time.sleep(RENAME_RETRY_DELAY_SECONDS * (attempt + 1))


def _claim_image(database_path: Path, image_id: int, *, pending_only: bool) -> bool:
    connection = connect_database(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "DELETE FROM image_name_claims WHERE claimed_at < ?",
            (time.time() - CLAIM_STALE_SECONDS,),
        )
        cursor = connection.execute(
            """INSERT OR IGNORE INTO image_name_claims(image_id, claimed_at)
               SELECT id, ? FROM images
               WHERE id = ? AND (? = 0 OR name_state = 'pending')""",
            (time.time(), image_id, int(pending_only)),
        )
        connection.commit()
        return cursor.rowcount == 1
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _release_claim(database_path: Path, image_id: int) -> None:
    connection = connect_database(database_path)
    try:
        with connection:
            connection.execute("DELETE FROM image_name_claims WHERE image_id = ?", (image_id,))
    finally:
        connection.close()


def _finish_unchanged(database_path: Path, image_id: int) -> None:
    connection = connect_database(database_path)
    try:
        with connection:
            connection.execute("UPDATE images SET name_state = 'done' WHERE id = ?", (image_id,))
            connection.execute("DELETE FROM image_name_claims WHERE image_id = ?", (image_id,))
    finally:
        connection.close()


def rename_images_with_vision(
    database_path: Path,
    watch_folder: Path,
    model_name: str,
    *,
    vision_timeout_seconds: int = 600,
    image_ids: Iterable[int] | None = None,
    pending_only: bool = True,
    indexed_only: bool = False,
    before_move: Callable[[Path, Path], None] | None = None,
) -> RenameSummary:
    """Vision-name eligible images, atomically claiming each one before work."""
    root = watch_folder.expanduser().resolve()
    connection = connect_database(database_path)
    try:
        query = "SELECT id, path, ocr_text FROM images WHERE 1 = 1"
        parameters: list[object] = []
        if pending_only:
            query += " AND name_state = 'pending'"
        if indexed_only:
            query += " AND status = 'indexed'"
        if image_ids is not None:
            ids = tuple(dict.fromkeys(int(image_id) for image_id in image_ids))
            if not ids:
                return RenameSummary()
            query += f" AND id IN ({','.join('?' for _ in ids)})"
            parameters.extend(ids)
        query += " ORDER BY path COLLATE NOCASE"
        rows = connection.execute(query, parameters).fetchall()
    finally:
        connection.close()

    renamed = unchanged = skipped = failed = locked = model_errors = 0
    unavailable = False
    last_error = None
    for row in rows:
        image_id = int(row["id"])
        path = Path(str(row["path"])).expanduser().resolve()
        if not path.is_relative_to(root):
            continue
        if not _claim_image(database_path, image_id, pending_only=pending_only):
            skipped += 1
            continue

        try:
            if not path.is_file():
                raise FileNotFoundError(f"Screenshot no longer exists: {path}")
            description = describe_screenshot(
                path,
                str(row["ocr_text"] or ""),
                model_name,
                vision_timeout_seconds,
            )
            new_name = unique_filename(filename_from_description(description, path), path)
            if new_name == path.name:
                _finish_unchanged(database_path, image_id)
                unchanged += 1
                continue
            _rename_with_retries(
                database_path,
                image_id,
                path,
                new_name,
                root,
                before_move,
            )
            renamed += 1
        except PermissionError as exc:
            failed += 1
            locked += 1
            last_error = f"Screenshot is locked: {path} ({exc})"
            logger.warning("Screenshot is locked; leaving its name pending: %s (%s)", path, exc)
        except OllamaUnavailableError as exc:
            failed += 1
            model_errors += 1
            unavailable = True
            last_error = str(exc)
            logger.error("Ollama is unavailable; pending renames will be retried: %s", exc)
            break
        except (OSError, ValueError, VisionModelError) as exc:
            failed += 1
            if isinstance(exc, VisionModelError):
                model_errors += 1
            last_error = str(exc)
            logger.error("Could not name screenshot %s: %s", path, exc)
        finally:
            # Successful operations already clear the claim transactionally.
            _release_claim(database_path, image_id)

        if unavailable:
            break

    connection = connect_database(database_path)
    try:
        remaining = connection.execute(
            "SELECT path FROM images WHERE name_state = 'pending'"
        ).fetchall()
    finally:
        connection.close()
    pending = sum(
        Path(str(row["path"])).expanduser().resolve().is_relative_to(root) for row in remaining
    )
    return RenameSummary(
        renamed=renamed,
        unchanged=unchanged,
        skipped=skipped,
        failed=failed,
        unavailable=unavailable,
        pending=pending,
        locked=locked,
        model_errors=model_errors,
        last_error=last_error,
    )
