"""Hash reuse, idempotency, and safe cleanup tests for folder scans."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from slothshot.db import connect_database
from slothshot.indexer import scan_folder


class FakeReader:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def read(self, image_path: Path) -> SimpleNamespace:
        self.calls.append(image_path)
        return SimpleNamespace(
            text=f"words from {image_path.stem}", average_confidence=0.8, error=None
        )


class FakeEmbedder:
    model_name = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        self.calls += 1
        return [np.array([1.0, float(len(texts))], dtype=np.float32) for _ in texts]


def _save(path: Path, color: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 60), color).save(path)
    return path


def _open_db(tmp_path: Path):
    return connect_database(tmp_path / "screenshots.sqlite3")


def test_scan_indexes_once_then_skips_unchanged_bytes(tmp_path: Path) -> None:
    folder = tmp_path / "screenshots"
    _save(folder / "one.png", "white")
    connection = _open_db(tmp_path)
    reader = FakeReader()
    try:
        first = scan_folder(folder, connection, reader)
        second = scan_folder(folder, connection, reader)

        assert first.found == first.indexed == 1
        assert second.skipped == 1
        assert len(reader.calls) == 1
        assert connection.execute("SELECT count(*) FROM images").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT rowid FROM images_fts WHERE images_fts MATCH 'words'"
            ).fetchone()
            is not None
        )
    finally:
        connection.close()


def test_renamed_and_copied_images_reuse_cached_ocr(tmp_path: Path) -> None:
    folder = tmp_path / "screenshots"
    original = _save(folder / "old.png", "white")
    connection = _open_db(tmp_path)
    reader = FakeReader()
    try:
        scan_folder(folder, connection, reader)
        original.rename(folder / "renamed.png")
        (folder / "copy.png").write_bytes((folder / "renamed.png").read_bytes())

        result = scan_folder(folder, connection, reader)

        assert result.reused == 2
        assert result.removed == 1
        assert len(reader.calls) == 1
        rows = connection.execute("SELECT path, ocr_text FROM images ORDER BY path").fetchall()
        assert len(rows) == 2
        assert all(row["ocr_text"] == "words from old" for row in rows)
    finally:
        connection.close()


def test_renamed_images_reuse_cached_vectors_without_reembedding(tmp_path: Path) -> None:
    folder = tmp_path / "screenshots"
    original = _save(folder / "old.png", "white")
    connection = _open_db(tmp_path)
    reader = FakeReader()
    embedder = FakeEmbedder()
    try:
        scan_folder(folder, connection, reader, embedder)
        original.rename(folder / "renamed.png")

        result = scan_folder(folder, connection, reader, embedder)

        assert result.reused == 1
        assert result.removed == 1
        assert len(reader.calls) == 1
        assert embedder.calls == 1
        row = connection.execute("SELECT image_id, chunk_text FROM embeddings").fetchone()
        image = connection.execute("SELECT id FROM images").fetchone()
        assert row["image_id"] == image["id"]
        assert row["chunk_text"] == "words from old"
    finally:
        connection.close()


def test_changed_content_is_reprocessed(tmp_path: Path) -> None:
    folder = tmp_path / "screenshots"
    image = _save(folder / "one.png", "white")
    connection = _open_db(tmp_path)
    reader = FakeReader()
    try:
        scan_folder(folder, connection, reader)
        _save(image, "black")
        result = scan_folder(folder, connection, reader)

        assert result.indexed == 1
        assert len(reader.calls) == 2
    finally:
        connection.close()


def test_cleanup_is_scoped_and_empty_folder_removes_its_rows(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _save(left / "one.png", "white")
    _save(right / "two.png", "black")
    connection = _open_db(tmp_path)
    reader = FakeReader()
    try:
        scan_folder(left, connection, reader)
        scan_folder(right, connection, reader)
        (left / "one.png").unlink()
        result = scan_folder(left, connection, reader)

        assert result.found == 0
        assert result.removed == 1
        assert [row["path"] for row in connection.execute("SELECT path FROM images")] == [
            str((right / "two.png").resolve())
        ]
    finally:
        connection.close()


def test_non_images_and_zero_byte_images_are_handled(tmp_path: Path) -> None:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    (folder / "notes.txt").write_text("not a screenshot", encoding="utf-8")
    (folder / "empty.png").write_bytes(b"")
    connection = _open_db(tmp_path)
    reader = FakeReader()
    try:
        summary = scan_folder(folder, connection, reader)

        assert summary.found == 1
        assert summary.indexed == 1
        assert summary.failed == 0
        assert len(reader.calls) == 1
        assert connection.execute("SELECT width FROM images").fetchone()[0] == 0
    finally:
        connection.close()


def test_missing_folder_does_not_delete_any_indexed_paths(tmp_path: Path) -> None:
    folder = tmp_path / "screenshots"
    _save(folder / "one.png", "white")
    connection = _open_db(tmp_path)
    try:
        scan_folder(folder, connection, FakeReader())
        try:
            scan_folder(tmp_path / "missing", connection, FakeReader())
        except NotADirectoryError:
            pass
        else:
            raise AssertionError("missing scan folder should be rejected")
        assert connection.execute("SELECT count(*) FROM images").fetchone()[0] == 1
    finally:
        connection.close()
