from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from slothshot import renaming
from slothshot.db import connect_database
from slothshot.indexer import index_image
from slothshot.renaming import rename_images, rename_images_with_vision
from slothshot.vision import OllamaTimeoutError


class Reader:
    def read(self, _path: Path) -> SimpleNamespace:
        return SimpleNamespace(text="screen content", average_confidence=1.0, error=None)


def test_locked_rename_retries_then_stays_pending_and_counts_locked(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    source = folder / "locked.png"
    Image.new("RGB", (24, 16), "white").save(source)
    database = tmp_path / "data.sqlite3"
    connection = connect_database(database)
    assert index_image(connection, source, Reader()) == "indexed"
    image_id = int(
        connection.execute(
            "SELECT id FROM images WHERE path = ?", (str(source.resolve()),)
        ).fetchone()["id"]
    )
    connection.close()

    move_attempts = 0
    sleeps: list[float] = []

    def locked(*_args, **_kwargs):
        nonlocal move_attempts
        move_attempts += 1
        raise PermissionError("image is still open")

    monkeypatch.setattr(renaming, "describe_screenshot", lambda *_args: "New Name")
    monkeypatch.setattr(renaming, "rename_indexed_image", locked)
    monkeypatch.setattr(renaming.time, "sleep", sleeps.append)

    summary = rename_images_with_vision(database, folder, "test-model", image_ids=[image_id])

    assert move_attempts == 6  # Initial try plus five retries.
    assert sleeps == [1.25] * 5
    assert summary.renamed == 0
    assert summary.locked == 1
    assert summary.model_errors == 0
    assert summary.pending == 1
    assert summary.last_error is not None
    assert "Close any image viewer" in summary.last_error
    assert source.is_file()
    check = connect_database(database)
    try:
        assert (
            check.execute("SELECT name_state FROM images WHERE id = ?", (image_id,)).fetchone()[0]
            == "pending"
        )
    finally:
        check.close()


def test_missing_screenshot_removes_stale_index_row_without_failing(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    source = folder / "removed.png"
    Image.new("RGB", (24, 16), "white").save(source)
    database = tmp_path / "data.sqlite3"
    connection = connect_database(database)
    assert index_image(connection, source, Reader()) == "indexed"
    image_id = int(
        connection.execute(
            "SELECT id FROM images WHERE path = ?", (str(source.resolve()),)
        ).fetchone()["id"]
    )
    connection.close()
    source.unlink()

    def should_not_describe(*_args, **_kwargs):
        raise AssertionError("A missing screenshot should be removed before model inference.")

    monkeypatch.setattr(renaming, "describe_screenshot", should_not_describe)
    summary = rename_images_with_vision(database, folder, "test-model", image_ids=[image_id])

    assert summary.missing == 1
    assert summary.failed == 0
    assert summary.pending == 0
    check = connect_database(database)
    try:
        assert check.execute("SELECT 1 FROM images WHERE id = ?", (image_id,)).fetchone() is None
        assert (
            check.execute(
                "SELECT 1 FROM image_name_claims WHERE image_id = ?", (image_id,)
            ).fetchone()
            is None
        )
    finally:
        check.close()


def test_ocr_naming_renames_without_calling_ollama(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    source = folder / "original.png"
    Image.new("RGB", (24, 16), "white").save(source)
    database = tmp_path / "data.sqlite3"
    connection = connect_database(database)
    assert index_image(connection, source, Reader()) == "indexed"
    connection.close()

    def ollama_must_not_run(*_args, **_kwargs):
        raise AssertionError("OCR naming must not contact Ollama.")

    monkeypatch.setattr(renaming, "suggest_name_from_text", ollama_must_not_run)
    monkeypatch.setattr(renaming, "describe_screenshot", ollama_must_not_run)

    summary = rename_images(database, folder)

    destination = folder / "Screen Content.png"
    assert summary.renamed == 1
    assert summary.failed == 0
    assert destination.is_file()
    assert not source.exists()


def test_text_naming_uses_configured_text_model_and_ocr_only(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    source = folder / "original.png"
    Image.new("RGB", (24, 16), "white").save(source)
    database = tmp_path / "data.sqlite3"
    connection = connect_database(database)
    assert index_image(connection, source, Reader()) == "indexed"
    connection.close()
    received: list[tuple[str, str, int]] = []

    def text_suggestion(ocr_text: str, model_name: str, timeout_seconds: int) -> str:
        received.append((ocr_text, model_name, timeout_seconds))
        return "Text Model Name"

    def vision_must_not_run(*_args, **_kwargs):
        raise AssertionError("Text naming must not inspect the screenshot pixels.")

    monkeypatch.setattr(renaming, "suggest_name_from_text", text_suggestion)
    monkeypatch.setattr(renaming, "describe_screenshot", vision_must_not_run)

    summary = rename_images(
        database,
        folder,
        naming="text",
        text_model="test-text-model",
        vision_model="unused-vision-model",
        vision_timeout_seconds=31,
    )

    assert received == [("screen content", "test-text-model", 31)]
    assert summary.renamed == 1
    assert (folder / "Text Model Name.png").is_file()


def test_vision_timeout_stops_after_first_request_and_keeps_remaining_pending(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    sources = [folder / "a.png", folder / "b.png"]
    database = tmp_path / "data.sqlite3"
    connection = connect_database(database)
    for source, color in zip(sources, ("white", "red"), strict=True):
        Image.new("RGB", (24, 16), color).save(source)
        assert index_image(connection, source, Reader()) == "indexed"
    connection.close()

    requests: list[Path] = []

    def timed_out(path: Path, *_args, **_kwargs):
        requests.append(path)
        raise OllamaTimeoutError("Ollama timed out after 7 s.")

    monkeypatch.setattr(renaming, "describe_screenshot", timed_out)

    summary = rename_images(
        database,
        folder,
        naming="vision",
        vision_model="test-model",
        vision_timeout_seconds=7,
        pending_only=False,
        indexed_only=True,
    )

    assert len(requests) == 1
    assert summary.timed_out is True
    assert summary.failed == 1
    assert summary.model_errors == 1
    assert summary.pending == 2
    assert summary.last_error is not None
    assert "stopping this run" in summary.last_error
    check = connect_database(database)
    try:
        states = check.execute("SELECT name_state FROM images ORDER BY path").fetchall()
        assert [row[0] for row in states] == ["pending", "pending"]
        assert check.execute("SELECT COUNT(*) FROM image_name_claims").fetchone()[0] == 0
    finally:
        check.close()
