from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from slothshot import renaming
from slothshot.db import connect_database
from slothshot.indexer import index_image
from slothshot.renaming import rename_images_with_vision


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
    assert source.is_file()
    check = connect_database(database)
    try:
        assert (
            check.execute("SELECT name_state FROM images WHERE id = ?", (image_id,)).fetchone()[0]
            == "pending"
        )
    finally:
        check.close()
