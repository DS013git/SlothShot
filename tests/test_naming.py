"""Filename suggestion and filesystem rename behavior tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from slothshot.db import connect_database
from slothshot.indexer import index_image
from slothshot.naming import rename_indexed_image, suggest_filename


class Reader:
    def read(self, _path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            text="Docker address already in use", average_confidence=1.0, error=None
        )


def _indexed_image(tmp_path: Path) -> tuple[Path, Path, int]:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    source = folder / "capture.png"
    Image.new("RGB", (64, 48), "navy").save(source)
    database = tmp_path / "data" / "slothshot.sqlite3"
    connection = connect_database(database)
    try:
        assert index_image(connection, source, Reader()) == "indexed"
        row = connection.execute(
            "SELECT id FROM images WHERE path = ?", (str(source.resolve()),)
        ).fetchone()
        return folder, database, int(row["id"])
    finally:
        connection.close()


def test_suggest_filename_uses_site_and_page_context() -> None:
    suggestion = suggest_filename("GitHub\nSettings\nRepository preferences", Path("capture.png"))

    assert suggestion == "GitHub Settings.png"


def test_suggest_filename_uses_readable_ocr_words_when_no_site_is_known() -> None:
    suggestion = suggest_filename("Docker port problem address already in use", Path("capture.jpg"))

    assert suggestion == "Docker Port Problem Address Already Use.jpg"


def test_rename_indexed_image_moves_file_and_updates_database_and_fts(tmp_path: Path) -> None:
    folder, database, image_id = _indexed_image(tmp_path)
    source = folder / "capture.png"

    destination = rename_indexed_image(database, image_id, source, "Docker Error.png", folder)

    assert destination == folder / "Docker Error.png"
    assert destination.is_file()
    assert not source.exists()
    connection = connect_database(database)
    try:
        row = connection.execute(
            "SELECT path, name_state FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        assert row["path"] == str(destination.resolve())
        assert row["name_state"] == "done"
        assert (
            connection.execute(
                "SELECT rowid FROM images_fts WHERE images_fts MATCH 'docker'"
            ).fetchone()[0]
            == image_id
        )
    finally:
        connection.close()


def test_rename_indexed_image_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    folder, database, image_id = _indexed_image(tmp_path)
    source = folder / "capture.png"
    existing = folder / "Already Here.png"
    existing.write_bytes(b"keep this file")

    with pytest.raises(FileExistsError):
        rename_indexed_image(database, image_id, source, existing.name, folder)

    assert source.is_file()
    assert existing.read_bytes() == b"keep this file"
