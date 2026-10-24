"""Small, local-only checks for the Streamlit interface helpers."""

from __future__ import annotations

import sqlite3
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

import slothshot.cli as cli
from slothshot.config import load_config
from slothshot.db import connect_database
from slothshot.indexer import index_image
from slothshot.ui import (
    _index_completion_message,
    _readonly_connection,
    _start_index_process,
    count_pending_files,
    highlighted_snippet,
    load_index_snapshot,
    load_thumbnail,
)


def test_snippet_highlighting_escapes_untrusted_ocr_text() -> None:
    marked = highlighted_snippet("<script>alert(1)</script> [docker] port")

    assert "&lt;script&gt;" in marked
    assert "<script>" not in marked
    assert "<mark>docker</mark>" in marked


def test_thumbnail_is_resized_and_cached_by_file_timestamp(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    Image.new("RGB", (1200, 800), "navy").save(image_path)
    modified_ns = image_path.stat().st_mtime_ns

    thumbnail = load_thumbnail(str(image_path), modified_ns, size=(300, 200))

    assert thumbnail is not None
    with Image.open(BytesIO(thumbnail)) as image:
        assert image.width <= 300
        assert image.height <= 200
        assert image.format == "JPEG"


def test_read_only_connection_does_not_create_or_change_database(tmp_path: Path) -> None:
    database_path = tmp_path / "app-data" / "slothshot.sqlite3"
    assert load_index_snapshot(str(database_path)).total == 0
    assert not database_path.exists()

    writable = connect_database(database_path)
    readonly = _readonly_connection(database_path)
    try:
        assert readonly.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            readonly.execute("CREATE TABLE should_not_exist (id INTEGER)")
    finally:
        readonly.close()
        writable.close()


def test_pending_count_ignores_indexed_and_temporary_paths(tmp_path: Path) -> None:
    count_pending_files.clear()
    indexed = tmp_path / "indexed.png"
    pending = tmp_path / "nested" / "pending.jpg"
    partial = tmp_path / "unfinished.partial.png"
    temporary = tmp_path / "copy.png.tmp"
    for path in (indexed, pending, partial, temporary):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 8), "white").save(path, format="PNG")

    count = count_pending_files(str(tmp_path), (str(indexed.resolve()),))

    assert count == 1


def test_streamlit_page_renders_empty_state_without_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[paths]\n"
        f"watch_folder = '{(tmp_path / 'screenshots').as_posix()}'\n"
        f"data_dir = '{(tmp_path / 'app-data').as_posix()}'\n"
        f"model_cache = '{(tmp_path / 'models').as_posix()}'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SLOTHSHOT_CONFIG_PATH", str(config_path))

    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "src" / "slothshot" / "ui.py"),
        default_timeout=20,
    ).run()

    assert not app.exception
    assert any("No screenshots are indexed yet" in element.value for element in app.info)
    assert not (tmp_path / "app-data" / "slothshot.sqlite3").exists()


def test_streamlit_page_runs_keyword_search_and_renders_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch_folder = tmp_path / "screenshots"
    watch_folder.mkdir()
    image_path = watch_folder / "terminal.png"
    Image.new("RGB", (240, 120), "black").save(image_path)
    database_path = tmp_path / "app-data" / "slothshot.sqlite3"
    connection = connect_database(database_path)

    class Reader:
        def read(self, path: Path) -> SimpleNamespace:
            return SimpleNamespace(
                text="docker address already in use",
                average_confidence=0.9,
                error=None,
            )

    try:
        index_image(connection, image_path, Reader())
    finally:
        connection.close()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[paths]\n"
        f"watch_folder = '{watch_folder.as_posix()}'\n"
        f"data_dir = '{(tmp_path / 'app-data').as_posix()}'\n"
        f"model_cache = '{(tmp_path / 'models').as_posix()}'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SLOTHSHOT_CONFIG_PATH", str(config_path))
    load_index_snapshot.clear()
    load_thumbnail.clear()

    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "src" / "slothshot" / "ui.py"),
        default_timeout=20,
    ).run()
    app.text_input[0].set_value("docker")
    app.button[0].click().run()

    assert not app.exception
    assert any("Results for 'docker'" in element.value for element in app.subheader)
    assert any("<mark>docker</mark>" in element.value for element in app.markdown)


def test_ui_cli_binds_to_loopback_and_disables_usage_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_call(command: list[str], *, env: dict[str, str]) -> int:
        captured["command"] = command
        captured["env"] = env
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    result = CliRunner().invoke(cli.app, ["ui"])

    assert result.exit_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert "--server.address=127.0.0.1" in command
    assert captured["env"]["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"


def test_index_now_child_process_uses_local_config_and_completes(tmp_path: Path) -> None:
    watch_folder = tmp_path / "screenshots"
    watch_folder.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[paths]\n"
        f'watch_folder = "{watch_folder.as_posix()}"\n'
        f'data_dir = "{(tmp_path / "app-data").as_posix()}"\n'
        f'model_cache = "{(tmp_path / "models").as_posix()}"\n',
        encoding="utf-8",
    )
    config = load_config(config_path)
    process = _start_index_process(config, config_path)
    try:
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    log = (config.data_dir / "ui-index.log").read_text(encoding="utf-8")
    assert "Index complete: 0 found" in log
    assert "0 indexed, 0 renamed, 0 locked, 0 model errors" in log


def test_ui_index_message_shows_counts_pending_renames_or_errors(tmp_path: Path) -> None:
    log_path = tmp_path / "ui-index.log"
    log_path.write_text(
        "one\ntwo\nthree\nfour\n3 indexed, 3 renamed, 1 locked, 0 model errors\n",
        encoding="utf-8",
    )

    assert _index_completion_message(log_path, 0) == (
        "3 indexed, 3 renamed, 1 locked, 0 model errors"
    )
    log_path.write_text(
        "Index complete: 1 found, 1 indexed, 0 reused, 0 skipped, 0 failed, 0 removed\n"
        "Automatic renaming: 0 renamed, 1 locked, 0 model error(s), 0 unchanged or busy, "
        "1 pending, 1 error(s)\n"
        "Last rename error: Screenshot is locked: capture.png (access denied)\n",
        encoding="utf-8",
    )
    warning = _index_completion_message(log_path, 2)
    assert warning.startswith("Indexing completed; 1 screenshot pending rename.")
    assert "Automatic renaming: 0 renamed, 1 locked" in warning
    assert "Last rename error:" in warning

    error = _index_completion_message(log_path, 1)
    assert "Indexing failed with code 1" in error
