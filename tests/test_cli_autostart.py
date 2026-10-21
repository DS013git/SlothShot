"""Tests for the pieces of the CLI that unattended autostart depends on."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slothshot.cli import _configure_file_logging, app


@pytest.fixture
def restore_root_logging() -> Iterator[None]:
    """Undo logging.basicConfig(force=True) so other tests keep their handlers."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def test_file_logging_creates_parent_folder_and_writes(
    tmp_path: Path, restore_root_logging: None
) -> None:
    log_file = tmp_path / "nested" / "logs" / "watch.log"

    _configure_file_logging(log_file)
    logging.getLogger("slothshot.test").info("hello from autostart")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "hello from autostart" in log_file.read_text(encoding="utf-8")


def test_watch_failure_is_recorded_in_the_log_file(
    tmp_path: Path, restore_root_logging: None
) -> None:
    """A missing screenshots folder must leave a trace when no console exists."""
    config = tmp_path / "config.toml"
    config.write_text(
        f'[paths]\nwatch_folder = "{(tmp_path / "missing").as_posix()}"\n'
        f'data_dir = "{(tmp_path / "data").as_posix()}"\n',
        encoding="utf-8",
    )
    log_file = tmp_path / "watch.log"

    result = CliRunner().invoke(
        app, ["watch", "--config", str(config), "--log-file", str(log_file)]
    )
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert result.exit_code == 1
    assert "Screenshot folder does not exist" in log_file.read_text(encoding="utf-8")
