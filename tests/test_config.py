"""Tests for configuration defaults and validation."""

from pathlib import Path

import pytest

from slothshot.config import default_config, load_config


def test_default_config_uses_screenshots_and_user_data() -> None:
    config = default_config()
    assert config.watch_folder == Path.home() / "Pictures" / "Screenshots"
    assert config.database_path == config.data_dir / "slothshot.sqlite3"
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
    assert config.auto_rename is False


def test_missing_file_uses_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path / "missing.toml") == default_config()


def test_loads_overrides_and_relative_paths(tmp_path: Path) -> None:
    file = tmp_path / "config.toml"
    file.write_text(
        '[paths]\nwatch_folder="screens"\ndata_dir="data"\n'
        '[models]\nembedding="custom/model"\n'
        "[ocr]\nmax_image_side=2048\npreprocess=false\n",
        encoding="utf-8",
    )

    config = load_config(file)
    assert config.watch_folder == tmp_path / "screens"
    assert config.data_dir == tmp_path / "data"
    assert config.model_cache == config.data_dir / "models"
    assert config.embedding_model == "custom/model"
    assert config.auto_rename is False
    assert config.vision_timeout_seconds == 600
    assert config.max_image_side == 2048
    assert config.preprocess is False


def test_model_cache_defaults_under_custom_data_dir(tmp_path: Path) -> None:
    file = tmp_path / "config.toml"
    file.write_text('[paths]\ndata_dir="my-data"\n', encoding="utf-8")

    config = load_config(file)
    assert config.model_cache == tmp_path / "my-data" / "models"


def test_rejects_invalid_ocr_size(tmp_path: Path) -> None:
    file = tmp_path / "config.toml"
    file.write_text("[ocr]\nmax_image_side=0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        load_config(file)


def test_loads_auto_rename_setting(tmp_path: Path) -> None:
    file = tmp_path / "config.toml"
    file.write_text("[models]\nauto_rename = true\n", encoding="utf-8")
    assert load_config(file).auto_rename is True


def test_loads_vision_timeout_setting(tmp_path: Path) -> None:
    file = tmp_path / "config.toml"
    file.write_text("[models]\nvision_timeout_seconds = 77\n", encoding="utf-8")
    assert load_config(file).vision_timeout_seconds == 77


def test_rejects_invalid_auto_rename_setting(tmp_path: Path) -> None:
    file = tmp_path / "config.toml"
    file.write_text('[models]\nauto_rename = "yes"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="models.auto_rename"):
        load_config(file)


@pytest.mark.parametrize("value", [0, -1, '"slow"'])
def test_rejects_invalid_vision_timeout(tmp_path: Path, value: str | int) -> None:
    file = tmp_path / "config.toml"
    file.write_text(f"[models]\nvision_timeout_seconds = {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="models.vision_timeout_seconds"):
        load_config(file)


def test_rejects_invalid_toml(tmp_path: Path) -> None:
    file = tmp_path / "config.toml"
    file.write_text("[paths\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid TOML"):
        load_config(file)
