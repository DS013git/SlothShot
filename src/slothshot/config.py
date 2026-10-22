"""Load and validate SlothShot's TOML configuration."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class AppConfig:
    """Settings used by SlothShot, resolved to concrete local paths."""

    watch_folder: Path
    data_dir: Path
    model_cache: Path
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vision_model: str = "qwen3-vl:2b-instruct"
    max_image_side: int = 1600
    preprocess: bool = True
    invert_dark_mode: bool = False

    @property
    def database_path(self) -> Path:
        """Return the SQLite database path inside the application data folder."""
        return self.data_dir / "slothshot.sqlite3"


def _default_data_dir() -> Path:
    """Choose the conventional per-user application data directory."""
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "SlothShot"
    return Path.home() / ".local" / "share" / "SlothShot"


def default_config() -> AppConfig:
    """Build defaults without creating directories or touching user files."""
    profile = Path.home()
    watch_folder = profile / "Pictures" / "Screenshots"
    data_dir = _default_data_dir()
    return AppConfig(
        watch_folder=watch_folder,
        data_dir=data_dir,
        model_cache=data_dir / "models",
    )


def default_config_path() -> Path:
    """Return the standard per-user location of config.toml."""
    if os.name == "nt":
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "SlothShot" / "config.toml"
    return Path.home() / ".config" / "slothshot" / "config.toml"


def save_watch_folder(folder: Path, config_path: Path | None = None) -> Path:
    """Persist only the watch-folder setting, preserving the rest of the TOML file."""
    if not str(folder).strip():
        raise ValueError("Choose a screenshot folder first.")
    selected_folder = folder.expanduser().resolve()
    if not selected_folder.is_dir():
        raise NotADirectoryError(f"Screenshot folder does not exist: {selected_folder}")

    destination = (config_path or default_config_path()).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        original = destination.read_text(encoding="utf-8")
    except FileNotFoundError:
        original = ""

    newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines()
    encoded_folder = json.dumps(str(selected_folder), ensure_ascii=False)
    watch_folder_line = f"watch_folder = {encoded_folder}"

    paths_start = next((i for i, line in enumerate(lines) if line.strip() == "[paths]"), None)
    if paths_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(("[paths]", watch_folder_line))
    else:
        paths_end = next(
            (
                i
                for i in range(paths_start + 1, len(lines))
                if re.match(r"^\s*\[[^]]+\]\s*(?:#.*)?$", lines[i])
            ),
            len(lines),
        )
        watch_folder_index = next(
            (
                i
                for i in range(paths_start + 1, paths_end)
                if re.match(r"^\s*watch_folder\s*=", lines[i])
            ),
            None,
        )
        if watch_folder_index is None:
            lines.insert(paths_end, watch_folder_line)
        else:
            lines[watch_folder_index] = watch_folder_line

    updated = newline.join(lines)
    if original.endswith(("\n", "\r")) or not original:
        updated += newline

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(updated)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination


def _expand_path(value: str, *, base: Path) -> Path:
    """Expand environment variables and resolve relative paths beside config."""
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def load_config(path: Path | None = None) -> AppConfig:
    """Load TOML overrides, falling back to safe local defaults.

    Raises:
        ValueError: The TOML is malformed or contains unsupported values.
    """
    config_path = path or default_config_path()
    defaults = default_config()
    if not config_path.exists():
        return defaults

    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root in {config_path} must be a table")
    paths = raw.get("paths", {})
    models = raw.get("models", {})
    ocr = raw.get("ocr", {})
    if not all(isinstance(section, dict) for section in (paths, models, ocr)):
        raise ValueError("[paths], [models], and [ocr] must be TOML tables")

    base = config_path.parent
    try:
        max_image_side = int(ocr.get("max_image_side", defaults.max_image_side))
    except (TypeError, ValueError) as exc:
        raise ValueError("ocr.max_image_side must be a positive integer") from exc
    if max_image_side < 1:
        raise ValueError("ocr.max_image_side must be a positive integer")

    embedding_model = models.get("embedding", defaults.embedding_model)
    vision_model = models.get("vision", defaults.vision_model)
    preprocess = ocr.get("preprocess", defaults.preprocess)
    invert_dark_mode = ocr.get("invert_dark_mode", defaults.invert_dark_mode)
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("models.embedding must be a non-empty string")
    if not isinstance(vision_model, str) or not vision_model.strip():
        raise ValueError("models.vision must be a non-empty string")
    if not isinstance(preprocess, bool):
        raise ValueError("ocr.preprocess must be true or false")
    if not isinstance(invert_dark_mode, bool):
        raise ValueError("ocr.invert_dark_mode must be true or false")

    def path_value(key: str, fallback: Path) -> Path:
        value = paths.get(key)
        if value is None:
            return fallback
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a non-empty path string")
        return _expand_path(value, base=base)

    data_dir = path_value("data_dir", defaults.data_dir)
    return AppConfig(
        watch_folder=path_value("watch_folder", defaults.watch_folder),
        data_dir=data_dir,
        model_cache=path_value("model_cache", data_dir / "models"),
        embedding_model=embedding_model.strip(),
        vision_model=vision_model.strip(),
        max_image_side=max_image_side,
        preprocess=preprocess,
        invert_dark_mode=invert_dark_mode,
    )
