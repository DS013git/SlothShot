"""Command-line entry points for SlothShot."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import typer

from slothshot import __version__
from slothshot.config import AppConfig, load_config
from slothshot.ocr import OCR_MODEL_FILES, OCRReader

app = typer.Typer(help="A local, offline librarian for your screenshots.")
logger = logging.getLogger("slothshot")


@app.callback()
def main() -> None:
    """Manage local screenshot indexing and search."""


def _model_cache_present(config: AppConfig) -> bool:
    """Return whether the configured model cache contains any downloaded files."""
    return config.model_cache.is_dir() and any(config.model_cache.rglob("*"))


def _bundled_ocr_status() -> tuple[bool, str]:
    """Check bundled OCR model files without initializing the OCR engine."""
    spec = importlib.util.find_spec("rapidocr")
    if spec is None or spec.origin is None:
        return False, "RapidOCR package is missing"

    model_dir = Path(spec.origin).parent / "models"
    missing = [name for name in OCR_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        return False, f"missing {', '.join(missing)} in {model_dir}"
    return True, f"bundled files found in {model_dir}"


def _dependency_status(distribution: str) -> str:
    """Describe an installed distribution without importing heavyweight packages."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


@app.command()
def doctor(
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
) -> None:
    """Check Python, dependencies, SQLite FTS5, local models, and watch folder."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(config_file)
    except (OSError, ValueError) as exc:
        typer.echo(f"Config: error ({exc})", err=True)
        raise typer.Exit(code=1) from exc

    okay = True
    python_ok = sys.version_info >= (3, 10) and sys.version_info < (3, 12)
    python_status = "ok" if python_ok else "expected 3.10 or 3.11"
    typer.echo(f"Python {sys.version_info.major}.{sys.version_info.minor}: {python_status}")
    okay &= python_ok

    distributions = {
        "fastembed": "fastembed",
        "NumPy": "numpy",
        "onnxruntime": "onnxruntime",
        "Pillow": "Pillow",
        "psutil": "psutil",
        "RapidOCR": "rapidocr",
        "Streamlit": "streamlit",
        "Typer": "typer",
        "watchdog": "watchdog",
    }
    for label, distribution in distributions.items():
        version = _dependency_status(distribution)
        present = version != "missing"
        typer.echo(f"{label}: {'installed ' + version if present else 'missing'}")
        okay &= present

    ocr_models_ready, ocr_models_message = _bundled_ocr_status()
    model_status = ocr_models_message if ocr_models_ready else f"unavailable ({ocr_models_message})"
    typer.echo(f"RapidOCR models: {model_status}")
    okay &= ocr_models_ready

    try:
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(text)")
            connection.execute("DROP TABLE fts_probe")
        typer.echo("SQLite FTS5: available")
    except sqlite3.Error as exc:
        typer.echo(f"SQLite FTS5: unavailable ({exc})")
        okay = False

    watch_exists = config.watch_folder.is_dir()
    typer.echo(f"Watch folder: {'exists' if watch_exists else 'missing'} ({config.watch_folder})")
    okay &= watch_exists

    cache_present = _model_cache_present(config)
    if cache_present:
        logger.info("Embedding model cache found locally at %s.", config.model_cache)
        typer.echo(f"Embedding model cache: files found ({config.model_cache})")
    else:
        logger.info(
            "Embedding model cache is empty at %s; it will be downloaded on first semantic use.",
            config.model_cache,
        )
        typer.echo(f"Embedding model cache: empty ({config.model_cache})")

    typer.echo(f"Database: {config.database_path}")
    typer.echo(f"SlothShot: {__version__}")
    if not okay:
        raise typer.Exit(code=1)


@app.command()
def ocr(
    image: Path = typer.Argument(..., exists=False, help="Screenshot to read."),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
    no_preprocess: bool = typer.Option(
        False, "--no-preprocess", help="Skip grayscale and upscaling."
    ),
) -> None:
    """Print detected text and per-line confidence for one image."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(config_file)
    except (OSError, ValueError) as exc:
        typer.echo(f"Config: error ({exc})", err=True)
        raise typer.Exit(code=1) from exc
    if no_preprocess:
        config = replace(config, preprocess=False)

    result = OCRReader(config).read(image)
    if result.error:
        typer.echo(f"{result.error} ({result.elapsed_seconds:.3f}s)", err=True)
        raise typer.Exit(code=1)
    if not result.lines:
        typer.echo(f"No text detected ({result.elapsed_seconds:.3f}s)")
        return

    for line in result.lines:
        typer.echo(f"[{line.confidence:.3f}] {line.text}")
    typer.echo(f"Elapsed: {result.elapsed_seconds:.3f}s")


if __name__ == "__main__":
    app()
