"""Command-line entry points for SlothShot."""

from __future__ import annotations

import importlib.metadata
import logging
import sqlite3
import sys
from pathlib import Path

import typer

from slothshot import __version__
from slothshot.config import AppConfig, load_config

app = typer.Typer(help="A local, offline librarian for your screenshots.")
logger = logging.getLogger("slothshot")


@app.callback()
def main() -> None:
    """Manage local screenshot indexing and search."""


def _model_cache_present(config: AppConfig) -> bool:
    """Return whether the configured model cache contains any downloaded files."""
    return config.model_cache.is_dir() and any(config.model_cache.rglob("*"))


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
        logger.info("Model cache found locally at %s; no download is needed by doctor.", config.model_cache)
        typer.echo(f"Model cache: files found ({config.model_cache})")
    else:
        logger.info("No cached models found at %s. Model download will be needed on first use.", config.model_cache)
        typer.echo(f"Model cache: empty ({config.model_cache})")

    typer.echo(f"Database: {config.database_path}")
    typer.echo(f"SlothShot: {__version__}")
    if not okay:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
