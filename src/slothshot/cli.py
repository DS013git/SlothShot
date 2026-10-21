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
from slothshot.config import load_config
from slothshot.db import connect_database
from slothshot.embed import EmbeddingModel, embedding_model_cached
from slothshot.indexer import scan_folder
from slothshot.ocr import OCR_MODEL_FILES, OCRReader
from slothshot.search import search_hybrid, search_keyword, search_semantic

app = typer.Typer(help="A local, offline librarian for your screenshots.")
logger = logging.getLogger("slothshot")


@app.callback()
def main() -> None:
    """Manage local screenshot indexing and search."""


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

    try:
        cache_present = embedding_model_cached(config.embedding_model, config.model_cache)
    except (ImportError, OSError, ValueError) as exc:
        cache_present = False
        logger.warning("Could not inspect the embedding model cache: %s", exc)
    if cache_present:
        logger.info(
            "Embedding model %s is cached locally under %s",
            config.embedding_model,
            config.model_cache,
        )
        typer.echo(f"Embedding model cache: ready for {config.embedding_model}")
    else:
        logger.info(
            "Embedding model %s is not cached under %s; first semantic use may download it.",
            config.embedding_model,
            config.model_cache,
        )
        typer.echo(f"Embedding model cache: not found for {config.embedding_model}")

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


@app.command()
def index(
    folder: Path = typer.Argument(..., help="Screenshot folder to scan recursively."),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
) -> None:
    """OCR and index new or changed images from a folder."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(config_file)
        connection = connect_database(config.database_path)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        typer.echo(f"Could not prepare local database: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        summary = scan_folder(folder, connection, OCRReader(config), EmbeddingModel(config))
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        typer.echo(f"Index scan failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        connection.close()

    typer.echo(
        "Index complete: "
        f"{summary.found} found, {summary.indexed} indexed, {summary.reused} reused, "
        f"{summary.skipped} unchanged, {summary.failed} failed, {summary.removed} removed"
    )
    if summary.failed:
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Words to find in indexed screenshots."),
    mode: str = typer.Option(
        "hybrid", "--mode", help="Search mode: keyword, semantic, or hybrid."
    ),
    limit: int = typer.Option(20, min=1, max=100, help="Maximum number of results."),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
) -> None:
    """Find screenshot text by exact words, meaning, or both."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if mode not in {"keyword", "semantic", "hybrid"}:
        typer.echo("Choose --mode keyword, semantic, or hybrid.", err=True)
        raise typer.Exit(code=2)
    try:
        config = load_config(config_file)
        connection = connect_database(config.database_path)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        typer.echo(f"Could not open the local database: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        if mode == "keyword":
            results = search_keyword(connection, query, limit=limit)
        else:
            embedder = EmbeddingModel(config)
            if mode == "semantic":
                results = search_semantic(connection, query, embedder, limit=limit)
            else:
                results = search_hybrid(connection, query, embedder, limit=limit)
    except (ValueError, RuntimeError, sqlite3.Error) as exc:
        typer.echo(f"Search failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        connection.close()

    if not results:
        typer.echo("No matching screenshots found.")
        return
    for result in results:
        typer.echo(f"{result.path}\n  score: {result.score:.4f}\n  {result.snippet}")


if __name__ == "__main__":
    app()
