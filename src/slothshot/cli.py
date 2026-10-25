"""Command-line entry points for SlothShot."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from logging.handlers import RotatingFileHandler
from pathlib import Path

import typer

from slothshot import __version__
from slothshot.config import load_config
from slothshot.db import connect_database
from slothshot.embed import EmbeddingModel, embedding_model_cached
from slothshot.indexer import scan_folder
from slothshot.ocr import OCR_MODEL_FILES, OCRReader
from slothshot.renaming import rename_images
from slothshot.search import search_hybrid, search_keyword, search_semantic
from slothshot.watcher import ScreenshotWatcher

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
    python_ok = sys.version_info >= (3, 11) and sys.version_info < (3, 12)
    python_status = "ok" if python_ok else "expected 3.11"
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
    rename_new: bool = typer.Option(
        False,
        "--rename-new",
        help="Use the configured naming method for newly indexed screenshots.",
    ),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
) -> None:
    """OCR and index new or changed images from a folder."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not folder.expanduser().is_dir():
        typer.echo(f"Screenshot folder does not exist or is not a directory: {folder}", err=True)
        raise typer.Exit(code=1)
    try:
        config = load_config(config_file)
        connection = connect_database(config.database_path)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        typer.echo(f"Could not prepare local database: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        summary = scan_folder(
            folder,
            connection,
            OCRReader(config),
            EmbeddingModel(config),
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        typer.echo(f"Index scan failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        connection.close()

    rename_counts = None
    if rename_new:
        try:
            rename_counts = rename_images(
                config.database_path,
                folder,
                naming=config.naming,
                text_model=config.text_model,
                vision_model=config.vision_model,
                vision_timeout_seconds=config.vision_timeout_seconds,
            )
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            typer.echo(f"Could not process pending screenshot names: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    typer.echo(
        "Index complete: "
        f"{summary.found} found, {summary.indexed} indexed, {summary.reused} reused, "
        f"{summary.skipped} unchanged, {summary.failed} failed, {summary.removed} removed"
    )
    if rename_new:
        assert rename_counts is not None
        typer.echo(
            f"Automatic renaming: {rename_counts.renamed} renamed, "
            f"{rename_counts.locked} locked, {rename_counts.missing} missing, "
            f"{rename_counts.model_errors} model error(s), "
            f"{rename_counts.unchanged + rename_counts.skipped} unchanged or busy, "
            f"{rename_counts.pending} pending, {rename_counts.failed} error(s)"
        )
        if rename_counts.last_error and not rename_counts.timed_out:
            typer.echo(f"Last rename error: {rename_counts.last_error}")
    typer.echo(
        f"{summary.indexed} indexed, "
        f"{rename_counts.renamed if rename_counts else 0} renamed, "
        f"{rename_counts.locked if rename_counts else 0} locked, "
        f"{rename_counts.model_errors if rename_counts else 0} model errors"
    )
    if summary.failed:
        raise typer.Exit(code=1)
    if rename_new and rename_counts is not None and (rename_counts.failed or rename_counts.pending):
        raise typer.Exit(code=2)


@app.command("rename-all")
def rename_all(
    folder: Path = typer.Argument(..., help="Configured screenshot folder to rename recursively."),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
) -> None:
    """Use the configured naming method to rename every indexed screenshot in a folder."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not folder.expanduser().is_dir():
        typer.echo(f"Screenshot folder does not exist or is not a directory: {folder}", err=True)
        raise typer.Exit(code=1)
    try:
        config = load_config(config_file)
        summary = rename_images(
            config.database_path,
            folder,
            naming=config.naming,
            text_model=config.text_model,
            vision_model=config.vision_model,
            vision_timeout_seconds=config.vision_timeout_seconds,
            pending_only=False,
            indexed_only=True,
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        typer.echo(f"Could not prepare local database: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Screenshot rename complete: {summary.renamed} renamed, "
        f"{summary.locked} locked, {summary.missing} missing, "
        f"{summary.model_errors} model error(s), "
        f"{summary.unchanged + summary.skipped} unchanged or busy"
    )
    if summary.timed_out and summary.last_error:
        typer.echo(summary.last_error, err=True)
    if summary.failed:
        raise typer.Exit(code=2)


@app.command()
def search(
    query: str = typer.Argument(..., help="Words to find in indexed screenshots."),
    mode: str = typer.Option("hybrid", "--mode", help="Search mode: keyword, semantic, or hybrid."),
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


def _configure_file_logging(log_file: Path) -> None:
    """Send all logs to a size-limited file, for runs that have no console.

    ``pythonw.exe`` (used by the autostart task) starts without stdout or
    stderr, so anything written there would be lost or could raise. The file
    rotates at 1 MiB and keeps three old copies so it never grows without bound.
    """
    log_file = log_file.expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=3, encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[handler],
        force=True,
    )
    # Third-party libraries (RapidOCR) attach their own console handlers and
    # print to stderr; give them a harmless sink when there is no console.
    if sys.stdout is None:
        sys.stdout = Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115


@app.command()
def watch(
    folder: Path | None = typer.Argument(
        None, help="Screenshot folder; defaults to the configured watch folder."
    ),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
    rename_new: bool = typer.Option(
        False, "--rename-new", help="Use the configured naming method to name new screenshots."
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        help="Write logs to this file instead of the console (used by autostart).",
    ),
) -> None:
    """Catch up the index, then watch for screenshot changes until Ctrl+C."""
    if log_file is not None:
        _configure_file_logging(log_file)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        config = load_config(config_file)
        if folder is not None:
            config = replace(config, watch_folder=folder.expanduser().resolve())
        if rename_new:
            config = replace(config, auto_rename=True)
        ScreenshotWatcher(config).run()
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        logger.error("Could not start screenshot watcher: %s", exc)
        typer.echo(f"Could not start screenshot watcher: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def ui(
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
) -> None:
    """Start the local Streamlit interface on this computer only."""
    try:
        load_config(config_file)
    except (OSError, ValueError) as exc:
        typer.echo(f"Could not load SlothShot config: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    script_path = Path(__file__).with_name("ui.py").resolve()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.address=127.0.0.1",
    ]
    environment = os.environ.copy()
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    if config_file is not None:
        environment["SLOTHSHOT_CONFIG_PATH"] = str(config_file.expanduser().resolve())
    else:
        environment.pop("SLOTHSHOT_CONFIG_PATH", None)

    try:
        return_code = subprocess.call(command, env=environment)
    except OSError as exc:
        typer.echo(f"Could not start the Streamlit interface: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if return_code:
        raise typer.Exit(code=return_code)


if __name__ == "__main__":
    app()
