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
from slothshot.naming import filename_from_description, rename_indexed_image, unique_filename
from slothshot.ocr import OCR_MODEL_FILES, OCRReader
from slothshot.search import search_hybrid, search_keyword, search_semantic
from slothshot.vision import VisionModelError, describe_screenshot
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
    rename_new: bool = typer.Option(
        False,
        "--rename-new",
        help="Use the local vision model to name newly indexed screenshots.",
    ),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
) -> None:
    """OCR and index new or changed images from a folder."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not folder.expanduser().is_dir():
        typer.echo(
            f"Screenshot folder does not exist or is not a directory: {folder}", err=True
        )
        raise typer.Exit(code=1)
    try:
        config = load_config(config_file)
        connection = connect_database(config.database_path)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        typer.echo(f"Could not prepare local database: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    rename_counts = {"renamed": 0, "skipped": 0, "failed": 0}

    def rename_new_image(image_id: int, path: Path, ocr_text: str) -> None:
        if rename_counts["failed"]:
            return
        try:
            description = describe_screenshot(path, ocr_text, config.vision_model)
            suggested_name = unique_filename(
                filename_from_description(description, path), path
            )
        except (OSError, ValueError, VisionModelError) as exc:
            rename_counts["failed"] += 1
            logger.error("Could not name new screenshot %s: %s", path, exc)
            return
        if suggested_name == path.name:
            rename_counts["skipped"] += 1
            return
        try:
            rename_indexed_image(
                config.database_path, image_id, path, suggested_name, folder
            )
        except (FileExistsError, FileNotFoundError, OSError, ValueError, sqlite3.Error):
            rename_counts["skipped"] += 1
            logger.exception("Could not automatically rename newly indexed screenshot %s", path)
        else:
            rename_counts["renamed"] += 1

    try:
        summary = scan_folder(
            folder,
            connection,
            OCRReader(config),
            EmbeddingModel(config),
            on_indexed=rename_new_image if rename_new else None,
        )
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
    if rename_new:
        typer.echo(
            f"Automatic renaming: {rename_counts['renamed']} renamed, "
            f"{rename_counts['skipped']} skipped, {rename_counts['failed']} model error(s)"
        )
    if summary.failed:
        raise typer.Exit(code=1)
    if rename_counts["failed"]:
        raise typer.Exit(code=2)


@app.command("rename-all")
def rename_all(
    folder: Path = typer.Argument(..., help="Configured screenshot folder to rename recursively."),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to config.toml; defaults to the user config path."
    ),
) -> None:
    """Use the local vision model to rename every indexed screenshot in a folder."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not folder.expanduser().is_dir():
        typer.echo(
            f"Screenshot folder does not exist or is not a directory: {folder}", err=True
        )
        raise typer.Exit(code=1)
    try:
        config = load_config(config_file)
        connection = connect_database(config.database_path)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        typer.echo(f"Could not prepare local database: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    root = folder.expanduser().resolve()
    try:
        rows = connection.execute(
            "SELECT id, path, ocr_text FROM images WHERE status = 'indexed' "
            "ORDER BY path COLLATE NOCASE"
        ).fetchall()
    finally:
        connection.close()

    renamed = skipped = failed = 0
    for row in rows:
        path = Path(str(row["path"])).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if not path.is_file():
            skipped += 1
            continue
        try:
            description = describe_screenshot(path, str(row["ocr_text"] or ""), config.vision_model)
            new_name = unique_filename(filename_from_description(description, path), path)
            if new_name == path.name:
                skipped += 1
                continue
            rename_indexed_image(
                config.database_path, int(row["id"]), path, new_name, root
            )
            renamed += 1
        except (FileExistsError, FileNotFoundError, OSError, ValueError, VisionModelError) as exc:
            failed += 1
            logger.error("Could not rename screenshot %s: %s", path, exc)
            if isinstance(exc, VisionModelError):
                break
        except sqlite3.Error:
            failed += 1
            logger.exception("Could not update the index while renaming %s", path)

    typer.echo(
        f"Vision rename complete: {renamed} renamed, {skipped} unchanged or unavailable, "
        f"{failed} error(s)"
    )
    if failed:
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
