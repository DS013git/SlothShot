"""Local Streamlit interface for browsing and searching indexed screenshots."""

from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, UnidentifiedImageError

from slothshot.config import AppConfig, load_config
from slothshot.embed import EmbeddingModel, embedding_model_cached
from slothshot.indexer import SUPPORTED_IMAGE_SUFFIXES
from slothshot.search import SearchResult, search_hybrid, search_keyword, search_semantic

logger = logging.getLogger("slothshot.ui")
TEMPORARY_SUFFIXES = {".crdownload", ".part", ".partial", ".temp", ".tmp"}


@dataclass(frozen=True)
class IndexSnapshot:
    """Small read-only view of the local index for the sidebar."""

    total: int = 0
    errors: int = 0
    latest_indexed: str | None = None
    indexed_paths: tuple[str, ...] = ()


def _readonly_connection(database_path: Path) -> sqlite3.Connection:
    """Open the existing index in read-only mode without creating its folder."""
    connection = sqlite3.connect(
        f"{database_path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


@st.cache_data(ttl=3, max_entries=4, show_spinner=False)
def load_index_snapshot(database_path: str) -> IndexSnapshot:
    """Read index totals and paths; return an empty snapshot before first use."""
    database = Path(database_path)
    if not database.is_file():
        return IndexSnapshot()

    connection = _readonly_connection(database)
    try:
        counts = connection.execute(
            """SELECT count(*) AS total,
                      sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
                      max(CASE WHEN status = 'indexed' THEN indexed_at END) AS latest_indexed
               FROM images"""
        ).fetchone()
        paths = tuple(str(row["path"]) for row in connection.execute("SELECT path FROM images"))
        return IndexSnapshot(
            total=int(counts["total"] or 0),
            errors=int(counts["errors"] or 0),
            latest_indexed=counts["latest_indexed"],
            indexed_paths=paths,
        )
    finally:
        connection.close()


@st.cache_data(ttl=10, max_entries=8, show_spinner=False)
def count_pending_files(watch_folder: str, indexed_paths: tuple[str, ...]) -> int:
    """Count image paths that are not indexed yet, with a short cache window."""
    root = Path(watch_folder)
    if not root.is_dir():
        return 0
    known_paths = {os.path.normcase(os.path.normpath(path)) for path in indexed_paths}
    pending = 0
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            if any(suffix.lower() in TEMPORARY_SUFFIXES for suffix in path.suffixes):
                continue
            if path.name.startswith("~$") or path.name.endswith("~"):
                continue
            normalized = os.path.normcase(os.path.normpath(str(path.resolve())))
            if normalized not in known_paths:
                pending += 1
    except OSError:
        logger.exception("Could not count screenshot files under %s", root)
    return pending


@st.cache_data(ttl=30, max_entries=8, show_spinner=False)
def _embedding_model_is_cached(model_name: str, cache_directory: str) -> bool:
    """Check the local model files without initializing an inference session."""
    return embedding_model_cached(model_name, Path(cache_directory))


@st.cache_data(max_entries=96, show_spinner=False)
def load_thumbnail(path: str, modified_ns: int, size: tuple[int, int] = (520, 360)) -> bytes | None:
    """Decode a bounded thumbnail and cache its JPEG bytes in memory."""
    del modified_ns  # This value is part of Streamlit's cache key for invalidation.
    try:
        with Image.open(path) as source:
            source.thumbnail(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
            thumbnail = source.convert("RGB")
            output = BytesIO()
            thumbnail.save(output, format="JPEG", quality=82, optimize=True)
            return output.getvalue()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        logger.info("Could not create a thumbnail for %s", path, exc_info=True)
        return None


def highlighted_snippet(snippet: str) -> str:
    """Escape OCR text, then turn search.py's bracket markers into safe highlights."""
    safe_text = html.escape(snippet)
    return re.sub(r"\[([^\]]+)\]", r"<mark>\1</mark>", safe_text)


def _config_from_environment() -> tuple[AppConfig, Path | None]:
    """Load the config path forwarded by ``slothshot ui`` when present."""
    configured_path = os.environ.get("SLOTHSHOT_CONFIG_PATH")
    path = Path(configured_path).expanduser().resolve() if configured_path else None
    return load_config(path), path


def _search(config: AppConfig, query: str, mode: str, *, limit: int = 30) -> list[SearchResult]:
    """Run one local search using a short-lived read-only database connection."""
    if not config.database_path.is_file():
        return []
    connection = _readonly_connection(config.database_path)
    try:
        if mode == "keyword":
            return search_keyword(connection, query, limit=limit)
        model = EmbeddingModel(config)
        if mode == "semantic":
            return search_semantic(connection, query, model, limit=limit)
        return search_hybrid(connection, query, model, limit=limit)
    finally:
        connection.close()


def _open_image(path: Path) -> None:
    """Ask the local operating system to open one screenshot."""
    if not path.is_file():
        raise FileNotFoundError(f"Screenshot no longer exists: {path}")
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.resolve().as_uri())


def _open_folder(path: Path) -> None:
    """Open the containing folder, selecting the screenshot where supported."""
    if os.name == "nt":
        subprocess.Popen(["explorer.exe", "/select,", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])


def _start_index_process(config: AppConfig, config_path: Path | None) -> subprocess.Popen[bytes]:
    """Run the existing index command as a child so Streamlit stays responsive."""
    command = [
        sys.executable,
        "-m",
        "slothshot.cli",
        "index",
        str(config.watch_folder),
    ]
    environment = os.environ.copy()
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    if config_path is not None:
        command.extend(["--config", str(config_path)])
        environment["SLOTHSHOT_CONFIG_PATH"] = str(config_path)
    else:
        environment.pop("SLOTHSHOT_CONFIG_PATH", None)

    log_path = config.data_dir / "ui-index.log"
    config.data_dir.mkdir(parents=True, exist_ok=True)
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with log_path.open("ab") as log_file:
        return subprocess.Popen(
            command,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )


def _format_timestamp(value: str | None) -> str:
    """Show stored ISO timestamps in a compact local-time format."""
    if not value:
        return "Not yet"
    try:
        timestamp = datetime.fromisoformat(value).astimezone()
    except ValueError:
        return value
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _render_sidebar(config: AppConfig, config_path: Path | None) -> IndexSnapshot:
    """Show index status and launch indexing outside the Streamlit process."""
    st.sidebar.header("Local index")
    try:
        snapshot = load_index_snapshot(str(config.database_path))
    except (OSError, sqlite3.Error) as exc:
        snapshot = IndexSnapshot()
        st.sidebar.error(f"Could not read the local database: {exc}")

    pending = count_pending_files(str(config.watch_folder), snapshot.indexed_paths)
    st.sidebar.metric("Screenshots", snapshot.total)
    st.sidebar.metric("Last indexed", _format_timestamp(snapshot.latest_indexed))
    st.sidebar.metric("Pending", pending)
    if snapshot.errors:
        st.sidebar.caption(f"{snapshot.errors} image(s) have OCR errors.")

    try:
        model_ready = _embedding_model_is_cached(config.embedding_model, str(config.model_cache))
    except (ImportError, OSError, ValueError) as exc:
        model_ready = False
        logger.warning("Could not inspect the embedding model cache: %s", exc)
    if model_ready:
        logger.info("Embedding model %s is cached locally", config.embedding_model)
        st.sidebar.caption("Semantic model: cached locally")
    else:
        logger.info("Embedding model %s is not cached locally", config.embedding_model)
        st.sidebar.caption("Semantic model: download needed for first semantic search")

    process = st.session_state.get("index_process")
    if process is not None:
        return_code = process.poll()
        if return_code is None:
            st.sidebar.info("Indexing is running in the background.")
        elif return_code == 0:
            st.sidebar.success("Indexing finished. Refresh to update the counts.")
        else:
            st.sidebar.error(
                f"Indexing exited with code {return_code}. See {config.data_dir / 'ui-index.log'}."
            )

    index_disabled = process is not None and process.poll() is None
    if st.sidebar.button("Index now", disabled=index_disabled, use_container_width=True):
        try:
            st.session_state["index_process"] = _start_index_process(config, config_path)
            st.sidebar.success("Indexing started in the background.")
        except OSError as exc:
            st.sidebar.error(f"Could not start indexing: {exc}")

    if st.sidebar.button("Refresh stats", use_container_width=True):
        load_index_snapshot.clear()
        count_pending_files.clear()
        st.rerun()
    st.sidebar.caption(f"Watching: {config.watch_folder}")
    return snapshot


def _render_result(result: SearchResult, slot: Any) -> None:
    """Render one thumbnail card with safe text and local file actions."""
    path = Path(result.path)
    with slot.container(border=True):
        try:
            modified_ns = path.stat().st_mtime_ns
            thumbnail = load_thumbnail(str(path), modified_ns)
        except OSError:
            thumbnail = None
        if thumbnail is None:
            st.info("Image unavailable")
        else:
            st.image(thumbnail, caption=path.name, width="stretch")

        st.markdown(f"**Score:** {result.score:.4f}")
        st.markdown(highlighted_snippet(result.snippet), unsafe_allow_html=True)
        st.caption("Path (use the copy button on the path below)")
        st.code(result.path, language=None)
        action_columns = st.columns(2)
        if action_columns[0].button("Open image", key=f"open-image-{result.image_id}"):
            try:
                _open_image(path)
            except OSError as exc:
                st.error(f"Could not open screenshot: {exc}")
        if action_columns[1].button("Open folder", key=f"open-folder-{result.image_id}"):
            try:
                _open_folder(path)
            except OSError as exc:
                st.error(f"Could not open screenshot folder: {exc}")


def run_app() -> None:
    """Build the Streamlit page from local SQLite data only."""
    st.set_page_config(page_title="SlothShot", page_icon="🦥", layout="wide")
    st.title("SlothShot")
    st.caption("Your local librarian for screenshots. Search words or describe what you remember.")

    try:
        config, config_path = _config_from_environment()
    except (OSError, ValueError) as exc:
        st.error(f"Could not load SlothShot config: {exc}")
        return

    snapshot = _render_sidebar(config, config_path)
    with st.form("screenshot-search", clear_on_submit=False):
        query_columns = st.columns([4, 1])
        query = query_columns[0].text_input(
            "Search screenshots", placeholder="Try: docker port problem"
        )
        mode = query_columns[1].selectbox("Search mode", ("hybrid", "keyword", "semantic"), index=0)
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        st.session_state["search_query"] = query.strip()
        st.session_state["search_mode"] = mode
        try:
            st.session_state["search_results"] = _search(config, query.strip(), mode)
            st.session_state["search_error"] = None
        except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            st.session_state["search_results"] = []
            st.session_state["search_error"] = str(exc)

    search_error = st.session_state.get("search_error")
    if search_error:
        st.error(f"Search failed: {search_error}")
    elif snapshot.total == 0 and not st.session_state.get("search_query"):
        st.info(
            "No screenshots are indexed yet. Use **Index now** in the sidebar, or run "
            "`slothshot index <your screenshots folder>` in PowerShell."
        )
    elif st.session_state.get("search_query"):
        results = st.session_state.get("search_results", [])
        st.subheader(
            f"Results for {st.session_state['search_query']!r} "
            f"({st.session_state.get('search_mode', 'hybrid')})"
        )
        if not results:
            st.info("No matching screenshots found. Try different words or switch search mode.")
        else:
            columns = st.columns(3, gap="medium")
            for index, result in enumerate(results):
                _render_result(result, columns[index % len(columns)])
    elif snapshot.total:
        st.info("Search your indexed screenshots above.")


if __name__ == "__main__":
    run_app()
