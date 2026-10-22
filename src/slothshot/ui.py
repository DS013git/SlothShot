"""Local Streamlit interface for browsing and searching indexed screenshots."""

from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
import subprocess
import sys
import unicodedata
import webbrowser
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, UnidentifiedImageError

from slothshot.config import AppConfig, default_config_path, load_config, save_watch_folder
from slothshot.db import connect_database
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


@dataclass(frozen=True)
class BulkRenameCandidate:
    """One indexed screenshot and its safe OCR-based destination name."""

    image_id: int
    source: Path
    destination_name: str


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


def _browse_watch_folder() -> None:
    """Open the local native folder picker and fill the watch-folder input."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        current_value = st.session_state.get("watch_folder_input", "")
        current_folder = Path(current_value).expanduser() if current_value else Path.home()
        initial_folder = current_folder if current_folder.is_dir() else Path.home()

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(
                parent=root,
                initialdir=str(initial_folder),
                title="Choose the screenshot folder to watch",
            )
        finally:
            root.destroy()
    except Exception as exc:
        logger.exception("Could not open the screenshot folder picker")
        st.session_state["watch_folder_feedback"] = (
            "error",
            f"Folder picker could not open: {exc}",
        )
        return

    if selected:
        st.session_state["watch_folder_input"] = selected
        st.session_state.pop("watch_folder_feedback", None)


def _save_watch_folder(config_path: Path | None) -> None:
    """Save the selected screenshot folder to the active local config."""
    selected_value = st.session_state.get("watch_folder_input", "")
    try:
        if not str(selected_value).strip():
            raise ValueError("Choose a screenshot folder first.")
        selected = Path(selected_value).expanduser()
        saved_path = save_watch_folder(selected, config_path)
    except (OSError, ValueError) as exc:
        st.session_state["watch_folder_feedback"] = ("error", str(exc))
        return

    st.session_state["watch_folder_input"] = str(selected.resolve())
    st.session_state["watch_folder_feedback"] = (
        "success",
        f"Saved to {saved_path}. Restart the background watcher to monitor this folder.",
    )
    load_index_snapshot.clear()
    count_pending_files.clear()


def _render_watch_folder(config: AppConfig, config_path: Path | None) -> None:
    """Render controls for choosing and saving the local screenshot folder."""
    st.sidebar.subheader("Watch folder")
    st.session_state.setdefault("watch_folder_input", str(config.watch_folder))
    st.sidebar.text_input(
        "Screenshot folder",
        key="watch_folder_input",
        help="Choose the local folder SlothShot should scan and monitor.",
    )
    picker_column, save_column = st.sidebar.columns(2)
    picker_column.button(
        "Choose folder",
        key="browse-watch-folder",
        on_click=_browse_watch_folder,
        use_container_width=True,
    )
    save_column.button(
        "Save folder",
        key="save-watch-folder",
        on_click=_save_watch_folder,
        args=(config_path or default_config_path(),),
        type="primary",
        use_container_width=True,
    )

    feedback = st.session_state.get("watch_folder_feedback")
    if feedback:
        message_type, message = feedback
        if message_type == "success":
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
    if config.watch_folder.is_dir():
        st.sidebar.caption(f"Configured folder: {config.watch_folder}")
    else:
        st.sidebar.warning(f"The configured folder does not exist: {config.watch_folder}")


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


_FILENAME_SKIP_WORDS = {
    "and", "are", "for", "from", "http", "https", "into", "the", "this", "that",
    "with", "www", "com", "org", "net", "home", "file", "edit", "view", "help",
    "chrome", "firefox", "safari", "google", "website", "browser", "bookmarks", "new", "tab",
}
_DATE_OR_TIME_LINE = re.compile(
    r"(?i)(?:\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b)"
)


def suggest_filename(ocr_text: str, current_path: Path) -> str:
    """Summarize recognizable website/app context into a concise filename."""
    text = unicodedata.normalize("NFKC", ocr_text)
    folded = text.casefold()

    browser = next(
        (
            name
            for marker, name in (
                ("chrome", "Chrome"),
                ("firefox", "Firefox"),
                ("microsoft edge", "Edge"),
                ("edge://", "Edge"),
                ("safari", "Safari"),
            )
            if marker in folded
        ),
        None,
    )

    domain_match = re.search(
        r"(?i)(?:https?://)?(?:www\.)?([a-z0-9-]+\.)+(com|net|org|io|app|dev|ai|co|uk|me|xyz)(?:/[^\s]*)?",
        text,
    )
    known_sites = (
        "Google Docs", "Google Drive", "Stack Overflow", "GitHub", "Render", "Figma", "Notion",
        "YouTube", "Vercel", "Netlify", "Slack", "Discord", "Reddit", "ChatGPT", "Claude",
        "Trello", "Linear", "Jira", "Canva", "Wikipedia",
    )
    site = next((name for name in known_sites if name.casefold() in folded), None)
    if not site and domain_match:
        host = domain_match.group(0).split("/")[0].casefold()
        host = re.sub(r"^https?://", "", host).removeprefix("www.")
        domain_site = host.split(".")[0]
        if domain_site not in {"google", "bing", "yahoo", "duckduckgo", "microsoft"}:
            site = domain_site

    page_context = next(
        (
            label
            for term, label in (
                ("dashboard", "Dashboard"),
                ("deploy", "Deployments"),
                ("project", "Projects"),
                ("settings", "Settings"),
                ("sign in", "Sign In"),
                ("log in", "Sign In"),
                ("documentation", "Documentation"),
                ("docs", "Documentation"),
                ("search results", "Search Results"),
                ("error", "Error Message"),
            )
            if term in folded
        ),
        None,
    )

    title = ""
    if site:
        summary = site[:1].upper() + site[1:]
        if page_context:
            summary += f" {page_context}"
        elif "website" in folded or domain_match:
            summary += " Website"
        if browser:
            summary += f" in {browser}"
        title = f"{summary}{current_path.suffix.lower()}"
    elif browser:
        summary = page_context or ("Website" if domain_match else None)
        if summary:
            title = f"{summary} in {browser}{current_path.suffix.lower()}"
    selected: list[str] = []
    seen: set[str] = set()
    for raw_line in ocr_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" \t|_—-:;,.•")
        if not line or _DATE_OR_TIME_LINE.search(line):
            continue
        words = re.findall(r"[^\W_]+", line, flags=re.UNICODE)
        meaningful = [
            word for word in words
            if len(word) >= 3 and not word.isdigit() and word.casefold() not in _FILENAME_SKIP_WORDS
        ]
        if not meaningful or len(meaningful) > 14:
            continue
        for word in meaningful:
            key = word.casefold()
            if key in seen:
                continue
            selected.append(word)
            seen.add(key)
            if len(selected) == 7:
                break
        if len(selected) >= 5:
            break

    if not title and selected:
        title = " ".join(word[:1].upper() + word[1:] for word in selected)
        title += current_path.suffix.lower()
    if not title:
        title = current_path.name

    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", Path(title).stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")[:90].rstrip(" .")
    if not stem or stem.upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        stem = "Screenshot"
    return f"{stem}{current_path.suffix.lower()}"


def _ocr_text_for_image(database_path: Path, image_id: int) -> str:
    """Read the indexed OCR text for one image without opening a write handle."""
    if not database_path.is_file():
        return ""
    connection = _readonly_connection(database_path)
    try:
        row = connection.execute("SELECT ocr_text FROM images WHERE id = ?", (image_id,)).fetchone()
        return str(row["ocr_text"] or "") if row else ""
    finally:
        connection.close()


def bulk_rename_candidates(
    database_path: Path, watch_folder: Path
) -> tuple[list[BulkRenameCandidate], int]:
    """Build a safe rename preview for indexed files inside the watch folder."""
    if not database_path.is_file():
        return [], 0

    root = watch_folder.expanduser().resolve()
    connection = _readonly_connection(database_path)
    try:
        rows = connection.execute(
            "SELECT id, path, ocr_text FROM images "
            "WHERE status = 'indexed' ORDER BY path COLLATE NOCASE"
        ).fetchall()
    finally:
        connection.close()

    candidates: list[BulkRenameCandidate] = []
    destinations: set[str] = set()
    skipped = 0
    for row in rows:
        source = Path(str(row["path"])).expanduser().resolve()
        try:
            source.relative_to(root)
        except ValueError:
            skipped += 1
            continue
        if not source.is_file():
            skipped += 1
            continue

        destination_name = suggest_filename(str(row["ocr_text"] or ""), source)
        destination = source.with_name(destination_name)
        if destination == source:
            continue

        destination_key = os.path.normcase(str(destination))
        if destination.exists() or destination_key in destinations:
            skipped += 1
            continue

        destinations.add(destination_key)
        candidates.append(
            BulkRenameCandidate(int(row["id"]), source, destination_name)
        )
    return candidates, skipped


def rename_indexed_image(
    database_path: Path, image_id: int, old_path: Path, new_name: str, watch_folder: Path
) -> Path:
    """Rename one indexed image in place and keep its SQLite/FTS path synchronized."""
    root = watch_folder.expanduser().resolve()
    source = old_path.expanduser().resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        message = "Only screenshots inside the configured watch folder can be renamed."
        raise ValueError(message) from exc
    if not source.is_file():
        raise FileNotFoundError(f"Screenshot no longer exists: {source}")

    candidate = Path(new_name.strip())
    if (
        candidate.name != new_name.strip()
        or candidate.suffix.casefold() != source.suffix.casefold()
    ):
        raise ValueError("Enter a filename only and keep the original file extension.")
    stem = candidate.stem.strip(" .")
    if not stem or stem in {".", ".."}:
        raise ValueError("Enter a valid filename.")
    if any(char in stem for char in '<>:"/\\|?*') or any(ord(char) < 32 for char in stem):
        raise ValueError("The filename contains characters that Windows does not allow.")
    reserved_names = {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if stem.upper() in reserved_names:
        raise ValueError("That filename is reserved by Windows.")
    destination = source.with_name(f"{stem}{source.suffix}")
    if destination == source:
        raise ValueError("Choose a different filename.")
    if destination.exists():
        raise FileExistsError(f"A file named {destination.name} already exists.")
    if not database_path.is_file():
        raise FileNotFoundError("The SlothShot index database is unavailable.")

    connection = connect_database(database_path)
    moved = False
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT path FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None or Path(str(row["path"])).resolve() != source:
            raise ValueError("The indexed path changed. Refresh search results and try again.")
        source.rename(destination)
        moved = True
        cursor = connection.execute(
            "UPDATE images SET path = ? WHERE id = ?", (str(destination), image_id)
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Could not update the indexed screenshot path.")
        connection.commit()
    except Exception:
        connection.rollback()
        if moved and destination.exists() and not source.exists():
            destination.rename(source)
        raise
    finally:
        connection.close()
    return destination


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


def _apply_bulk_rename(config: AppConfig) -> None:
    """Apply the confirmed preview one file at a time without overwriting files."""
    candidates = st.session_state.get("bulk_rename_candidates", [])
    renamed_paths: dict[str, str] = {}
    renamed = 0
    failed = 0
    for candidate in candidates:
        try:
            destination = rename_indexed_image(
                config.database_path,
                candidate.image_id,
                candidate.source,
                candidate.destination_name,
                config.watch_folder,
            )
        except (FileExistsError, FileNotFoundError, OSError, ValueError, sqlite3.Error):
            failed += 1
            logger.exception("Could not bulk-rename screenshot %s", candidate.source)
            continue
        renamed_paths[str(candidate.source)] = str(destination)
        renamed += 1

    if renamed_paths:
        st.session_state["search_results"] = [
            replace(item, path=renamed_paths.get(item.path, item.path))
            for item in st.session_state.get("search_results", [])
        ]
    st.session_state["bulk_rename_feedback"] = (
        f"Renamed {renamed} screenshot(s); skipped {failed} that changed or could not be renamed."
    )
    st.session_state.pop("bulk_rename_candidates", None)
    st.session_state.pop("bulk_rename_skipped", None)
    st.session_state.pop("confirm_bulk_rename", None)
    load_index_snapshot.clear()
    count_pending_files.clear()


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
    _render_watch_folder(config, config_path)
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

    feedback = st.session_state.pop("bulk_rename_feedback", None)
    if feedback:
        st.sidebar.success(feedback)

    if st.sidebar.button(
        "Rename all screenshots",
        disabled=index_disabled or snapshot.total == 0,
        use_container_width=True,
        help="Review OCR-based filename suggestions, then confirm before changing files.",
    ):
        try:
            candidates, skipped = bulk_rename_candidates(
                config.database_path, config.watch_folder
            )
            st.session_state["bulk_rename_candidates"] = candidates
            st.session_state["bulk_rename_skipped"] = skipped
            st.session_state["confirm_bulk_rename"] = True
        except (OSError, sqlite3.Error) as exc:
            st.sidebar.error(f"Could not prepare rename preview: {exc}")

    if st.session_state.get("confirm_bulk_rename"):
        candidates = st.session_state.get("bulk_rename_candidates", [])
        skipped = st.session_state.get("bulk_rename_skipped", 0)
        if candidates:
            st.sidebar.warning(
                f"Rename {len(candidates)} screenshot(s) in the configured folder using "
                "saved OCR text? Files already using the suggested name and name conflicts "
                "will be skipped."
            )
        else:
            st.sidebar.info("No screenshots need renaming based on their saved OCR text.")
        for candidate in candidates[:5]:
            st.sidebar.caption(f"{candidate.source.name} → {candidate.destination_name}")
        if len(candidates) > 5:
            st.sidebar.caption(f"…and {len(candidates) - 5} more")
        if skipped:
            st.sidebar.caption(f"{skipped} screenshot(s) will be skipped.")
        confirm_columns = st.sidebar.columns(2)
        if confirm_columns[0].button(
            f"Rename {len(candidates)} files",
            disabled=index_disabled or not candidates,
            key="confirm-bulk-rename",
            use_container_width=True,
        ):
            _apply_bulk_rename(config)
            st.rerun()
        if confirm_columns[1].button(
            "Cancel", key="cancel-bulk-rename", use_container_width=True
        ):
            st.session_state.pop("bulk_rename_candidates", None)
            st.session_state.pop("bulk_rename_skipped", None)
            st.session_state.pop("confirm_bulk_rename", None)
            st.rerun()

    if st.sidebar.button("Refresh stats", use_container_width=True):
        load_index_snapshot.clear()
        count_pending_files.clear()
        st.rerun()
    st.sidebar.caption(f"Watching: {config.watch_folder}")
    return snapshot


def _render_result(result: SearchResult, slot: Any, config: AppConfig) -> None:
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

        if path.is_file() and path.is_relative_to(config.watch_folder.resolve()):
            try:
                ocr_text = _ocr_text_for_image(config.database_path, result.image_id)
                suggested_name = suggest_filename(ocr_text, path)
            except (OSError, sqlite3.Error) as exc:
                suggested_name = path.name
                st.caption(f"Could not read OCR text for a filename suggestion: {exc}")
            with st.form(f"rename-image-{result.image_id}"):
                st.text_input(
                    "Suggested filename",
                    value=suggested_name,
                    key=f"suggested-name-{result.image_id}",
                    help=(
                        "Edit the OCR-based suggestion before renaming. "
                        "The file extension is kept."
                    ),
                )
                renamed = st.form_submit_button("Rename file")
            if renamed:
                try:
                    new_path = rename_indexed_image(
                        config.database_path,
                        result.image_id,
                        path,
                        st.session_state[f"suggested-name-{result.image_id}"],
                        config.watch_folder,
                    )
                    load_index_snapshot.clear()
                    st.session_state[f"suggested-name-{result.image_id}"] = new_path.name
                    st.session_state["search_results"] = [
                        replace(item, path=str(new_path))
                        if item.image_id == result.image_id
                        else item
                        for item in st.session_state.get("search_results", [])
                    ]
                    st.session_state["rename_feedback"] = f"Renamed to {new_path.name}"
                    st.rerun()
                except (
                    FileExistsError, FileNotFoundError, OSError, ValueError, sqlite3.Error
                ) as exc:
                    st.error(f"Could not rename screenshot: {exc}")


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

    rename_feedback = st.session_state.pop("rename_feedback", None)
    if rename_feedback:
        st.success(rename_feedback)

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
                _render_result(result, columns[index % len(columns)], config)
    elif snapshot.total:
        st.info("Search your indexed screenshots above.")


if __name__ == "__main__":
    run_app()
