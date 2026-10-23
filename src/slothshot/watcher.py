"""Background file watcher with debounced events and one SQLite worker."""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from slothshot.config import AppConfig, load_config
from slothshot.db import connect_database, image_by_path
from slothshot.embed import EmbeddingModel, reuse_embeddings_for_hash, save_image_embeddings
from slothshot.indexer import (
    SUPPORTED_IMAGE_SUFFIXES,
    ImageReader,
    index_image,
)
from slothshot.ocr import OCRReader
from slothshot.renaming import rename_images_with_vision

logger = logging.getLogger("slothshot.watcher")


def _event_path(value: bytes | str) -> Path:
    """Convert a watchdog event path, which may be ``bytes``, to a ``Path``."""
    return Path(os.fsdecode(value))


IGNORED_SUFFIXES = {".crdownload", ".part", ".partial", ".temp", ".tmp"}
STABLE_SECONDS = 0.75
STABILITY_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 0.15
DEBOUNCE_SECONDS = 0.4


@dataclass(frozen=True)
class WatchTask:
    """A filesystem operation to perform on the database worker thread."""

    action: str
    path: Path
    destination: Path | None = None


class TaskQueue(Protocol):
    """The small queue API used by watchdog's event handler."""

    def put(self, item: WatchTask) -> None: ...


def is_supported_image(path: Path) -> bool:
    """Return whether a path is an image that may be indexed."""
    suffix = path.suffix.lower()
    return (
        suffix in SUPPORTED_IMAGE_SUFFIXES
        and not any(part.lower() in IGNORED_SUFFIXES for part in path.suffixes)
        and not path.name.startswith("~$")
        and not path.name.endswith("~")
    )


def wait_for_stable_file(
    path: Path,
    *,
    stable_seconds: float = STABLE_SECONDS,
    timeout_seconds: float = STABILITY_TIMEOUT_SECONDS,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    stop_event: threading.Event | None = None,
) -> bool:
    """Wait for a file's size and modification time to remain unchanged.

    Returns false when the file disappears, does not settle before the timeout,
    or shutdown is requested. The worker can then safely move on without
    holding the entire queue behind a partial download indefinitely.
    """
    deadline = time.monotonic() + timeout_seconds
    last_state: tuple[int, int] | None = None
    unchanged_since: float | None = None
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return False
        try:
            stat = path.stat()
            state = (stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            # A queued create or modify event may arrive after the file has
            # already been deleted. Do not block later delete tasks for it.
            return False
        except OSError:
            last_state = None
            unchanged_since = None
            time.sleep(poll_interval)
            continue

        now = time.monotonic()
        if state != last_state:
            last_state = state
            unchanged_since = now
        elif unchanged_since is not None and now - unchanged_since >= stable_seconds:
            return True
        time.sleep(poll_interval)
    return False


class DebouncedEventHandler(FileSystemEventHandler):
    """Translate watchdog events into a small deduplicated task queue."""

    def __init__(
        self,
        root: Path,
        tasks: TaskQueue,
        *,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.root = root.resolve()
        self.tasks = tasks
        self.debounce_seconds = debounce_seconds
        self.clock = clock
        self._last_enqueued: dict[tuple[str, str, str], float] = {}
        self._suppressed_moves: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        path = _event_path(event.src_path)
        if event.is_directory:
            self._enqueue("scan_tree", path)
        elif is_supported_image(path):
            self._enqueue("upsert", path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            path = _event_path(event.src_path)
            if is_supported_image(path):
                self._enqueue("upsert", path)

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        action = "delete_tree" if event.is_directory else "delete"
        self._enqueue(action, _event_path(event.src_path), allow_outside=True)

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        source = _event_path(event.src_path)
        destination = _event_path(event.dest_path)
        if self._consume_suppressed_move(source, destination):
            return
        if event.is_directory:
            if self._is_inside_root(destination):
                self._enqueue("move_tree", source, destination, allow_outside=True)
            else:
                self._enqueue("delete_tree", source, allow_outside=True)
        elif is_supported_image(destination) and self._is_inside_root(destination):
            self._enqueue("move", source, destination, allow_outside=True)
        else:
            self._enqueue("delete", source, allow_outside=True)

    def suppress_rename_event(self, source: Path, destination: Path) -> None:
        """Ignore the move event generated by SlothShot's own database rename."""
        key = (str(source.resolve()).casefold(), str(destination.resolve()).casefold())
        with self._lock:
            self._suppressed_moves[key] = self.clock() + 30.0

    def _consume_suppressed_move(self, source: Path, destination: Path) -> bool:
        key = (str(source.resolve()).casefold(), str(destination.resolve()).casefold())
        now = self.clock()
        with self._lock:
            self._suppressed_moves = {
                old_key: expires
                for old_key, expires in self._suppressed_moves.items()
                if expires > now
            }
            return self._suppressed_moves.pop(key, None) is not None

    def _is_inside_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    def _enqueue(
        self,
        action: str,
        path: Path,
        destination: Path | None = None,
        *,
        allow_outside: bool = False,
    ) -> None:
        resolved_path = path.resolve()
        resolved_destination = destination.resolve() if destination else None
        if not allow_outside and not self._is_inside_root(resolved_path):
            return
        key = (
            action,
            str(resolved_path).casefold(),
            str(resolved_destination).casefold() if resolved_destination else "",
        )
        now = self.clock()
        with self._lock:
            self._last_enqueued = {
                old_key: stamp
                for old_key, stamp in self._last_enqueued.items()
                if now - stamp < self.debounce_seconds
            }
            last = self._last_enqueued.get(key)
            if last is not None and now - last < self.debounce_seconds:
                logger.debug("Debounced duplicate %s event for %s", action, resolved_path)
                return
            self._last_enqueued[key] = now
        self.tasks.put(WatchTask(action, resolved_path, resolved_destination))


class ScreenshotWatcher:
    """Coordinate catch-up indexing, watchdog, and one database-owning worker."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        reader: ImageReader | None = None,
        embedder: EmbeddingModel | None = None,
        observer_factory: Callable[[], BaseObserver] = Observer,
        connection_factory: Callable[[Path], sqlite3.Connection] = connect_database,
        stability_waiter: Callable[..., bool] = wait_for_stable_file,
    ) -> None:
        self.config = config or load_config()
        self.root = self.config.watch_folder.expanduser().resolve()
        self.reader = reader
        self.embedder = embedder
        self.observer_factory = observer_factory
        self.connection_factory = connection_factory
        self.stability_waiter = stability_waiter
        self.tasks: queue.Queue[WatchTask | None] = queue.Queue()
        self.stop_event = threading.Event()
        self.catchup_complete = threading.Event()
        self.worker_ready = threading.Event()
        self.worker_error: BaseException | None = None
        self.observer: BaseObserver | None = None
        self.handler: DebouncedEventHandler | None = None
        self.worker: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        """Start observation and queue catch-up before handling new events."""
        if self._started:
            return
        if not self.root.is_dir():
            raise NotADirectoryError(f"Screenshot folder does not exist: {self.root}")
        reader = self.reader or OCRReader(self.config)
        embedder = self.embedder or EmbeddingModel(self.config)
        self.reader = reader
        self.embedder = embedder
        self.observer = self.observer_factory()
        self.handler = DebouncedEventHandler(self.root, self.tasks)
        self.observer.schedule(self.handler, str(self.root), recursive=True)

        # The catch-up task goes in first. Start watching before the worker so
        # filesystem events queue behind that scan instead of slipping through.
        self.tasks.put(WatchTask("catchup", self.root))
        self.observer.start()
        self.worker = threading.Thread(
            target=self._worker_main,
            name="slothshot-index-worker",
            daemon=True,
        )
        self.worker.start()
        self.worker_ready.wait()
        if self.worker_error is not None:
            self.observer.stop()
            self.observer.join()
            raise RuntimeError(f"Could not start the index worker: {self.worker_error}")
        self._started = True
        logger.info("Watching %s for screenshot changes", self.root)

    def run(self) -> None:
        """Run until Ctrl+C and then drain the queue and stop all threads."""
        try:
            self.start()
            logger.info("Catch-up scan queued; new events wait behind it")
            while not self.stop_event.wait(1.0):
                pass
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received; stopping SlothShot watcher")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop observing, let queued work finish, and close the worker DB."""
        self.stop_event.set()
        if self.observer is not None and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        if self.worker is not None and self.worker.is_alive():
            self.tasks.join()
            self.tasks.put(None)
            self.worker.join()
        self._started = False

    def _worker_main(self) -> None:
        """Open one SQLite connection here and use it only on this thread."""
        try:
            connection = self.connection_factory(self.config.database_path)
        except BaseException as exc:
            self.worker_error = exc
            self.worker_ready.set()
            logger.exception("Could not open watcher database")
            return

        self.worker_ready.set()
        try:
            while True:
                task = self.tasks.get()
                try:
                    if task is None:
                        return
                    self._process_task(connection, task)
                except Exception:
                    logger.exception("Watcher task failed: %s", task)
                finally:
                    self.tasks.task_done()
        finally:
            connection.close()
            logger.info("Watcher database connection closed")

    def _process_task(self, connection: sqlite3.Connection, task: WatchTask) -> None:
        """Run one queued scan, upsert, move, or delete operation."""
        if task.action == "catchup":
            logger.info("Scanning for screenshots missed while SlothShot was stopped")
            assert self.reader is not None and self.embedder is not None
            self._catchup_scan(connection, task.path)
            self.catchup_complete.set()
            return
        if task.action == "upsert":
            self._process_upsert(connection, task.path)
        elif task.action == "delete":
            self._delete_path(connection, task.path)
        elif task.action == "delete_tree":
            self._delete_tree(connection, task.path)
        elif task.action == "move":
            if task.destination is not None:
                self._process_upsert(connection, task.destination)
            self._delete_path(connection, task.path)
        elif task.action == "scan_tree":
            self._scan_tree(connection, task.path)
        elif task.action == "move_tree":
            if task.destination is not None:
                self._scan_tree(connection, task.destination)
            self._delete_tree(connection, task.path)
        else:
            logger.warning("Ignoring unknown watcher task: %s", task.action)

    def _process_upsert(self, connection: sqlite3.Connection, path: Path) -> None:
        if not is_supported_image(path):
            return
        if not self.stability_waiter(path, stop_event=self.stop_event):
            logger.info("Skipped file that did not become stable: %s", path)
            return
        assert self.reader is not None and self.embedder is not None
        outcome = index_image(connection, path, self.reader)
        if outcome == "failed":
            return
        row = image_by_path(connection, path)
        if row is None or row["status"] != "indexed":
            return
        if outcome == "reused":
            reuse_embeddings_for_hash(
                connection, image_id=int(row["id"]), sha256=str(row["sha256"])
            )
        save_image_embeddings(
            connection,
            image_id=int(row["id"]),
            text=str(row["ocr_text"]),
            model=self.embedder,
        )
        if self.config.auto_rename:
            rename_images_with_vision(
                self.config.database_path,
                self.root,
                self.config.vision_model,
                vision_timeout_seconds=self.config.vision_timeout_seconds,
                image_ids=[int(row["id"])],
                before_move=self._suppress_rename_event,
            )
        logger.info("Indexed screenshot: %s", path)

    def _suppress_rename_event(self, source: Path, destination: Path) -> None:
        if self.handler is not None:
            self.handler.suppress_rename_event(source, destination)

    def _scan_tree(self, connection: sqlite3.Connection, folder: Path) -> None:
        if not folder.is_dir():
            return
        for path in sorted(folder.rglob("*"), key=lambda item: str(item).casefold()):
            if path.is_file() and is_supported_image(path):
                self._process_upsert(connection, path)

    def _catchup_scan(self, connection: sqlite3.Connection, folder: Path) -> None:
        """Wait for current files to settle, index them, and prune stale rows."""
        paths = sorted(
            (path for path in folder.rglob("*") if path.is_file() and is_supported_image(path)),
            key=lambda item: str(item).casefold(),
        )
        discovered = {str(path.resolve()) for path in paths}
        logger.info("Catch-up found %d supported images under %s", len(paths), folder)
        for path in paths:
            self._process_upsert(connection, path)

        root = str(folder.resolve()).rstrip("\\/")
        prefix = _escape_like(root + _separator(root)) + "%"
        stale_rows = connection.execute(
            "SELECT id, path FROM images WHERE path = ? OR path LIKE ? ESCAPE '\\'",
            (root, prefix),
        ).fetchall()
        removed = 0
        for row in stale_rows:
            if row["path"] not in discovered:
                with connection:
                    connection.execute("DELETE FROM images WHERE id = ?", (row["id"],))
                removed += 1
        logger.info("Catch-up scan complete; removed %d stale record(s)", removed)

    def _delete_path(self, connection: sqlite3.Connection, path: Path) -> None:
        normalized = str(path.resolve())
        with connection:
            cursor = connection.execute("DELETE FROM images WHERE path = ?", (normalized,))
        if cursor.rowcount:
            logger.info("Removed deleted screenshot from index: %s", normalized)

    def _delete_tree(self, connection: sqlite3.Connection, folder: Path) -> None:
        normalized = str(folder.resolve()).rstrip("\\/")
        prefix = _escape_like(normalized) + _escape_like(_separator(normalized)) + "%"
        with connection:
            cursor = connection.execute(
                "DELETE FROM images WHERE path = ? OR path LIKE ? ESCAPE '\\'",
                (normalized, prefix),
            )
        if cursor.rowcount:
            logger.info("Removed %d indexed screenshot(s) under %s", cursor.rowcount, normalized)


def _separator(path: str) -> str:
    """Choose the host separator for a normalized absolute path."""
    return "\\" if "\\" in path else "/"


def _escape_like(value: str) -> str:
    """Escape LIKE metacharacters and the escape character in a path."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
