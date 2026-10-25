"""Stability, debounce, task ordering, and worker lifecycle tests."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from slothshot import renaming
from slothshot.config import default_config
from slothshot.db import connect_database
from slothshot.indexer import index_image
from slothshot.watcher import (
    DebouncedEventHandler,
    ScreenshotWatcher,
    WatchTask,
    wait_for_stable_file,
)


class FakeReader:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def read(self, image_path: Path) -> SimpleNamespace:
        self.calls.append(image_path)
        return SimpleNamespace(
            text=f"words from {image_path.stem}", average_confidence=0.9, error=None
        )


class FakeEmbedder:
    model_name = "test-model"

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [np.array([1, len(texts)], dtype=np.float32) for _ in texts]


class FakeObserver:
    def __init__(self) -> None:
        self.handler = None
        self.started = False

    def schedule(self, event_handler, path: str, *, recursive: bool = False) -> None:
        assert recursive
        self.handler = event_handler

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def join(self) -> None:
        pass

    def is_alive(self) -> bool:
        return self.started


class TrackedConnection:
    """Record worker closure while delegating SQLite operations safely."""

    def __init__(self, connection) -> None:
        self.connection = connection
        self.closed = False

    def __getattr__(self, name: str):
        return getattr(self.connection, name)

    def __enter__(self):
        return self.connection.__enter__()

    def __exit__(self, *args):
        return self.connection.__exit__(*args)

    def close(self) -> None:
        self.connection.close()
        self.closed = True


def _save(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = "blue" if path.stem == "new" else "white"
    Image.new("RGB", (80, 50), color).save(path)
    return path


def test_stability_waiter_requires_unchanged_file_and_handles_shutdown(tmp_path: Path) -> None:
    file_path = tmp_path / "stable.png"
    file_path.write_bytes(b"image data")
    stop = threading.Event()

    assert wait_for_stable_file(
        file_path, stable_seconds=0.02, timeout_seconds=0.2, poll_interval=0.01
    )
    stop.set()
    assert not wait_for_stable_file(
        file_path,
        stable_seconds=0.02,
        timeout_seconds=0.2,
        poll_interval=0.01,
        stop_event=stop,
    )
    started = time.monotonic()
    assert not wait_for_stable_file(
        tmp_path / "missing.png",
        stable_seconds=0.01,
        timeout_seconds=2.0,
        poll_interval=0.01,
    )
    assert time.monotonic() - started < 0.5


def test_event_handler_debounces_create_and_modified_events(tmp_path: Path) -> None:
    tasks: queue.Queue[WatchTask] = queue.Queue()
    times = iter((10.0, 10.1))
    handler = DebouncedEventHandler(
        tmp_path, tasks, debounce_seconds=0.4, clock=lambda: next(times)
    )
    path = tmp_path / "shot.png"

    handler.dispatch(FileCreatedEvent(str(path)))
    handler.dispatch(FileModifiedEvent(str(path)))

    task = tasks.get_nowait()
    assert task.action == "upsert"
    assert task.path == path.resolve()
    assert tasks.empty()


def test_event_handler_ignores_partial_files_and_queues_file_moves(tmp_path: Path) -> None:
    tasks: queue.Queue[WatchTask] = queue.Queue()
    handler = DebouncedEventHandler(tmp_path, tasks, clock=lambda: time.monotonic())
    partial = tmp_path / "shot.png.partial"
    partial_with_image_suffix = tmp_path / "shot.partial.png"
    source = tmp_path / "old.png"
    destination = tmp_path / "new.png"

    handler.dispatch(FileCreatedEvent(str(partial)))
    handler.dispatch(FileCreatedEvent(str(partial_with_image_suffix)))
    handler.dispatch(FileMovedEvent(str(source), str(destination)))

    task = tasks.get_nowait()
    assert task.action == "move"
    assert task.path == source.resolve()
    assert task.destination == destination.resolve()
    assert tasks.empty()


def test_event_handler_queues_directory_scan_and_cleanup(tmp_path: Path) -> None:
    tasks: queue.Queue[WatchTask] = queue.Queue()
    handler = DebouncedEventHandler(tmp_path, tasks)
    created = tmp_path / "new-folder"
    deleted = tmp_path / "gone-folder"

    handler.dispatch(DirCreatedEvent(str(created)))
    handler.dispatch(DirDeletedEvent(str(deleted)))

    assert [tasks.get_nowait().action, tasks.get_nowait().action] == ["scan_tree", "delete_tree"]


def test_worker_catches_up_then_handles_file_events_and_closes_connection(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    config = replace(
        default_config(),
        watch_folder=folder,
        data_dir=tmp_path / "app-data",
        model_cache=tmp_path / "models",
    )
    observer = FakeObserver()
    worker_thread_ids: list[int] = []
    connections: list[TrackedConnection] = []
    stability_checks: list[Path] = []
    _save(folder / "existing.png")

    def connection_factory(path: Path):
        worker_thread_ids.append(threading.get_ident())
        connection = TrackedConnection(connect_database(path))
        connections.append(connection)
        return connection

    watcher = ScreenshotWatcher(
        config,
        reader=FakeReader(),
        embedder=FakeEmbedder(),
        observer_factory=lambda: observer,
        connection_factory=connection_factory,
        stability_waiter=lambda path, **_kwargs: stability_checks.append(path) or True,
    )
    watcher.start()
    try:
        assert watcher.catchup_complete.wait(2.0)
        assert folder / "existing.png" in stability_checks
        image_path = _save(folder / "new.png")
        observer.handler.dispatch(FileCreatedEvent(str(image_path)))
        watcher.tasks.join()
        read_connection = connect_database(config.database_path)
        texts = {
            item["ocr_text"] for item in read_connection.execute("SELECT ocr_text FROM images")
        }
        assert texts == {"words from existing", "words from new"}

        observer.handler.dispatch(FileDeletedEvent(str(image_path)))
        watcher.tasks.join()
        assert read_connection.execute("SELECT count(*) FROM images").fetchone()[0] == 1
        read_connection.close()
    finally:
        watcher.stop()

    assert len(worker_thread_ids) == 1
    assert worker_thread_ids[0] != threading.get_ident()
    assert not observer.is_alive()
    assert connections[0].closed


def test_watcher_renames_new_image_and_ignores_its_move_event(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    config = replace(
        default_config(),
        watch_folder=folder,
        data_dir=tmp_path / "app-data",
        model_cache=tmp_path / "models",
        naming="vision",
        auto_rename=True,
    )
    observer = FakeObserver()
    reader = FakeReader()
    monkeypatch.setattr(renaming, "describe_screenshot", lambda *_args: "New Name")
    watcher = ScreenshotWatcher(
        config,
        reader=reader,
        embedder=FakeEmbedder(),
        observer_factory=lambda: observer,
        stability_waiter=lambda *_args, **_kwargs: True,
    )
    watcher.start()
    try:
        assert watcher.catchup_complete.wait(2.0)
        source = _save(folder / "new.png")
        observer.handler.dispatch(FileCreatedEvent(str(source)))
        watcher.tasks.join()

        destination = folder / "New Name.png"
        assert destination.is_file()
        assert len(reader.calls) == 1
        connection = connect_database(config.database_path)
        try:
            rows = connection.execute("SELECT path, name_state FROM images").fetchall()
            assert len(rows) == 1
            assert rows[0]["path"] == str(destination.resolve())
            assert rows[0]["name_state"] == "done"
        finally:
            connection.close()

        observer.handler.dispatch(FileMovedEvent(str(source), str(destination)))
        assert watcher.tasks.empty()
        assert len(reader.calls) == 1
    finally:
        watcher.stop()


def test_two_workers_claim_one_pending_rename_once(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    source = _save(folder / "race.png")
    database = tmp_path / "slothshot.sqlite3"
    connection = connect_database(database)
    assert index_image(connection, source, FakeReader()) == "indexed"
    row = connection.execute(
        "SELECT id FROM images WHERE path = ?", (str(source.resolve()),)
    ).fetchone()
    image_id = int(row["id"])
    connection.close()

    calls = 0
    call_lock = threading.Lock()

    def describe(*_args) -> str:
        nonlocal calls
        with call_lock:
            calls += 1
        time.sleep(0.1)
        return "Single Rename"

    monkeypatch.setattr(renaming, "describe_screenshot", describe)
    results = []
    errors: list[BaseException] = []

    def rename_worker() -> None:
        try:
            results.append(
                renaming.rename_images_with_vision(
                    database, folder, "fake-model", image_ids=[image_id]
                )
            )
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=rename_worker) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert not errors
    assert calls == 1
    assert sum(summary.renamed for summary in results) == 1
    assert (folder / "Single Rename.png").is_file()
    check = connect_database(database)
    try:
        assert check.execute("SELECT count(*) FROM images").fetchone()[0] == 1
        assert check.execute("SELECT count(*) FROM image_name_claims").fetchone()[0] == 0
    finally:
        check.close()
