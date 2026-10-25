from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from slothshot import cli, renaming, vision
from slothshot.db import connect_database
from slothshot.naming import rename_indexed_image
from slothshot.vision import OllamaTimeoutError


class _Reader:
    def __init__(self, _config: object) -> None:
        pass

    def read(self, image_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            text=f"OCR for {image_path.name}", average_confidence=1.0, error=None
        )


class _Embedder:
    model_name = "test-model"

    def __init__(self, _config: object) -> None:
        pass

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [np.array([1.0, float(len(text))], dtype=np.float32) for text in texts]


@contextmanager
def _fake_ollama() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        requests: list[str] = []

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = request["messages"][0]["content"]
            self.requests.append(prompt)
            if "bad.png" in prompt:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"temporary model error")
                return
            if "same.png" in prompt:
                name = "same"
            else:
                name = "Named A" if "a.png" in prompt else "Named Pending"
            body = json.dumps({"message": {"content": json.dumps({"name": name})}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 11434), Handler)
    except OSError:
        # A developer may already have Ollama on 11434. Keep tests isolated and
        # use the same local fake server on an ephemeral port in that case.
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/api/chat"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _setup(tmp_path: Path, monkeypatch: object) -> tuple[Path, Path, Path]:
    folder = tmp_path / "screenshots"
    folder.mkdir()
    data_dir = tmp_path / "data"
    config = tmp_path / "config.toml"
    config.write_text(
        f'[paths]\nwatch_folder = "{folder.as_posix()}"\n'
        f'data_dir = "{data_dir.as_posix()}"\n'
        f'model_cache = "{(tmp_path / "models").as_posix()}"\n',
        encoding="utf-8",
    )
    config.write_text(
        config.read_text(encoding="utf-8") + '\n[models]\nnaming = "vision"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "OCRReader", _Reader)
    monkeypatch.setattr(cli, "EmbeddingModel", _Embedder)
    return folder, data_dir / "slothshot.sqlite3", config


def _image(folder: Path, name: str) -> Path:
    path = folder / name
    color = "red" if name.startswith("bad") else "white"
    Image.new("RGB", (32, 24), color).save(path)
    return path


def _run(folder: Path, config: Path, rename: bool = True):
    args = ["index", str(folder), "--config", str(config)]
    if rename:
        args.append("--rename-new")
    return CliRunner().invoke(cli.app, args)


def _name_state(database: Path, path: Path) -> str:
    connection = connect_database(database)
    try:
        return str(
            connection.execute(
                "SELECT name_state FROM images WHERE path = ?", (str(path.resolve()),)
            ).fetchone()[0]
        )
    finally:
        connection.close()


def test_index_rename_new_renames_new_image(tmp_path: Path, monkeypatch) -> None:
    folder, database, config = _setup(tmp_path, monkeypatch)
    _image(folder, "a.png")
    with _fake_ollama() as url:
        monkeypatch.setattr(vision, "OLLAMA_CHAT_URL", url)
        result = _run(folder, config)

    destination = folder / "Named A.png"
    assert result.exit_code == 0, result.output
    assert "1 indexed, 1 renamed, 0 locked, 0 model errors" in result.output
    assert destination.is_file()
    assert _name_state(database, destination) == "done"


def test_index_rename_new_retries_preexisting_pending_image(tmp_path: Path, monkeypatch) -> None:
    folder, database, config = _setup(tmp_path, monkeypatch)
    source = _image(folder, "pending.png")
    first = _run(folder, config, rename=False)
    assert first.exit_code == 0, first.output
    assert _name_state(database, source) == "pending"

    with _fake_ollama() as url:
        monkeypatch.setattr(vision, "OLLAMA_CHAT_URL", url)
        second = _run(folder, config)

    destination = folder / "Named Pending.png"
    assert second.exit_code == 0, second.output
    assert destination.is_file()
    assert _name_state(database, destination) == "done"


def test_unavailable_ollama_keeps_pending_for_later_retry(tmp_path: Path, monkeypatch) -> None:
    folder, database, config = _setup(tmp_path, monkeypatch)
    source = _image(folder, "a.png")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
    monkeypatch.setattr(vision, "OLLAMA_CHAT_URL", f"http://127.0.0.1:{closed_port}/api/chat")
    failed = _run(folder, config)
    assert failed.exit_code == 2, failed.output
    assert _name_state(database, source) == "pending"

    with _fake_ollama() as url:
        monkeypatch.setattr(vision, "OLLAMA_CHAT_URL", url)
        retried = _run(folder, config)
    destination = folder / "Named A.png"
    assert retried.exit_code == 0, retried.output
    assert destination.is_file()
    assert _name_state(database, destination) == "done"


def test_one_vision_error_does_not_block_later_pending_names(tmp_path: Path, monkeypatch) -> None:
    folder, database, config = _setup(tmp_path, monkeypatch)
    bad = _image(folder, "bad.png")
    _image(folder, "a.png")
    with _fake_ollama() as url:
        monkeypatch.setattr(vision, "OLLAMA_CHAT_URL", url)
        result = _run(folder, config)

    assert result.exit_code == 2, result.output
    assert bad.is_file()
    assert _name_state(database, bad) == "pending"
    assert (folder / "Named A.png").is_file()
    assert _name_state(database, folder / "Named A.png") == "done"


def test_unchanged_vision_name_and_manual_rename_mark_done(tmp_path: Path, monkeypatch) -> None:
    folder, database, config = _setup(tmp_path, monkeypatch)
    source = _image(folder, "same.png")
    with _fake_ollama() as url:
        monkeypatch.setattr(vision, "OLLAMA_CHAT_URL", url)
        result = _run(folder, config)
    assert result.exit_code == 0, result.output
    assert _name_state(database, source) == "done"

    another = _image(folder, "manual.png")
    _run(folder, config, rename=False)
    connection = connect_database(database)
    try:
        image_id = int(
            connection.execute(
                "SELECT id FROM images WHERE path = ?", (str(another.resolve()),)
            ).fetchone()[0]
        )
    finally:
        connection.close()
    destination = rename_indexed_image(database, image_id, another, "User Name.png", folder)
    assert destination.is_file()
    assert _name_state(database, destination) == "done"


def test_rename_all_drops_missing_screenshot_from_index(tmp_path: Path, monkeypatch) -> None:
    folder, database, config = _setup(tmp_path, monkeypatch)
    source = _image(folder, "removed.png")
    indexed = _run(folder, config, rename=False)
    assert indexed.exit_code == 0, indexed.output
    source.unlink()

    result = CliRunner().invoke(cli.app, ["rename-all", str(folder), "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "1 missing" in result.output
    connection = connect_database(database)
    try:
        assert (
            connection.execute(
                "SELECT 1 FROM images WHERE path = ?", (str(source.resolve()),)
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_rename_all_stops_after_first_vision_timeout(tmp_path: Path, monkeypatch) -> None:
    folder, database, config = _setup(tmp_path, monkeypatch)
    first = _image(folder, "a.png")
    _image(folder, "b.png")
    indexed = _run(folder, config, rename=False)
    assert indexed.exit_code == 0, indexed.output

    requests: list[Path] = []

    def timed_out(path: Path, *_args, **_kwargs) -> str:
        requests.append(path)
        raise OllamaTimeoutError("Ollama timed out after 7 s.")

    monkeypatch.setattr(renaming, "describe_screenshot", timed_out)
    result = CliRunner().invoke(cli.app, ["rename-all", str(folder), "--config", str(config)])

    assert result.exit_code == 2, result.output
    assert len(requests) == 1
    assert result.output.count("Vision naming timed out") == 1
    assert "stopping this run" in result.output
    assert first.is_file()
    assert (folder / "b.png").is_file()
    connection = connect_database(database)
    try:
        states = connection.execute("SELECT name_state FROM images ORDER BY path").fetchall()
        assert [row[0] for row in states] == ["pending", "pending"]
        assert connection.execute("SELECT COUNT(*) FROM image_name_claims").fetchone()[0] == 0
    finally:
        connection.close()
