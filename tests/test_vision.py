from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from slothshot import vision
from slothshot.vision import (
    OllamaTimeoutError,
    describe_screenshot,
    suggest_name_from_text,
)


def test_vision_request_resizes_image_to_512_and_uses_configured_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (3000, 1500), "navy").save(path)
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return json.dumps({"message": {"content": '{"name":"A Name"}'}}).encode()

    def fake_urlopen(request, *, timeout: int) -> Response:
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(vision, "urlopen", fake_urlopen)
    assert describe_screenshot(path, "ocr", "test-model", timeout_seconds=43) == "A Name"
    body = captured["body"]
    assert captured["timeout"] == 43
    assert body["keep_alive"] == "10m"
    image_data = base64.b64decode(body["messages"][0]["images"][0])
    with Image.open(BytesIO(image_data)) as resized:
        assert max(resized.size) == 512


def test_text_naming_sends_ocr_only_to_fake_ollama_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"message": {"content": '{"name":"GitHub Settings"}'}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        vision, "OLLAMA_CHAT_URL", f"http://127.0.0.1:{server.server_address[1]}/api/chat"
    )
    try:
        result = suggest_name_from_text(
            "GitHub account settings", "test-text-model", timeout_seconds=17
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result == "GitHub Settings"
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "test-text-model"
    assert request["format"] == "json"
    message = request["messages"][0]
    assert "GitHub account settings" in message["content"]
    assert "images" not in message


def test_vision_timeout_message_includes_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "shot.png"
    Image.new("RGB", (20, 12), "white").save(path)

    def timed_out(*_args: object, **_kwargs: object):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(vision, "urlopen", timed_out)
    with pytest.raises(OllamaTimeoutError, match="Ollama timed out after 7 s"):
        describe_screenshot(path, "", "test-model", timeout_seconds=7)
