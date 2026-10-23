from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from slothshot import vision
from slothshot.vision import VisionModelError, describe_screenshot


def test_vision_request_resizes_image_and_uses_configured_timeout(
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
        assert max(resized.size) == 1024


def test_vision_timeout_message_includes_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "shot.png"
    Image.new("RGB", (20, 12), "white").save(path)

    def timed_out(*_args: object, **_kwargs: object):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(vision, "urlopen", timed_out)
    with pytest.raises(VisionModelError, match="Ollama timed out after 7 s"):
        describe_screenshot(path, "", "test-model", timeout_seconds=7)
