"""Local Ollama vision model access for screenshot naming."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
VISION_TIMEOUT_SECONDS = 600
MAX_VISION_IMAGE_SIDE = 1024


class VisionModelError(RuntimeError):
    """Raised when the local vision model cannot name a screenshot."""


class OllamaUnavailableError(VisionModelError):
    """Raised when no connection to the local Ollama server can be made."""


def describe_screenshot(
    image_path: Path,
    ocr_text: str,
    model_name: str,
    timeout_seconds: int | None = None,
) -> str:
    """Ask a local open-weight model for a short, evidence-grounded screenshot name."""
    timeout = VISION_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    with Image.open(image_path) as source:
        source.thumbnail((MAX_VISION_IMAGE_SIDE, MAX_VISION_IMAGE_SIDE), Image.Resampling.LANCZOS)
        resized = source.convert("RGB")
    image_buffer = BytesIO()
    resized.save(image_buffer, format="JPEG", quality=85, optimize=True)
    image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    prompt = (
        "Look at this screenshot and choose a short, specific filename that tells a person "
        "what is shown. Prefer a recognizable website or app plus its page or purpose, and "
        "include the browser only when it is visible. Use only evidence in the image; do not "
        "guess. Return a JSON object with one string field named name. The name should be "
        "2 to 7 words, without an extension, date, screenshot number, quotes, or explanation."
    )
    if ocr_text.strip():
        prompt += f"\nText extracted from this same screenshot for reference:\n{ocr_text[:2000]}"

    request_body = json.dumps(
        {
            "model": model_name,
            "keep_alive": "10m",
            "messages": [
                {"role": "user", "content": prompt, "images": [image]},
            ],
            "format": "json",
            "options": {"temperature": 0, "num_predict": 48},
            "stream": False,
        }
    ).encode("utf-8")
    request = Request(
        OLLAMA_CHAT_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise VisionModelError(
                f"Ollama model {model_name!r} is unavailable. Run `ollama pull {model_name}`."
            ) from exc
        raise VisionModelError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except TimeoutError as exc:
        raise VisionModelError(f"Ollama timed out after {timeout} s.") from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError) or "timed out" in str(exc.reason).casefold():
            raise VisionModelError(f"Ollama timed out after {timeout} s.") from exc
        raise OllamaUnavailableError(
            "Could not contact local Ollama at 127.0.0.1:11434. Make sure Ollama is running."
        ) from exc
    except OSError as exc:
        raise OllamaUnavailableError(
            "Could not contact local Ollama at 127.0.0.1:11434. Make sure Ollama is running."
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisionModelError("Ollama returned an unreadable response.") from exc

    if not isinstance(result, dict):
        raise VisionModelError("Ollama returned an unreadable response.")
    message = result.get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise VisionModelError("The vision model returned an empty filename suggestion.")
    try:
        parsed = json.loads(content)
        suggestion = parsed.get("name", "") if isinstance(parsed, dict) else ""
    except json.JSONDecodeError:
        suggestion = content
    if not isinstance(suggestion, str) or not suggestion.strip():
        raise VisionModelError("The vision model did not return a filename suggestion.")
    return suggestion.strip()
