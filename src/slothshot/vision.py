"""Local Ollama access for OCR-only and vision-assisted screenshot naming."""

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
MAX_VISION_IMAGE_SIDE = 512


class VisionModelError(RuntimeError):
    """Raised when a local Ollama model cannot name a screenshot."""


class OllamaUnavailableError(VisionModelError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaTimeoutError(VisionModelError):
    """Raised when a local Ollama naming request exceeds its time budget."""


def _request_name(
    prompt: str,
    model_name: str,
    timeout_seconds: int,
    *,
    image: str | None = None,
) -> str:
    message: dict[str, object] = {"role": "user", "content": prompt}
    if image is not None:
        message["images"] = [image]
    request_body = json.dumps(
        {
            "model": model_name,
            "keep_alive": "10m",
            "messages": [message],
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
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise VisionModelError(
                f"Ollama model {model_name!r} is unavailable. Run `ollama pull {model_name}`."
            ) from exc
        raise VisionModelError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except TimeoutError as exc:
        raise OllamaTimeoutError(f"Ollama timed out after {timeout_seconds} s.") from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError) or "timed out" in str(exc.reason).casefold():
            raise OllamaTimeoutError(f"Ollama timed out after {timeout_seconds} s.") from exc
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
    message_result = result.get("message", {})
    content = message_result.get("content") if isinstance(message_result, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise VisionModelError("The naming model returned an empty filename suggestion.")
    try:
        parsed = json.loads(content)
        suggestion = parsed.get("name", "") if isinstance(parsed, dict) else ""
    except json.JSONDecodeError:
        suggestion = content
    if not isinstance(suggestion, str) or not suggestion.strip():
        raise VisionModelError("The naming model did not return a filename suggestion.")
    return suggestion.strip()


def suggest_name_from_text(
    ocr_text: str,
    model_name: str = "qwen3:1.7b",
    timeout_seconds: int = VISION_TIMEOUT_SECONDS,
) -> str:
    """Ask a small text model for a concise name using OCR text only."""
    prompt = (
        "Choose a short, specific filename for a screenshot using only the OCR text below. "
        "Return a JSON object with one string field named name. The name should be 2 to 7 "
        "words, without an extension, date, screenshot number, quotes, or explanation.\n"
        "OCR text:\n"
        f"{ocr_text[:2000]}"
    )
    return _request_name(prompt, model_name, timeout_seconds)


def describe_screenshot(
    image_path: Path,
    ocr_text: str,
    model_name: str,
    timeout_seconds: int | None = None,
) -> str:
    """Ask a local vision model for a short, evidence-grounded screenshot name."""
    timeout = VISION_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    # Detach image decoding from the source path before the eventual in-place rename.
    image_bytes = image_path.read_bytes()
    with Image.open(BytesIO(image_bytes)) as source:
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
    return _request_name(prompt, model_name, timeout, image=image)
