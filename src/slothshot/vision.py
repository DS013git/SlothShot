"""Local Ollama vision model access for screenshot naming."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
VISION_TIMEOUT_SECONDS = 180


class VisionModelError(RuntimeError):
    """Raised when the local vision model cannot name a screenshot."""


def describe_screenshot(image_path: Path, ocr_text: str, model_name: str) -> str:
    """Ask a local open-weight model for a short, evidence-grounded screenshot name."""
    image = base64.b64encode(image_path.read_bytes()).decode("ascii")
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
        with urlopen(request, timeout=VISION_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise VisionModelError(
                f"Ollama model {model_name!r} is unavailable. Run `ollama pull {model_name}`."
            ) from exc
        raise VisionModelError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except (OSError, TimeoutError, URLError) as exc:
        raise VisionModelError(
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
