"""OCR-based screenshot names and safe, index-aware file renaming."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

_FILENAME_SKIP_WORDS = {
    "and", "are", "for", "from", "http", "https", "into", "the", "this", "that",
    "with", "www", "com", "org", "net", "home", "file", "edit", "view", "help",
    "chrome", "firefox", "safari", "google", "website", "browser", "bookmarks", "new", "tab",
}
_DATE_OR_TIME_LINE = re.compile(
    r"(?i)(?:\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b)"
)
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


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
        title = summary
    elif browser:
        summary = page_context or ("Website" if domain_match else None)
        if summary:
            title = f"{summary} in {browser}"

    selected: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" \t|_—-:;,.•")
        if not line or _DATE_OR_TIME_LINE.search(line):
            continue
        words = re.findall(r"[^\W_]+", line, flags=re.UNICODE)
        meaningful = [
            word
            for word in words
            if len(word) >= 3
            and not word.isdigit()
            and word.casefold() not in _FILENAME_SKIP_WORDS
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
    if not title:
        title = "Screenshot"

    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    stem = re.sub(r"\s+", " ", stem).strip(" .")[:90].rstrip(" .")
    if not stem or stem.upper() in _RESERVED_WINDOWS_NAMES:
        stem = "Screenshot"
    return f"{stem}{current_path.suffix.lower()}"


def rename_indexed_image(
    connection: sqlite3.Connection,
    image_id: int,
    old_path: Path,
    new_name: str,
    watch_folder: Path,
) -> Path:
    """Rename one indexed screenshot without overwriting files or losing index sync."""
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
    if any(char in stem for char in '<>:/\\|?*') or any(ord(char) < 32 for char in stem):
        raise ValueError("The filename contains characters that Windows does not allow.")
    if stem.upper() in _RESERVED_WINDOWS_NAMES:
        raise ValueError("That filename is reserved by Windows.")

    connection.execute("BEGIN IMMEDIATE")
    moved = False
    try:
        row = connection.execute("SELECT path FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None or Path(str(row["path"])).resolve() != source:
            raise ValueError("The indexed path changed. Refresh search results and try again.")

        destination = source.with_name(f"{stem}{source.suffix}")
        if destination == source:
            connection.rollback()
            return source

        destination = _collision_safe_path(connection, image_id, destination)
        source.rename(destination)
        moved = True
        cursor = connection.execute(
            "UPDATE images SET path = ? WHERE id = ?", (str(destination.resolve()), image_id)
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Could not update the indexed screenshot path.")
        connection.commit()
    except Exception:
        connection.rollback()
        if moved and destination.exists() and not source.exists():
            destination.rename(source)
        raise
    return destination


def _collision_safe_path(
    connection: sqlite3.Connection, image_id: int, preferred: Path
) -> Path:
    """Keep a stable image-id suffix when OCR produces duplicate filenames."""
    if _path_is_available(connection, image_id, preferred):
        return preferred
    suffix_number = 1
    while True:
        collision_suffix = (
            f" ({image_id})" if suffix_number == 1 else f" ({image_id}-{suffix_number})"
        )
        candidate = preferred.with_name(f"{preferred.stem}{collision_suffix}{preferred.suffix}")
        if _path_is_available(connection, image_id, candidate):
            return candidate
        suffix_number += 1


def _path_is_available(connection: sqlite3.Connection, image_id: int, path: Path) -> bool:
    """Return whether a destination is unused by both files and index rows."""
    indexed = connection.execute(
        "SELECT id FROM images WHERE path = ?", (str(path.resolve()),)
    ).fetchone()
    if indexed is not None:
        return int(indexed["id"]) == image_id and path.is_file()
    return not path.exists()
