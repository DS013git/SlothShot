"""Generate synthetic screenshot files and YAML search ground truth."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("slothshot.samples")
SCENARIOS = (
    (
        "terminal",
        "docker port problem",
        "Docker Compose failed",
        "Error: bind: address already in use",
        "Port 5432 is already allocated",
        "Stop the other service and retry",
    ),
    (
        "terminal",
        "python package install failed",
        "pip install my-tool",
        "ERROR: Could not find a version satisfying the requirement my-tool",
        "No matching distribution found",
        "Python 3.11 environment",
    ),
    (
        "terminal",
        "git branch merge conflict",
        "git pull origin main",
        "CONFLICT in src/app.py",
        "Automatic merge failed; fix conflicts and commit",
        "branch feature/search",
    ),
    (
        "code",
        "python sqlite database locked",
        "def save_result(connection, row):",
        "with connection:",
        "connection.execute(SQL, row)",
        "sqlite3.OperationalError: database is locked",
    ),
    (
        "chat",
        "teammate shared meeting notes",
        "Maya  10:42 AM",
        "I put the meeting notes in the shared folder.",
        "The action items are due Friday.",
        "Arjun  10:43 AM  I will review them.",
    ),
    (
        "settings",
        "wifi network password settings",
        "Network settings",
        "Connected to: Home Office",
        "Security: WPA2-Personal",
        "Metered connection: Off",
    ),
    (
        "terminal",
        "npm server port already used",
        "npm run dev",
        "Error: listen EADDRINUSE: address already in use",
        "127.0.0.1:5173",
        "Stop the running process",
    ),
    (
        "chat",
        "friend sent recipe ingredients",
        "Riya  8:16 PM",
        "For the lentil soup use cumin, garlic, and lemon.",
        "Simmer until the lentils are soft.",
        "You  8:18 PM  Saving this!",
    ),
)

# Hard cases appended after the regular screenshots. Each one stresses a known OCR
# weakness: tiny text, low contrast, many lines, or no text at all.
# Fields: (file stem, query or None, text lines, font size, background, text color)
TRICKY_CASES = (
    (
        "tricky_tiny_text",
        "program crashed with segmentation fault",
        ("$ ./render --scene city", "Segmentation fault (core dumped)", "exit code 139"),
        11,
        (27, 31, 38),
        (210, 215, 222),
    ),
    (
        "tricky_low_contrast",
        "payment received confirmation",
        ("Payment received", "Amount: 2,450 rupees", "Reference: UPI-77310"),
        30,
        (120, 124, 130),
        (150, 154, 160),
    ),
    (
        "tricky_otp_message",
        "bank one time password code",
        ("Your verification code is 482915.", "Never share this code with anyone."),
        30,
        (247, 248, 250),
        (32, 40, 50),
    ),
    (
        "tricky_dense_config",
        "javascript package dependencies list",
        (
            '"name": "demo-app", "version": "1.2.0",',
            '"dependencies": { "react": "18.2.0",',
            '"express": "4.19.2", "zod": "3.23.8" },',
            '"scripts": { "dev": "vite", "test": "vitest" }',
        ),
        15,
        (255, 255, 255),
        (40, 44, 52),
    ),
    ("tricky_blank", None, (), 30, (247, 248, 250), (32, 40, 50)),
)


def _draw_tricky(path: Path, case: tuple) -> None:
    """Draw one hard case: small, faint, dense, or completely empty text."""
    _stem, _query, lines, font_size, background, color = case
    image = Image.new("RGB", (1000, 620), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)
    spacing = max(28, round(font_size * 2.2))
    for index, line in enumerate(lines):
        draw.text((60, 80 + index * spacing), line, fill=color, font=font)
    image.save(path, format="PNG", optimize=True)


def _draw_screenshot(path: Path, number: int, scenario: tuple[str, ...]) -> None:
    """Draw a legible mock application window in an alternating color theme."""
    category, _query, *lines = scenario
    dark = number % 2 == 0
    background = (27, 31, 38) if dark else (247, 248, 250)
    foreground = (232, 237, 242) if dark else (32, 40, 50)
    accent = (100, 190, 150) if dark else (45, 115, 190)
    image = Image.new("RGB", (1000, 620), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=25 if number % 9 == 0 else 30)
    small = ImageFont.load_default(size=19)
    draw.rectangle((0, 0, 1000, 54), fill=(50, 56, 66) if dark else (225, 230, 236))
    draw.text(
        (24, 16), f"{category.title()} | synthetic window {number:03d}", fill=accent, font=small
    )
    draw.rounded_rectangle(
        (45, 94, 955, 570),
        radius=14,
        fill=(36, 41, 49) if dark else (255, 255, 255),
        outline=(68, 76, 88) if dark else (208, 215, 223),
        width=2,
    )
    y = 142
    for line_number, line in enumerate(lines):
        color = (
            (255, 111, 105)
            if any(word in line for word in ("ERROR", "failed", "CONFLICT"))
            else foreground
        )
        if category == "chat" and line_number in {0, 3}:
            draw.rounded_rectangle(
                (75, y - 10, 925, y + 42), radius=12, fill=(50, 63, 76) if dark else (230, 241, 250)
            )
        draw.text((86, y), line, fill=color, font=font)
        y += 76
    draw.text((78, 526), f"window id {number:04d} | local demo data", fill=accent, font=small)
    image.save(path, format="PNG", optimize=True)


def generate_samples(output: Path, count: int = 120) -> Path:
    """Create ``count`` regular PNGs plus the tricky cases; return the truth-set path."""
    if count < 100:
        raise ValueError("At least 100 screenshots are required")
    output.mkdir(parents=True, exist_ok=True)
    truth = {scenario[1]: [] for scenario in SCENARIOS}
    for number in range(count):
        scenario = SCENARIOS[number % len(SCENARIOS)]
        name = f"{scenario[0]}_{number:03d}.png"
        _draw_screenshot(output / name, number, scenario)
        truth[scenario[1]].append(name)
    for case in TRICKY_CASES:
        name = f"{case[0]}.png"
        _draw_tricky(output / name, case)
        if case[1] is not None:
            truth[case[1]] = [name]
    truth_path = output / "ground_truth.yaml"
    # JSON quoted strings are also valid YAML scalars.
    truth_path.write_text(
        "\n".join(f"{json.dumps(query)}: {json.dumps(images)}" for query, images in truth.items())
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Created %d regular + %d tricky synthetic screenshots in %s",
        count,
        len(TRICKY_CASES),
        output.resolve(),
    )
    logger.info("Ground truth saved to %s", truth_path.resolve())
    return truth_path


def main() -> None:
    """Run the generator from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic-screenshots"))
    parser.add_argument("--count", type=int, default=120)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        generate_samples(args.output, args.count)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
