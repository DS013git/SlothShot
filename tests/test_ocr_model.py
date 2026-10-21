"""Slow end-to-end OCR checks using the bundled CPU ONNX models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from rapidocr import RapidOCR

from slothshot.config import default_config
from slothshot.ocr import OCRReader


@pytest.mark.slow
def test_dark_mode_inversion_experiment(tmp_path: Path) -> None:
    """Compare OCR with and without inverting the same synthetic dark screen."""
    image_path = tmp_path / "dark-terminal.png"
    image = Image.new("RGB", (1000, 280), (24, 27, 32))
    ImageDraw.Draw(image).text(
        (44, 80),
        "ADDRESS ALREADY IN USE",
        font=ImageFont.load_default(size=52),
        fill=(240, 240, 240),
    )
    image.save(image_path)

    config = replace(default_config(), preprocess=True, invert_dark_mode=False)
    # Reuse the loaded model between both runs so this compares image input,
    # rather than model startup time.
    engine = RapidOCR()
    base_reader = OCRReader(config, engine_factory=lambda: engine)
    baseline = base_reader.read(image_path)
    inverted_config = replace(config, invert_dark_mode=True)
    inverted = OCRReader(inverted_config, engine_factory=lambda: engine).read(image_path)

    assert baseline.error is None
    assert inverted.error is None
    assert "ADDRESS ALREADY IN USE" in baseline.text
    assert "ADDRESS ALREADY IN USE" in inverted.text

    blank_path = tmp_path / "blank.png"
    Image.new("RGB", (1000, 280), "white").save(blank_path)
    blank_result = base_reader.read(blank_path)
    assert blank_result.error is None
    assert blank_result.text == ""
    assert blank_result.lines == ()

    # The result is available in the test output so behavior can be compared
    # if RapidOCR or the model version is changed later.
    print(
        f"dark screenshot confidences: original={baseline.average_confidence:.5f}, "
        f"inverted={inverted.average_confidence:.5f}"
    )
