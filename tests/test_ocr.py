"""Fast, model-free checks for screenshot preparation and OCR error handling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from slothshot.config import default_config
from slothshot.ocr import OCRReader, is_dark_image, prepare_image


def test_large_images_are_bounded_even_without_preprocessing() -> None:
    image = Image.new("RGB", (2400, 1600), "white")

    prepared = prepare_image(image, max_side=1200, enabled=False)

    assert max(prepared.size) == 1200
    assert prepared.size == (1200, 800)


def test_small_screenshots_are_upscaled_to_a_safe_limit() -> None:
    image = Image.new("RGB", (400, 200), "white")

    prepared = prepare_image(image, max_side=1600)

    assert prepared.mode == "L"
    assert prepared.size == (800, 400)


def test_dark_mode_inversion_is_optional() -> None:
    image = Image.new("RGB", (300, 180), (20, 25, 30))
    ImageDraw.Draw(image).text((40, 80), "WHITE TEXT", fill="white")

    unchanged = prepare_image(image, enabled=True)
    inverted = prepare_image(image, enabled=True, invert_dark_mode=True)

    assert is_dark_image(image)
    assert unchanged.getpixel((0, 0)) < 100
    assert inverted.getpixel((0, 0)) > 150


def test_corrupt_image_returns_a_timed_error_without_loading_engine(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    reader = OCRReader(engine_factory=lambda: pytest.fail("engine should stay lazy"))

    result = reader.read(path)

    assert result.error is not None
    assert result.text == ""
    assert result.lines == ()
    assert result.elapsed_seconds >= 0


def test_empty_ocr_result_is_a_success_with_no_text(tmp_path: Path) -> None:
    path = tmp_path / "blank.png"
    Image.new("RGB", (320, 240), "white").save(path)
    empty_output = SimpleNamespace(txts=None, scores=None)
    reader = OCRReader(engine_factory=lambda: lambda _image: empty_output)

    result = reader.read(path)

    assert result.error is None
    assert result.text == ""
    assert result.lines == ()
    assert result.average_confidence == 0


def test_engine_failure_is_returned_instead_of_raised(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    Image.new("RGB", (320, 240), "white").save(path)

    def fail_on_image(_image: Image.Image) -> object:
        raise RuntimeError("model unavailable")

    reader = OCRReader(engine_factory=lambda: fail_on_image)
    result = reader.read(path)

    assert result.error == "OCR failed: model unavailable"
    assert result.elapsed_seconds >= 0


def test_line_text_and_confidence_are_returned(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    Image.new("RGB", (320, 240), "white").save(path)
    output = SimpleNamespace(txts=(" docker port ", "address already in use"), scores=(0.91, 0.83))
    reader = OCRReader(engine_factory=lambda: lambda _image: output)

    result = reader.read(path)

    assert result.text == "docker port\naddress already in use"
    assert [line.confidence for line in result.lines] == [0.91, 0.83]
    assert result.average_confidence == pytest.approx(0.87)


def test_missing_or_partial_scores_keep_all_lines(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    Image.new("RGB", (320, 240), "white").save(path)
    output = SimpleNamespace(txts=("hello", "world"), scores=(0.7,))
    reader = OCRReader(engine_factory=lambda: lambda _image: output)

    result = reader.read(path)

    assert result.text == "hello\nworld"
    assert [line.confidence for line in result.lines] == [0.7, 0.0]


def test_config_default_does_not_invert_dark_images() -> None:
    assert default_config().invert_dark_mode is False
