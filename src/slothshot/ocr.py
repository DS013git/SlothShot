"""Screenshot loading, gentle preprocessing, and local RapidOCR inference."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps, ImageStat, UnidentifiedImageError

from slothshot.config import AppConfig, load_config

logger = logging.getLogger("slothshot.ocr")
_UPSCALE_BELOW_SIDE = 900
OCR_MODEL_FILES = (
    "PP-OCRv6_det_small.onnx",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "PP-OCRv6_rec_small.onnx",
)


@dataclass(frozen=True)
class OCRLine:
    """One recognized text line and RapidOCR's confidence from 0 to 1."""

    text: str
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    """The result of one image read, including timing and recoverable errors."""

    text: str
    lines: tuple[OCRLine, ...]
    elapsed_seconds: float
    error: str | None = None

    @property
    def average_confidence(self) -> float:
        """Return average line confidence, or zero when OCR found no lines."""
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)


def is_dark_image(image: Image.Image) -> bool:
    """Estimate whether a screenshot has a dark background from its luminance."""
    luminance = ImageOps.grayscale(image)
    return ImageStat.Stat(luminance).mean[0] < 110


def prepare_image(
    image: Image.Image,
    *,
    max_side: int = 1600,
    enabled: bool = True,
    invert_dark_mode: bool = False,
) -> Image.Image:
    """Orient, bound, and optionally improve an image before sending it to OCR.

    Large images are always reduced to ``max_side`` to bound inference memory.
    When preprocessing is enabled, small screenshots are enlarged up to 900 px
    on their longest side, then converted to grayscale with mild contrast
    enhancement. Dark-mode inversion is available as an opt-in experiment.
    """
    if max_side < 1:
        raise ValueError("max_side must be a positive integer")

    prepared = ImageOps.exif_transpose(image).convert("RGB")
    prepared.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    if not enabled:
        return prepared

    longest_side = max(prepared.size)
    if longest_side < _UPSCALE_BELOW_SIDE:
        scale = min(2.0, _UPSCALE_BELOW_SIDE / longest_side, max_side / longest_side)
        if scale > 1.0:
            size = (round(prepared.width * scale), round(prepared.height * scale))
            prepared = prepared.resize(size, Image.Resampling.LANCZOS)

    grayscale = ImageOps.grayscale(prepared)
    grayscale = ImageOps.autocontrast(grayscale)
    grayscale = ImageEnhance.Contrast(grayscale).enhance(1.15)
    if invert_dark_mode and is_dark_image(prepared):
        grayscale = ImageOps.invert(grayscale)
    return grayscale


class OCRReader:
    """Reuse a lazily created RapidOCR engine for one or more screenshots."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        engine_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or load_config()
        self._engine_factory = engine_factory or self._make_engine
        self._engine: Any | None = None

    @staticmethod
    def _make_engine() -> Any:
        """Create RapidOCR only when the first valid image is ready."""
        import rapidocr
        from rapidocr import RapidOCR

        model_dir = Path(rapidocr.__file__).parent / "models"
        missing = [name for name in OCR_MODEL_FILES if not (model_dir / name).is_file()]
        if missing:
            missing_list = ", ".join(missing)
            raise RuntimeError(
                f"RapidOCR model files are missing from {model_dir}: {missing_list}. "
                "Reinstall SlothShot while online to restore the bundled models."
            )
        logger.info("RapidOCR ONNX models are available locally in %s", model_dir)
        # RapidOCR 3.x defaults to ONNX Runtime with CUDA disabled. This project
        # installs the CPU onnxruntime distribution, never the GPU package.
        return RapidOCR()

    def _get_engine(self) -> Any:
        if self._engine is None:
            self._engine = self._engine_factory()
        return self._engine

    def read(self, image_path: Path) -> OCRResult:
        """OCR one path without raising for unreadable or empty image files."""
        started = time.perf_counter()
        image_path = Path(image_path)
        try:
            with Image.open(image_path) as source:
                # Pillow can use decoder draft modes for some large formats;
                # thumbnail before loading a full-size copy when possible.
                source.thumbnail(
                    (self.config.max_image_side, self.config.max_image_side),
                    Image.Resampling.LANCZOS,
                )
                image = ImageOps.exif_transpose(source).copy()
        except (OSError, ValueError, Image.DecompressionBombError, UnidentifiedImageError) as exc:
            return self._error_result(image_path, started, f"Could not read image: {exc}")

        try:
            prepared = prepare_image(
                image,
                max_side=self.config.max_image_side,
                enabled=self.config.preprocess,
                invert_dark_mode=self.config.invert_dark_mode,
            )
            output = self._get_engine()(prepared)
            texts = output.txts or ()
            scores = list(output.scores or ())
            # A missing confidence should not discard a line or desynchronize
            # later text from its score if an engine returns partial metadata.
            scores.extend([0.0] * max(0, len(texts) - len(scores)))
            scores = scores[: len(texts)]
            lines = tuple(
                OCRLine(text=str(text).strip(), confidence=float(score))
                for text, score in zip(texts, scores, strict=True)
                if text is not None and str(text).strip()
            )
            text = "\n".join(line.text for line in lines)
        except Exception as exc:  # A bad image/model should not stop a folder scan.
            logger.exception("OCR failed for %s", image_path)
            return self._error_result(image_path, started, f"OCR failed: {exc}")

        elapsed = time.perf_counter() - started
        logger.info(
            "OCR completed for %s in %.3f seconds (%d lines)", image_path, elapsed, len(lines)
        )
        return OCRResult(text=text, lines=lines, elapsed_seconds=elapsed)

    @staticmethod
    def _error_result(image_path: Path, started: float, message: str) -> OCRResult:
        elapsed = time.perf_counter() - started
        logger.warning("%s: %s (%.3f seconds)", image_path, message, elapsed)
        return OCRResult(text="", lines=(), elapsed_seconds=elapsed, error=message)


def ocr_image(image_path: Path, config: AppConfig | None = None) -> OCRResult:
    """Convenience function for reading a single screenshot from the CLI."""
    return OCRReader(config).read(image_path)
