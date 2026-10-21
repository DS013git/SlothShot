"""Tests for synthetic screenshots that never contain user files."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.make_sample_data import TRICKY_CASES, generate_samples
from slothshot.evaluation import load_ground_truth


def test_generator_requires_at_least_one_hundred_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="[Aa]t least 100"):
        generate_samples(tmp_path, 99)


def test_generator_creates_synthetic_images_and_known_queries(tmp_path: Path) -> None:
    truth_path = generate_samples(tmp_path, 100)

    images = list(tmp_path.glob("*.png"))
    assert len(images) == 100 + len(TRICKY_CASES)
    assert Image.open(images[0]).size == (1000, 620)
    truth = load_ground_truth(truth_path)
    # Eight regular scenarios plus every tricky case that has a query.
    assert len(truth) == 8 + sum(case[1] is not None for case in TRICKY_CASES)
    assert not list(tmp_path.glob("*.jpg"))


def test_tricky_cases_are_generated_and_listed_in_ground_truth(tmp_path: Path) -> None:
    truth = load_ground_truth(generate_samples(tmp_path, 100))

    assert (tmp_path / "tricky_blank.png").is_file()
    assert {"tricky_otp_message.png"} in truth.values()
    assert not any("tricky_blank.png" in names for names in truth.values())
