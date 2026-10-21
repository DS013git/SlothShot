"""Unit tests for ground-truth parsing and retrieval metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from slothshot.evaluation import load_ground_truth, retrieval_metrics


def test_load_ground_truth_accepts_one_or_multiple_expected_images(tmp_path: Path) -> None:
    path = tmp_path / "truth.yaml"
    path.write_text(
        '"docker issue": "one.png"\n"network problem": ["two.png", "three.png"]\n',
        encoding="utf-8",
    )

    result = load_ground_truth(path)

    assert result == {"docker issue": {"one.png"}, "network problem": {"two.png", "three.png"}}


def test_load_ground_truth_rejects_invalid_yaml_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty mapping"):
        load_ground_truth(path)


def test_retrieval_metrics_compute_hits_and_reciprocal_rank() -> None:
    results = retrieval_metrics(
        [["one.png", "two.png"], ["other.png", "expected.png"], []],
        [{"one.png"}, {"expected.png"}, {"missing.png"}],
    )

    assert results == {
        "hit@1": pytest.approx(1 / 3),
        "hit@3": pytest.approx(2 / 3),
        "hit@5": pytest.approx(2 / 3),
        "mrr": pytest.approx(0.5),
    }


def test_retrieval_metrics_require_matching_non_empty_inputs() -> None:
    with pytest.raises(ValueError, match="same non-zero length"):
        retrieval_metrics([], [])
