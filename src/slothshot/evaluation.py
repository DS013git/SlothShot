"""Load search evaluation sets and calculate retrieval quality metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    """Load a YAML mapping of each query to one or more expected image names.

    Expected values may be a single string or a list of strings. The safe YAML
    loader prevents input files from constructing arbitrary Python objects.
    """
    try:
        with path.open(encoding="utf-8") as stream:
            raw: Any = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read evaluation YAML {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Evaluation YAML must be a non-empty mapping of queries to images")

    result: dict[str, set[str]] = {}
    for query, expected in raw.items():
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Each evaluation query must be a non-empty string")
        values = [expected] if isinstance(expected, str) else expected
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item.strip() for item in values)
        ):
            raise ValueError(
                f"Expected images for query {query!r} must be a string or list of strings"
            )
        result[query] = {item.casefold() for item in values}
    return result


def retrieval_metrics(
    rankings: list[list[str]],
    expected_sets: list[set[str]],
) -> dict[str, float]:
    """Compute top-k hit rates and reciprocal rank averaged across queries."""
    if len(rankings) != len(expected_sets) or not rankings:
        raise ValueError("Rankings and expected sets must have the same non-zero length")
    hits = {cutoff: 0 for cutoff in (1, 3, 5)}
    reciprocal_ranks: list[float] = []
    for ranking, expected in zip(rankings, expected_sets, strict=True):
        normalized = [Path(path).name.casefold() for path in ranking]
        first_rank = next(
            (rank for rank, name in enumerate(normalized, start=1) if name in expected), None
        )
        if first_rank is not None:
            reciprocal_ranks.append(1.0 / first_rank)
            for cutoff in hits:
                hits[cutoff] += first_rank <= cutoff
        else:
            reciprocal_ranks.append(0.0)
    total = len(rankings)
    return {
        "hit@1": hits[1] / total,
        "hit@3": hits[3] / total,
        "hit@5": hits[5] / total,
        "mrr": sum(reciprocal_ranks) / total,
    }
