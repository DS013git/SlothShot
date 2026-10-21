"""Optional real-model smoke test (may download the model on first run)."""

from __future__ import annotations

import numpy as np
import pytest

from slothshot.config import default_config
from slothshot.embed import EmbeddingModel, embedding_model_cached


@pytest.mark.slow
def test_fastembed_cpu_model_produces_finite_vectors() -> None:
    """Exercise the installed FastEmbed API and local ONNX model end to end."""
    config = default_config()
    model = EmbeddingModel(config)
    documents = model.embed_documents(["Address already in use", "A sunny garden"])
    query = model.embed_query("docker port problem")

    assert len(documents) == 2
    assert documents[0].shape == query.shape == (384,)
    assert all(np.isfinite(vector).all() for vector in [*documents, query])
    assert embedding_model_cached(config.embedding_model, config.model_cache)
