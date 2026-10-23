"""Chunking, embedding persistence, and semantic ranking tests."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from slothshot import embed as embed_module
from slothshot.config import default_config
from slothshot.db import connect_database
from slothshot.embed import (
    EmbeddingModel,
    chunk_text,
    embedding_model_cached,
    save_image_embeddings,
)
from slothshot.search import (
    SearchResult,
    cosine_similarity,
    reciprocal_rank_fusion,
    search_hybrid,
    search_semantic,
)


class FakeEngine:
    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []

    def embed(
        self, documents: list[str], batch_size: int = 16, parallel: int = 1
    ) -> list[np.ndarray]:
        self.document_batches.append(list(documents))
        return [np.array([len(text), 1], dtype=np.float32) for text in documents]

    def query_embed(self, query: str) -> list[np.ndarray]:
        return [np.array([1, 0], dtype=np.float32)]


def _add_image(connection: sqlite3.Connection, path: str, ocr_text: str = "sample OCR") -> int:
    cursor = connection.execute(
        """INSERT INTO images (
               path, sha256, width, height, ocr_text, avg_confidence,
               file_mtime, indexed_at, status
           ) VALUES (?, ?, 100, 50, ?, 0.8, 1.0,
                     '2026-10-21T00:00:00+00:00', 'indexed')""",
        (path, path, ocr_text),
    )
    return int(cursor.lastrowid)


def test_chunk_text_keeps_words_and_limits_chunk_size() -> None:
    text = "alpha beta\n" + "word " * 20

    chunks = chunk_text(text, max_characters=24)

    assert chunks[0] == "alpha beta word word"
    assert all(len(chunk) <= 24 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()
    assert chunk_text("") == []


def test_oversized_single_word_is_safely_sliced() -> None:
    chunks = chunk_text("abcdefghij", max_characters=4)

    assert chunks == ["abcd", "efgh", "ij"]


def test_embedding_wrapper_is_lazy_and_stores_float32_chunks(tmp_path: Path) -> None:
    engine = FakeEngine()
    model = EmbeddingModel(default_config(), engine_factory=lambda: engine)
    connection = connect_database(tmp_path / "embeddings.sqlite3")
    try:
        image_id = _add_image(connection, "C:/shots/one.png")
        assert engine.document_batches == []

        inserted = save_image_embeddings(
            connection, image_id=image_id, text="one two three", model=model
        )
        skipped = save_image_embeddings(
            connection, image_id=image_id, text="one two three", model=model
        )

        assert inserted == 1
        assert skipped == 0
        assert engine.document_batches == [["one two three"]]
        row = connection.execute(
            "SELECT embedding, chunk_text FROM embeddings WHERE image_id = ?", (image_id,)
        ).fetchone()
        assert np.frombuffer(row["embedding"], dtype=np.float32).shape == (2,)
        assert row["chunk_text"] == "one two three"
    finally:
        connection.close()


def test_concurrent_embedding_inserts_ignore_duplicate_chunks(tmp_path: Path) -> None:
    database = tmp_path / "racing-embeddings.sqlite3"
    connection = connect_database(database)
    image_id = _add_image(connection, "C:/shots/race.png")
    connection.commit()
    connection.close()
    barrier = threading.Barrier(2)

    class RacingModel:
        model_name = "race-model"

        def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
            barrier.wait(timeout=3)
            return [np.array([1.0, 2.0], dtype=np.float32) for _ in texts]

    errors: list[BaseException] = []

    def worker() -> None:
        worker_connection = connect_database(database)
        try:
            save_image_embeddings(
                worker_connection,
                image_id=image_id,
                text="same image content",
                model=RacingModel(),
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            worker_connection.close()

    workers = [threading.Thread(target=worker) for _ in range(2)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in workers)
    assert errors == []
    check = connect_database(database)
    try:
        assert (
            check.execute(
                "SELECT count(*) FROM embeddings WHERE image_id = ? AND model_name = 'race-model'",
                (image_id,),
            ).fetchone()[0]
            == 1
        )
    finally:
        check.close()


def test_cache_check_does_not_claim_empty_directory_is_ready(tmp_path: Path) -> None:
    assert embedding_model_cached("BAAI/bge-small-en-v1.5", tmp_path) is False


def test_cached_model_is_constructed_with_network_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastembed

    captured: dict[str, object] = {}

    class FakeTextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(embed_module, "embedding_model_cached", lambda *_args: True)
    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    model = EmbeddingModel(default_config())

    model._make_engine()

    assert captured["local_files_only"] is True
    assert captured["cuda"] is False
    assert captured["threads"] == 2


def test_semantic_search_uses_best_chunk_per_image_and_model_name(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "semantic.sqlite3")
    model = EmbeddingModel(default_config(), engine_factory=FakeEngine)
    try:
        nearer = _add_image(connection, "C:/shots/near.png")
        farther = _add_image(connection, "C:/shots/far.png")
        other_model = _add_image(connection, "C:/shots/other-model.png")
        connection.executemany(
            """INSERT INTO embeddings (
                   image_id, chunk_index, model_name, embedding, chunk_text
               ) VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    nearer,
                    0,
                    model.model_name,
                    np.array([1, 0], dtype=np.float32).tobytes(),
                    "best chunk",
                ),
                (
                    nearer,
                    1,
                    model.model_name,
                    np.array([0, 1], dtype=np.float32).tobytes(),
                    "worse chunk",
                ),
                (
                    farther,
                    0,
                    model.model_name,
                    np.array([0.8, 0.6], dtype=np.float32).tobytes(),
                    "near-ish",
                ),
                (
                    other_model,
                    0,
                    "another-model",
                    np.array([1, 0], dtype=np.float32).tobytes(),
                    "wrong model",
                ),
            ],
        )

        results = search_semantic(connection, "query", model)

        assert [result.image_id for result in results] == [nearer, farther]
        assert results[0].snippet == "best chunk"
        assert results[0].score == pytest.approx(1.0)
    finally:
        connection.close()


def test_cosine_similarity_and_dimension_validation() -> None:
    assert cosine_similarity(np.array([3, 0]), np.array([1, 0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([0, 0]), np.array([1, 0])) == 0.0
    with pytest.raises(ValueError, match="same number"):
        cosine_similarity(np.array([1, 0]), np.array([1, 2, 3]))


def test_rrf_combines_ranks_and_keeps_keyword_highlight(tmp_path: Path) -> None:
    keyword = [
        SearchResult(1, "one.png", 0.2, "[port] busy"),
        SearchResult(2, "two.png", 0.1, "keyword result"),
    ]
    semantic = [
        SearchResult(2, "two.png", 0.98, "meaning result"),
        SearchResult(1, "one.png", 0.95, "semantic result"),
    ]

    results = reciprocal_rank_fusion(keyword, semantic)

    assert {result.image_id for result in results} == {1, 2}
    assert "[port]" in next(result.snippet for result in results if result.image_id == 1)
    assert all(result.score > 0 for result in results)
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion(limit=0)


def test_hybrid_search_finds_exact_and_semantic_candidates(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "hybrid.sqlite3")
    model = EmbeddingModel(default_config(), engine_factory=FakeEngine)
    try:
        exact = _add_image(
            connection,
            "C:/shots/exact.png",
            "docker port already busy",
        )
        paraphrase = _add_image(
            connection,
            "C:/shots/paraphrase.png",
            "address already in use",
        )
        connection.executemany(
            """INSERT INTO embeddings (
                   image_id, chunk_index, model_name, embedding, chunk_text
               ) VALUES (?, 0, ?, ?, ?)""",
            [
                (
                    exact,
                    model.model_name,
                    np.array([0.8, 0.6], dtype=np.float32).tobytes(),
                    "docker port already busy",
                ),
                (
                    paraphrase,
                    model.model_name,
                    np.array([1, 0], dtype=np.float32).tobytes(),
                    "address already in use",
                ),
            ],
        )

        results = search_hybrid(connection, "docker port problem", model)

        assert {result.image_id for result in results} == {exact, paraphrase}
        assert results[0].image_id == exact
    finally:
        connection.close()
