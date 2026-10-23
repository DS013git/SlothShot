"""Local FastEmbed model loading, text chunking, and vector persistence."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from slothshot.config import AppConfig, load_config

logger = logging.getLogger("slothshot.embed")
DEFAULT_CHUNK_CHARACTERS = 1600


class EmbeddingEngine(Protocol):
    """Subset of the FastEmbed API used by SlothShot."""

    def embed(
        self, documents: Iterable[str], batch_size: int = 16, parallel: int = 1
    ) -> Iterable[np.ndarray]: ...

    def query_embed(self, query: str) -> Iterable[np.ndarray]: ...


class EmbeddingModel:
    """Lazily create a CPU-only FastEmbed model and reuse it for this process."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        engine_factory: Any | None = None,
    ) -> None:
        self.config = config or load_config()
        self.model_name = self.config.embedding_model
        self._engine_factory = engine_factory or self._make_engine
        self._engine: EmbeddingEngine | None = None

    def _make_engine(self) -> EmbeddingEngine:
        """Load cached files offline, or permit the documented first download."""
        from fastembed import TextEmbedding

        cached = embedding_model_cached(self.model_name, self.config.model_cache)
        if cached:
            logger.info(
                "Embedding model %s is cached locally; loading with network access disabled",
                self.model_name,
            )
        else:
            logger.info(
                "Embedding model %s is not cached; first use will download it to %s",
                self.model_name,
                self.config.model_cache,
            )
        # FastEmbed 0.9.0 accepts local_files_only through its constructor kwargs.
        # The cache probe above means it is true for every later, offline startup.
        return TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(self.config.model_cache),
            threads=2,
            cuda=False,
            local_files_only=cached,
        )

    @property
    def engine(self) -> EmbeddingEngine:
        """Load the ONNX model only when a search or indexing call needs it."""
        if self._engine is None:
            self._engine = self._engine_factory()
        return self._engine

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed document chunks in a small, single-worker batch."""
        if not texts:
            return []
        started = time.perf_counter()
        vectors = [
            np.asarray(item, dtype=np.float32)
            for item in self.engine.embed(texts, batch_size=16, parallel=1)
        ]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"FastEmbed returned {len(vectors)} vectors for {len(texts)} text chunks"
            )
        logger.info(
            "Embedded %d text chunk(s) in %.3f seconds",
            len(texts),
            time.perf_counter() - started,
        )
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one search query using FastEmbed's query-specific method."""
        started = time.perf_counter()
        vectors = iter(self.engine.query_embed(query))
        try:
            vector = np.asarray(next(vectors), dtype=np.float32)
        except StopIteration as exc:
            raise RuntimeError("FastEmbed returned no query embedding") from exc
        logger.info("Embedded search query in %.3f seconds", time.perf_counter() - started)
        return vector


def chunk_text(text: str, *, max_characters: int = DEFAULT_CHUNK_CHARACTERS) -> list[str]:
    """Split OCR text at word boundaries into bounded embedding chunks.

    FastEmbed's default BGE model truncates at 512 tokens. A 1,600-character
    bound leaves room for typical English OCR text while avoiding silent loss
    of the end of a long screenshot. Oversized individual tokens are sliced.
    """
    if max_characters < 1:
        raise ValueError("max_characters must be a positive integer")
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current:
            chunks.append(" ".join(current))
            current = []
            current_size = 0

    for word in words:
        # Keep even a single extremely long OCR token within the model bound.
        if len(word) > max_characters:
            flush()
            chunks.extend(
                word[offset : offset + max_characters]
                for offset in range(0, len(word), max_characters)
            )
            continue
        extra_size = len(word) + (1 if current else 0)
        if current_size + extra_size > max_characters:
            flush()
        current.append(word)
        current_size += len(word) + (1 if len(current) > 1 else 0)
    flush()
    return chunks


def embedding_model_cached(model_name: str, cache_dir: Path) -> bool:
    """Check for the configured model's ONNX file without triggering downloads."""
    from fastembed import TextEmbedding
    from huggingface_hub.file_download import repo_folder_name

    supported = {
        str(item["model"]).casefold(): item for item in TextEmbedding.list_supported_models()
    }
    description = supported.get(model_name.casefold())
    if description is None:
        return False
    hf_repo = description.get("sources", {}).get("hf")
    model_file = description.get("model_file")
    if not hf_repo or not model_file:
        return False
    try:
        repo_cache = Path(cache_dir) / repo_folder_name(repo_id=hf_repo, repo_type="model")
        snapshots = repo_cache / "snapshots"
        required_files = [
            model_file,
            "tokenizer.json",
            "tokenizer_config.json",
            *description.get("additional_files", []),
        ]
        return any(
            all((snapshot / filename).is_file() for filename in required_files)
            for snapshot in snapshots.iterdir()
        )
    except OSError:
        return False


def save_image_embeddings(
    connection: sqlite3.Connection,
    *,
    image_id: int,
    text: str,
    model: EmbeddingModel,
) -> int:
    """Save all chunks for an image unless this model already has vectors."""
    existing = connection.execute(
        "SELECT count(*) FROM embeddings WHERE image_id = ? AND model_name = ?",
        (image_id, model.model_name),
    ).fetchone()[0]
    if existing:
        return 0
    chunks = chunk_text(text)
    if not chunks:
        return 0

    vectors = model.embed_documents(chunks)
    with connection:
        connection.executemany(
            """INSERT OR IGNORE INTO embeddings (
                   image_id, chunk_index, model_name, embedding, chunk_text
               ) VALUES (?, ?, ?, ?, ?)""",
            [
                (image_id, index, model.model_name, vector.tobytes(), chunk)
                for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
    return len(vectors)


def reuse_embeddings_for_hash(
    connection: sqlite3.Connection,
    *,
    image_id: int,
    sha256: str,
) -> int:
    """Copy cached vectors from identical image bytes without model inference."""
    source = connection.execute(
        """SELECT id FROM images
           WHERE sha256 = ? AND id != ?
           ORDER BY id
           LIMIT 1""",
        (sha256, image_id),
    ).fetchone()
    if source is None:
        return 0
    with connection:
        cursor = connection.execute(
            """INSERT OR IGNORE INTO embeddings (
                   image_id, chunk_index, model_name, embedding, chunk_text
               )
               SELECT ?, chunk_index, model_name, embedding, chunk_text
               FROM embeddings
               WHERE image_id = ?""",
            (image_id, source["id"]),
        )
    return cursor.rowcount
