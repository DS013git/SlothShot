"""Safe keyword search over the local FTS5 index."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import numpy as np

from slothshot.embed import EmbeddingModel


@dataclass(frozen=True)
class SearchResult:
    """One ranked screenshot match returned by a search mode."""

    image_id: int
    path: str
    score: float
    snippet: str


def sanitize_query(query: str) -> tuple[str, ...]:
    """Keep only searchable Unicode words, safely removing FTS operators.

    The tokenizer is deliberately small: terms are always quoted in the FTS
    expression, so user input can never become a column filter or FTS operator.
    """
    return tuple(re.findall(r"[^\W_]+", query, flags=re.UNICODE)[:16])


def _build_fts_query(terms: tuple[str, ...], *, operator: str) -> str:
    """Quote terms and add FTS5's suffix wildcard for prefix matching."""
    if operator not in {"AND", "OR"}:
        raise ValueError("FTS term operator must be AND or OR")
    quoted = [f'"{term}"*' for term in terms]
    return f" {operator} ".join(quoted)


def search_keyword(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
) -> list[SearchResult]:
    """Search exact terms and prefixes with BM25, falling back to OR if needed.

    AND gives short, precise queries their expected behavior. If that has no
    hits, an OR retry lets longer natural-language queries still return useful
    candidates instead of an empty page. Lower BM25 scores rank first.
    """
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    terms = sanitize_query(query)
    if not terms:
        return []

    results = _run_search(connection, _build_fts_query(terms, operator="AND"), limit)
    if results or len(terms) == 1:
        return results
    return _run_search(connection, _build_fts_query(terms, operator="OR"), limit)


def _run_search(
    connection: sqlite3.Connection,
    fts_query: str,
    limit: int,
) -> list[SearchResult]:
    """Run one parameterized FTS query and convert rows to plain results."""
    rows = connection.execute(
        """SELECT images.id, images.path, bm25(images_fts) AS score,
                  snippet(images_fts, 1, '[', ']', ' … ', 18) AS snippet
           FROM images_fts
           JOIN images ON images.id = images_fts.rowid
           WHERE images_fts MATCH ?
           ORDER BY score ASC, images.id ASC
           LIMIT ?""",
        (fts_query, limit),
    ).fetchall()
    return [
        SearchResult(
            image_id=int(row["id"]),
            path=str(row["path"]),
            # SQLite's BM25 values are negative; invert the sign so that
            # higher scores consistently mean a stronger match across modes.
            score=-float(row["score"]),
            snippet=str(row["snippet"]),
        )
        for row in rows
    ]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity, safely handling empty and zero-length vectors."""
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("Embedding vectors must have the same number of dimensions")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def search_semantic(
    connection: sqlite3.Connection,
    query: str,
    model: EmbeddingModel,
    *,
    limit: int = 20,
) -> list[SearchResult]:
    """Find nearest OCR chunks by exact cosine similarity on local vectors."""
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    if not query.strip():
        return []
    vector_count = connection.execute(
        "SELECT count(*) FROM embeddings WHERE model_name = ?", (model.model_name,)
    ).fetchone()[0]
    if not vector_count:
        return []
    query_vector = model.embed_query(query).reshape(-1)
    rows = connection.execute(
        """SELECT e.image_id, e.embedding, e.chunk_text, i.path
           FROM embeddings AS e
           JOIN images AS i ON i.id = e.image_id
           WHERE e.model_name = ? AND i.status = 'indexed'""",
        (model.model_name,),
    )
    best_by_image: dict[int, SearchResult] = {}
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.size != query_vector.size:
            continue
        score = cosine_similarity(query_vector, vector)
        image_id = int(row["image_id"])
        current = best_by_image.get(image_id)
        if current is None or score > current.score:
            best_by_image[image_id] = SearchResult(
                image_id=image_id,
                path=str(row["path"]),
                score=score,
                snippet=str(row["chunk_text"]),
            )
    return sorted(best_by_image.values(), key=lambda item: (-item.score, item.image_id))[:limit]


def reciprocal_rank_fusion(
    *rankings: list[SearchResult],
    limit: int = 20,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Merge ranked result lists using Reciprocal Rank Fusion (RRF)."""
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    if rank_constant < 0:
        raise ValueError("rank_constant must not be negative")
    scores: dict[int, float] = {}
    details: dict[int, SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.image_id] = scores.get(result.image_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
            existing = details.get(result.image_id)
            # Keep the keyword-highlighted snippet when available; otherwise
            # the text from the closest semantic chunk is the best context.
            if existing is None or ("[" not in existing.snippet and "[" in result.snippet):
                details[result.image_id] = result
    ordered_ids = sorted(scores, key=lambda image_id: (-scores[image_id], image_id))[:limit]
    return [
        SearchResult(
            image_id=image_id,
            path=details[image_id].path,
            score=scores[image_id],
            snippet=details[image_id].snippet,
        )
        for image_id in ordered_ids
    ]


def search_hybrid(
    connection: sqlite3.Connection,
    query: str,
    model: EmbeddingModel,
    *,
    limit: int = 20,
) -> list[SearchResult]:
    """Combine FTS5 and semantic rankings with reciprocal rank fusion."""
    keyword_results = search_keyword(connection, query, limit=limit)
    semantic_results = search_semantic(connection, query, model, limit=limit)
    return reciprocal_rank_fusion(keyword_results, semantic_results, limit=limit)
