"""Measure local OCR, embedding, indexing memory, and search scale."""

from __future__ import annotations

import argparse
import logging
import math
import platform
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from statistics import median

import psutil
from make_sample_data import generate_samples

from slothshot.config import default_config
from slothshot.db import connect_database
from slothshot.embed import EmbeddingModel, embedding_model_cached
from slothshot.indexer import scan_folder
from slothshot.ocr import OCRReader
from slothshot.search import SearchResult, search_hybrid, search_keyword, search_semantic

logger = logging.getLogger("slothshot.bench")


class TimedReader:
    """Collect the OCR reader's own per-image timings during a real scan."""

    def __init__(self, reader: OCRReader) -> None:
        self.reader = reader
        self.seconds: list[float] = []

    def read(self, image_path: Path) -> object:
        result = self.reader.read(image_path)
        self.seconds.append(float(result.elapsed_seconds))
        return result


class PeakRss:
    """Sample current process RSS in a helper thread while indexing runs."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_bytes = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __enter__(self) -> PeakRss:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def _percentile_95(values: list[float]) -> float:
    """Return the nearest-rank 95th percentile for a non-empty sample."""
    if not values:
        return 0.0
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def _seed_scaled_database(
    path: Path,
    count: int,
    source_rows: list[sqlite3.Row],
    model_name: str,
) -> int:
    """Build a scaled search-only corpus from already indexed synthetic data."""
    if not source_rows:
        raise RuntimeError("The indexing pass produced no OCR text or embedding rows")
    connection = connect_database(path)
    now = "2026-10-21T00:00:00+00:00"
    try:
        with connection:
            for offset in range(count):
                source = source_rows[offset % len(source_rows)]
                cursor = connection.execute(
                    """INSERT INTO images (
                           path, sha256, width, height, ocr_text, avg_confidence,
                           file_mtime, indexed_at, status
                       ) VALUES (?, ?, 1000, 620, ?, 0.9, 1.0, ?, 'indexed')""",
                    (
                        f"synthetic://scaled/{offset:05d}.png",
                        f"scale-{count}-{offset}",
                        source["ocr_text"],
                        now,
                    ),
                )
                connection.execute(
                    """INSERT INTO embeddings (
                           image_id, chunk_index, model_name, embedding, chunk_text
                       ) VALUES (?, 0, ?, ?, ?)""",
                    (cursor.lastrowid, model_name, source["embedding"], source["chunk_text"]),
                )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return path.stat().st_size
    finally:
        connection.close()


def _latency(
    search: Callable[..., list[SearchResult]],
    connection: sqlite3.Connection,
    query: str,
    model: EmbeddingModel,
    repeats: int,
) -> tuple[float, float]:
    """Warm a search mode and report median and nearest-rank p95 milliseconds."""
    if search is search_keyword:
        search(connection, query, limit=20)
    else:
        search(connection, query, model, limit=20)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        if search is search_keyword:
            search(connection, query, limit=20)
        else:
            search(connection, query, model, limit=20)
        samples.append((time.perf_counter() - start) * 1000)
    return median(samples), _percentile_95(samples)


def run_benchmark(output_dir: Path, *, count: int = 100, repeats: int = 5) -> dict[str, object]:
    """Run real model inference on generated screenshots and scaled DB searches."""
    if count < 100:
        raise ValueError("Use at least 100 screenshots for this benchmark")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_path = generate_samples(output_dir / "screenshots", count)
    config = default_config()
    cached = embedding_model_cached(config.embedding_model, config.model_cache)
    logger.info(
        "Embedding cache check: model=%s cached=%s path=%s",
        config.embedding_model,
        cached,
        config.model_cache,
    )
    if not cached:
        logger.warning("First-use embedding will download once; later runs can work offline")

    db_path = output_dir / "indexed.sqlite3"
    config = replace(config, data_dir=output_dir, model_cache=config.model_cache)
    connection = connect_database(db_path)
    reader = TimedReader(OCRReader(config))
    model = EmbeddingModel(config)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    start = time.perf_counter()
    with PeakRss() as memory:
        summary = scan_folder(output_dir / "screenshots", connection, reader, model)
    indexing_seconds = time.perf_counter() - start
    peak_rss = memory.peak_bytes

    texts = [
        str(row["ocr_text"])
        for row in connection.execute(
            "SELECT ocr_text FROM images WHERE status = 'indexed' AND ocr_text != '' ORDER BY id"
        )
    ]
    embed_start = time.perf_counter()
    model.embed_documents(texts)
    embedding_seconds = time.perf_counter() - embed_start
    source_rows = connection.execute(
        """SELECT i.ocr_text, e.embedding, e.chunk_text
           FROM images AS i JOIN embeddings AS e ON e.image_id = i.id
           WHERE i.status = 'indexed' ORDER BY i.id LIMIT ?""",
        (count,),
    ).fetchall()
    connection.close()

    scale_results: list[dict[str, object]] = []
    for size in (100, 1000, 5000):
        scaled_path = output_dir / f"search-{size}.sqlite3"
        size_bytes = _seed_scaled_database(scaled_path, size, source_rows, config.embedding_model)
        search_connection = connect_database(scaled_path)
        try:
            modes = {
                "keyword": search_keyword,
                "semantic": search_semantic,
                "hybrid": search_hybrid,
            }
            latencies = {
                name: _latency(search, search_connection, "docker port problem", model, repeats)
                for name, search in modes.items()
            }
        finally:
            search_connection.close()
        scale_results.append(
            {
                "images": size,
                "db_mib": size_bytes / (1024 * 1024),
                "search_ms": latencies,
            }
        )

    specs = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "not reported by platform module",
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gib": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    return {
        "machine": specs,
        "samples": count,
        "ground_truth": str(truth_path),
        "indexed": {
            "summary": summary,
            "elapsed_seconds": indexing_seconds,
            "baseline_rss_mib": baseline_rss / (1024**2),
            "peak_rss_mib": peak_rss / (1024**2),
            "ocr_median_seconds": median(reader.seconds) if reader.seconds else 0.0,
            "ocr_p95_seconds": _percentile_95(reader.seconds),
            "embedding_seconds": embedding_seconds,
            "embedding_documents": len(texts),
            "db_mib": db_path.stat().st_size / (1024 * 1024),
        },
        "scales": scale_results,
        "repeats": repeats,
    }


def main() -> None:
    """Run the benchmark in an isolated temporary workspace by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Keep benchmark files in this directory")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.output:
        result = run_benchmark(args.output, count=args.count, repeats=args.repeats)
        _print_results(result)
    else:
        with tempfile.TemporaryDirectory(prefix="slothshot-bench-") as temporary:
            result = run_benchmark(Path(temporary), count=args.count, repeats=args.repeats)
            _print_results(result)


def _print_results(result: dict[str, object]) -> None:
    """Print measured values with their units and distinguish indexed from scaled data."""
    indexed = result["indexed"]
    print(f"Machine: {result['machine']}")
    print(f"Models: OCR packaged locally; embedding model: {default_config().embedding_model}")
    print(f"Model cache: {default_config().model_cache}")
    print(f"Indexed generated images ({result['samples']} regular + tricky): {indexed['summary']}")
    print(f"Index wall time: {indexed['elapsed_seconds']:.2f} s")
    print(
        f"OCR median {indexed['ocr_median_seconds']:.3f} s/image; "
        f"p95 {indexed['ocr_p95_seconds']:.3f} s/image"
    )
    print(
        f"Embedding {indexed['embedding_documents']} documents: "
        f"{indexed['embedding_seconds']:.2f} s (warm model)"
    )
    print(
        f"Index peak RSS: {indexed['peak_rss_mib']:.1f} MiB "
        f"(baseline {indexed['baseline_rss_mib']:.1f} MiB)"
    )
    print(f"Indexed DB: {indexed['db_mib']:.2f} MiB")
    print(f"Search latency (median / p95 ms, {result['repeats']} repeats):")
    print(f"{'images':>7} {'DB MiB':>10} {'keyword':>19} {'semantic':>19} {'hybrid':>19}")

    def format_latency(pair: tuple[float, float]) -> str:
        return f"{pair[0]:.2f} / {pair[1]:.2f}"

    for scale in result["scales"]:
        pairs = scale["search_ms"]
        print(
            f"{scale['images']:>7} {scale['db_mib']:>10.2f} "
            f"{format_latency(pairs['keyword']):>19} "
            f"{format_latency(pairs['semantic']):>19} "
            f"{format_latency(pairs['hybrid']):>19}"
        )
    print("Search scale databases reuse OCR text and vectors from the indexed synthetic corpus.")


if __name__ == "__main__":
    main()
