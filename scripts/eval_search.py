"""Compare keyword, semantic, and hybrid retrieval on a YAML truth set."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from slothshot.config import load_config
from slothshot.db import connect_database
from slothshot.embed import EmbeddingModel, embedding_model_cached
from slothshot.evaluation import load_ground_truth, retrieval_metrics
from slothshot.search import search_hybrid, search_keyword, search_semantic

logger = logging.getLogger("slothshot.eval")


def main() -> None:
    """Evaluate search modes against expected screenshot file names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth", type=Path, required=True, help="YAML query-to-image mapping"
    )
    parser.add_argument(
        "--db", type=Path, help="SQLite database (defaults to configured SlothShot DB)"
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.limit < 5:
        parser.error("--limit must be at least 5 so top-5 metrics can be computed")

    try:
        truth = load_ground_truth(args.ground_truth)
    except ValueError as exc:
        parser.error(str(exc))
    config = load_config()
    database_path = args.db or config.database_path
    if not database_path.is_file():
        parser.error(f"Database does not exist: {database_path}")
    cached = embedding_model_cached(config.embedding_model, config.model_cache)
    logger.info("Embedding model cache: %s", "available locally" if cached else "not cached")
    model = EmbeddingModel(config)
    connection = connect_database(database_path)
    try:
        modes = {
            "keyword": search_keyword,
            "semantic": search_semantic,
            "hybrid": search_hybrid,
        }
        metrics_by_mode: dict[str, dict[str, float]] = {}
        # mode -> [(query, rank of first correct result or None, top result name)]
        misses: dict[str, list[tuple[str, int | None, str]]] = {}
        for mode_name, search in modes.items():
            rankings: list[list[str]] = []
            expected: list[set[str]] = []
            for query, filenames in truth.items():
                results = (
                    search(connection, query, model, limit=args.limit)
                    if mode_name != "keyword"
                    else search(connection, query, limit=args.limit)
                )
                rankings.append([result.path for result in results])
                expected.append(filenames)
            metrics_by_mode[mode_name] = retrieval_metrics(rankings, expected)
            misses[mode_name] = []
            for query, ranking, wanted in zip(truth, rankings, expected, strict=True):
                names = [Path(path).name.casefold() for path in ranking]
                rank = next((i for i, name in enumerate(names, 1) if name in wanted), None)
                if rank != 1:
                    top = names[0] if names else "(no results)"
                    misses[mode_name].append((query, rank, top))
    finally:
        connection.close()

    print(f"Evaluation queries: {len(truth)}  |  database: {database_path}")
    print(f"{'mode':<10} {'top-1 hit':>10} {'top-3 hit':>10} {'top-5 hit':>10} {'MRR':>8}")
    for mode, metrics in metrics_by_mode.items():
        print(
            f"{mode:<10} {metrics['hit@1']:>10.3f} {metrics['hit@3']:>10.3f} "
            f"{metrics['hit@5']:>10.3f} {metrics['mrr']:>8.3f}"
        )
    for mode, mode_misses in misses.items():
        for query, rank, top in mode_misses:
            found = f"first correct at rank {rank}" if rank else "no correct result"
            print(f"miss  {mode:<9} {query!r}: {found}; top result {top}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
