"""Dynamic HNSW update demo without compact storage.

This script reproduces the minimal scenario we used while debugging on-the-fly
recompute:

1. Build a non-compact HNSW index from the first few paragraphs of a text file.
2. Print the top results with `recompute_embeddings=True`.
3. Append additional paragraphs with :meth:`LeannBuilder.update_index`.
4. Run the same query again to show the newly inserted passages.

Run it with ``uv`` (optionally pointing LEANN_HNSW_LOG_PATH at a file to inspect
ZMQ activity)::

    LEANN_HNSW_LOG_PATH=embedding_fetch.log \
    uv run python examples/dynamic_update_no_recompute.py \
      --text-path data/PrideandPrejudice.txt \
      --index-path .leann/examples/pride.leann

The script defaults to ``sentence-transformers/all-MiniLM-L6-v2`` and uses
``is_recompute=True`` so Faiss will pull existing vectors on demand via the
ZMQ embedding server, while freshly added paragraphs are embedded locally just
like the initial build.
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from leann.api import LeannBuilder, LeannSearcher
from leann.registry import register_project_directory

DEFAULT_QUERY = "Who is credited as the author of the work?"


def load_paragraphs(text_path: Path) -> list[str]:
    text = text_path.read_text(encoding="utf-8", errors="ignore")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def run_search(index_path: Path, query: str, top_k: int) -> list:
    searcher = LeannSearcher(str(index_path))
    try:
        return searcher.search(
            query=query,
            top_k=top_k,
            recompute_embeddings=True,
            batch_size=16,
        )
    finally:
        searcher.cleanup()


def print_results(title: str, results: Iterable) -> None:
    print(f"\n=== {title} ===")
    res_list = list(results)
    print(f"results count: {len(res_list)}")
    print("passages:")
    if not res_list:
        print("  (no passages returned)")
    for res in res_list:
        snippet = res.text.replace("\n", " ")[:120]
        print(f"  - {res.id}: {snippet}... (score={res.score:.4f})")


def build_initial_index(
    index_path: Path,
    paragraphs: list[str],
    model_name: str,
    embedding_mode: str,
    is_recompute: bool,
) -> None:
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=model_name,
        embedding_mode=embedding_mode,
        is_compact=False,
        is_recompute=is_recompute,
    )
    for idx, passage in enumerate(paragraphs):
        builder.add_text(passage, metadata={"id": str(idx)})
    builder.build_index(str(index_path))


def update_index(
    index_path: Path,
    start_id: int,
    paragraphs: list[str],
    model_name: str,
    embedding_mode: str,
    is_recompute: bool,
) -> None:
    updater = LeannBuilder(
        backend_name="hnsw",
        embedding_model=model_name,
        embedding_mode=embedding_mode,
        is_compact=False,
        is_recompute=is_recompute,
    )
    for offset, passage in enumerate(paragraphs, start=start_id):
        updater.add_text(passage, metadata={"id": str(offset)})
    updater.update_index(str(index_path))


def ensure_index_dir(index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-path",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "PrideandPrejudice.txt",
        help="Path to the source text file (default: data/PrideandPrejudice.txt)",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(".leann/examples/pride.leann"),
        help="Destination index path (default: .leann/examples/pride.leann)",
    )
    parser.add_argument(
        "--initial-count",
        type=int,
        default=1,
        help="Number of paragraphs for the initial build (default: 1)",
    )
    parser.add_argument(
        "--update-count",
        type=int,
        default=3,
        help="Number of paragraphs to append during update (default: 3)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of results to show after update (default: 4)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Query to run before/after the update",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "mlx", "ollama"],
        help="Embedding backend mode",
    )
    args = parser.parse_args()

    paragraphs = load_paragraphs(args.text_path)
    if len(paragraphs) < args.initial_count + args.update_count:
        raise ValueError("Not enough paragraphs in the source text for the requested counts.")

    ensure_index_dir(args.index_path)
    register_project_directory(Path.cwd())

    initial = paragraphs[: args.initial_count]
    to_add = paragraphs[args.initial_count : args.initial_count + args.update_count]

    print("Building initial index...")
    build_initial_index(
        args.index_path,
        initial,
        args.embedding_model,
        args.embedding_mode,
        is_recompute=True,
    )

    index_file = args.index_path.parent / f"{args.index_path.stem}.index"
    initial_size = index_file.stat().st_size if index_file.exists() else 0

    before_results = run_search(args.index_path, args.query, args.top_k)
    print_results("initial search", before_results)

    print("\nUpdating index with additional passages...")
    update_index(
        args.index_path,
        start_id=args.initial_count,
        paragraphs=to_add,
        model_name=args.embedding_model,
        embedding_mode=args.embedding_mode,
        is_recompute=True,
    )

    after_results = run_search(args.index_path, args.query, args.top_k)
    print_results("after update", after_results)

    updated_size = index_file.stat().st_size if index_file.exists() else 0
    print(
        f"\nIndex file size change: {initial_size} -> {updated_size} bytes"
        f" (Δ {updated_size - initial_size})"
    )

    meta_path = args.index_path.parent / f"{args.index_path.name}.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print("\n--- Metadata snapshot ---")
        print(json.dumps({k: meta.get(k) for k in ("is_compact", "is_pruned")}, indent=2))


if __name__ == "__main__":
    main()
