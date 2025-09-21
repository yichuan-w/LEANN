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
    uv run -m examples.dynamic_update_no_recompute \
      --index-path .leann/examples/leann-demo.leann

By default the script builds an index from ``data/2501.14312v1 (1).pdf`` and
then updates it with LEANN-related material from ``data/2506.08276v1.pdf``.
It issues the query "What's LEANN?" before and after the update to show how the
new passages become immediately searchable. The script uses the
``sentence-transformers/all-MiniLM-L6-v2`` model with ``is_recompute=True`` so
Faiss pulls existing vectors on demand via the ZMQ embedding server, while
freshly added passages are embedded locally just like the initial build.

To make storage comparisons easy, the script can also build a matching
``is_recompute=False`` baseline (enabled by default) and report the index size
delta after the update. Disable the baseline run with
``--skip-compare-no-recompute`` if you only need the recompute flow.
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from leann.api import LeannBuilder, LeannSearcher
from leann.registry import register_project_directory

from apps.chunking import create_text_chunks

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUERY = "What's LEANN?"
DEFAULT_INITIAL_FILES = [REPO_ROOT / "data" / "2501.14312v1 (1).pdf"]
DEFAULT_UPDATE_FILES = [REPO_ROOT / "data" / "2506.08276v1.pdf"]


def load_chunks_from_files(paths: list[Path]) -> list[str]:
    from llama_index.core import SimpleDirectoryReader

    documents = []
    for path in paths:
        p = path.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input path not found: {p}")
        if p.is_dir():
            reader = SimpleDirectoryReader(str(p), recursive=False)
            documents.extend(reader.load_data(show_progress=True))
        else:
            reader = SimpleDirectoryReader(input_files=[str(p)])
            documents.extend(reader.load_data(show_progress=True))

    if not documents:
        return []

    chunks = create_text_chunks(
        documents,
        chunk_size=512,
        chunk_overlap=128,
        use_ast_chunking=False,
    )
    return [c for c in chunks if isinstance(c, str) and c.strip()]


def run_search(index_path: Path, query: str, top_k: int, *, recompute_embeddings: bool) -> list:
    searcher = LeannSearcher(str(index_path))
    try:
        return searcher.search(
            query=query,
            top_k=top_k,
            recompute_embeddings=recompute_embeddings,
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


def cleanup_index_files(index_path: Path) -> None:
    """Remove leftover index artifacts for a clean rebuild."""

    parent = index_path.parent
    if not parent.exists():
        return
    stem = index_path.stem
    for file in parent.glob(f"{stem}*"):
        if file.is_file():
            file.unlink()


def index_file_size(index_path: Path) -> int:
    """Return the size of the primary .index file for the given index path."""

    index_file = index_path.parent / f"{index_path.stem}.index"
    return index_file.stat().st_size if index_file.exists() else 0


def load_metadata_snapshot(index_path: Path) -> dict[str, Any] | None:
    meta_path = index_path.parent / f"{index_path.name}.meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None


def run_workflow(
    *,
    label: str,
    index_path: Path,
    initial_paragraphs: list[str],
    update_paragraphs: list[str],
    model_name: str,
    embedding_mode: str,
    is_recompute: bool,
    query: str,
    top_k: int,
) -> dict[str, Any]:
    prefix = f"[{label}] " if label else ""

    ensure_index_dir(index_path)
    cleanup_index_files(index_path)

    print(f"{prefix}Building initial index...")
    build_initial_index(
        index_path,
        initial_paragraphs,
        model_name,
        embedding_mode,
        is_recompute=is_recompute,
    )

    initial_size = index_file_size(index_path)
    before_results = run_search(
        index_path,
        query,
        top_k,
        recompute_embeddings=is_recompute,
    )

    print(f"\n{prefix}Updating index with additional passages...")
    update_index(
        index_path,
        start_id=len(initial_paragraphs),
        paragraphs=update_paragraphs,
        model_name=model_name,
        embedding_mode=embedding_mode,
        is_recompute=is_recompute,
    )

    after_results = run_search(
        index_path,
        query,
        top_k,
        recompute_embeddings=is_recompute,
    )
    updated_size = index_file_size(index_path)

    return {
        "initial_size": initial_size,
        "updated_size": updated_size,
        "delta": updated_size - initial_size,
        "before_results": before_results,
        "after_results": after_results,
        "metadata": load_metadata_snapshot(index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-files",
        type=Path,
        nargs="+",
        default=DEFAULT_INITIAL_FILES,
        help="Initial document files (PDF/TXT) used to build the base index",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(".leann/examples/leann-demo.leann"),
        help="Destination index path (default: .leann/examples/leann-demo.leann)",
    )
    parser.add_argument(
        "--initial-count",
        type=int,
        default=8,
        help="Number of chunks to use from the initial documents (default: 8)",
    )
    parser.add_argument(
        "--update-files",
        type=Path,
        nargs="*",
        default=DEFAULT_UPDATE_FILES,
        help="Additional documents to add during update (PDF/TXT)",
    )
    parser.add_argument(
        "--update-count",
        type=int,
        default=4,
        help="Number of chunks to append from update documents (default: 4)",
    )
    parser.add_argument(
        "--update-text",
        type=str,
        default=(
            "LEANN (Lightweight Embedding ANN) is an indexing toolkit focused on "
            "recompute-aware HNSW graphs, allowing embeddings to be regenerated "
            "on demand to keep disk usage minimal."
        ),
        help="Fallback text to append if --update-files is omitted",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of results to show for each search (default: 4)",
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
    parser.add_argument(
        "--compare-no-recompute",
        dest="compare_no_recompute",
        action="store_true",
        help="Also run a baseline with is_recompute=False and report its index growth.",
    )
    parser.add_argument(
        "--skip-compare-no-recompute",
        dest="compare_no_recompute",
        action="store_false",
        help="Skip building the no-recompute baseline.",
    )
    parser.set_defaults(compare_no_recompute=True)
    args = parser.parse_args()

    ensure_index_dir(args.index_path)
    register_project_directory(REPO_ROOT)

    initial_chunks = load_chunks_from_files(list(args.initial_files))
    if not initial_chunks:
        raise ValueError("No text chunks extracted from the initial files.")

    initial = initial_chunks[: args.initial_count]
    if not initial:
        raise ValueError("Initial chunk set is empty after applying --initial-count.")

    if args.update_files:
        update_chunks = load_chunks_from_files(list(args.update_files))
        if not update_chunks:
            raise ValueError("No text chunks extracted from the update files.")
        to_add = update_chunks[: args.update_count]
    else:
        if not args.update_text:
            raise ValueError("Provide --update-files or --update-text for the update step.")
        to_add = [args.update_text]
    if not to_add:
        raise ValueError("Update chunk set is empty after applying --update-count.")

    recompute_stats = run_workflow(
        label="recompute",
        index_path=args.index_path,
        initial_paragraphs=initial,
        update_paragraphs=to_add,
        model_name=args.embedding_model,
        embedding_mode=args.embedding_mode,
        is_recompute=True,
        query=args.query,
        top_k=args.top_k,
    )

    print_results("initial search", recompute_stats["before_results"])
    print_results("after update", recompute_stats["after_results"])
    print(
        f"\n[recompute] Index file size change: {recompute_stats['initial_size']} -> {recompute_stats['updated_size']} bytes"
        f" (Δ {recompute_stats['delta']})"
    )

    if recompute_stats["metadata"]:
        meta_view = {k: recompute_stats["metadata"].get(k) for k in ("is_compact", "is_pruned")}
        print("[recompute] metadata snapshot:")
        print(json.dumps(meta_view, indent=2))

    if args.compare_no_recompute:
        baseline_path = (
            args.index_path.parent / f"{args.index_path.stem}-norecompute{args.index_path.suffix}"
        )
        baseline_stats = run_workflow(
            label="no-recompute",
            index_path=baseline_path,
            initial_paragraphs=initial,
            update_paragraphs=to_add,
            model_name=args.embedding_model,
            embedding_mode=args.embedding_mode,
            is_recompute=False,
            query=args.query,
            top_k=args.top_k,
        )

        print(
            f"\n[no-recompute] Index file size change: {baseline_stats['initial_size']} -> {baseline_stats['updated_size']} bytes"
            f" (Δ {baseline_stats['delta']})"
        )

        after_texts = [res.text for res in recompute_stats["after_results"]]
        baseline_after_texts = [res.text for res in baseline_stats["after_results"]]
        if after_texts == baseline_after_texts:
            print(
                "[no-recompute] Search results match recompute baseline; see above for the shared output."
            )
        else:
            print("[no-recompute] WARNING: search results differ from recompute baseline.")

        if baseline_stats["metadata"]:
            meta_view = {k: baseline_stats["metadata"].get(k) for k in ("is_compact", "is_pruned")}
            print("[no-recompute] metadata snapshot:")
            print(json.dumps(meta_view, indent=2))


if __name__ == "__main__":
    main()
