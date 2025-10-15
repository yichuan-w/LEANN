"""Benchmark incremental HNSW add() under different RNG pruning modes with real
embedding recomputation.

This script clones the structure of ``examples/dynamic_update_no_recompute.py``
so that we build a non-compact ``is_recompute=True`` index, spin up the
standard HNSW embedding server, and measure how long incremental ``add`` takes
when RNG pruning is fully enabled vs. partially/fully disabled.

Example usage (will download the sentence-transformers model on first run)::

    uv run -m examples.bench_hnsw_rng_recompute \
        --index-path .leann/bench/leann-demo.leann \
        --runs 1

You can tweak the input documents with ``--initial-files`` / ``--update-files``
if you want a larger or different workload, and change the embedding model via
``--model-name``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from leann.api import LeannBuilder

if os.environ.get("LEANN_FORCE_CPU", "").lower() in ("1", "true", "yes"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from leann.embedding_compute import compute_embeddings
from leann.embedding_server_manager import EmbeddingServerManager
from leann.registry import register_project_directory

from apps.chunking import create_text_chunks

from leann_backend_hnsw import faiss  # type: ignore
from leann_backend_hnsw.convert_to_csr import prune_hnsw_embeddings_inplace

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INITIAL_FILES = [
    REPO_ROOT / "data" / "2501.14312v1 (1).pdf",
    REPO_ROOT / "data" / "huawei_pangu.md",
]
DEFAULT_UPDATE_FILES = [REPO_ROOT / "data" / "2506.08276v1.pdf"]


def load_chunks_from_files(paths: list[Path], limit: int | None = None) -> list[str]:
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
    cleaned = [c for c in chunks if isinstance(c, str) and c.strip()]
    if limit is not None:
        cleaned = cleaned[:limit]
    return cleaned


def ensure_index_dir(index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)


def cleanup_index_files(index_path: Path) -> None:
    parent = index_path.parent
    if not parent.exists():
        return
    stem = index_path.stem
    for file in parent.glob(f"{stem}*"):
        if file.is_file():
            file.unlink()


def build_initial_index(
    index_path: Path,
    paragraphs: list[str],
    model_name: str,
    embedding_mode: str,
    distance_metric: str,
    ef_construction: int,
) -> None:
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=model_name,
        embedding_mode=embedding_mode,
        is_compact=False,
        is_recompute=True,
        distance_metric=distance_metric,
        backend_kwargs={
            "distance_metric": distance_metric,
            "is_compact": False,
            "is_recompute": True,
            "efConstruction": ef_construction,
        },
    )
    for idx, passage in enumerate(paragraphs):
        builder.add_text(passage, metadata={"id": str(idx)})
    builder.build_index(str(index_path))


def prepare_new_chunks(paragraphs: list[str]) -> list[dict[str, Any]]:
    return [{"text": text, "metadata": {}} for text in paragraphs]


def benchmark_update_with_mode(
    index_path: Path,
    new_chunks: list[dict[str, Any]],
    model_name: str,
    embedding_mode: str,
    distance_metric: str,
    disable_forward_rng: bool,
    disable_reverse_rng: bool,
    server_port: int,
    add_timeout: int,
) -> tuple[float, float]:
    meta_path = index_path.parent / f"{index_path.name}.meta.json"
    passages_file = index_path.parent / f"{index_path.name}.passages.jsonl"
    offset_file = index_path.parent / f"{index_path.name}.passages.idx"
    index_file = index_path.parent / f"{index_path.stem}.index"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    with open(offset_file, "rb") as f:
        offset_map: dict[str, int] = pickle.load(f)
    existing_ids = set(offset_map.keys())

    valid_chunks: list[dict[str, Any]] = []
    for chunk in new_chunks:
        text = chunk.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        metadata = chunk.setdefault("metadata", {})
        passage_id = chunk.get("id") or metadata.get("id")
        if passage_id and passage_id in existing_ids:
            raise ValueError(f"Passage ID '{passage_id}' already exists in the index.")
        valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError("No valid chunks to append.")

    texts_to_embed = [chunk["text"] for chunk in valid_chunks]
    embeddings = compute_embeddings(
        texts_to_embed,
        model_name,
        mode=embedding_mode,
        is_build=False,
        batch_size=16,
    )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if distance_metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

    index = faiss.read_index(str(index_file))
    index.is_recompute = True
    if getattr(index, "storage", None) is None:
        if index.metric_type == faiss.METRIC_INNER_PRODUCT:
            storage_index = faiss.IndexFlatIP(index.d)
        else:
            storage_index = faiss.IndexFlatL2(index.d)
        index.storage = storage_index
        index.own_fields = True
        try:
            storage_index.ntotal = index.ntotal
        except AttributeError:
            pass
    try:
        index.hnsw.set_disable_rng_during_add(disable_forward_rng)
        index.hnsw.set_disable_reverse_prune(disable_reverse_rng)
    except AttributeError:
        pass

    applied_forward = getattr(index.hnsw, "disable_rng_during_add", None)
    applied_reverse = getattr(index.hnsw, "disable_reverse_prune", None)
    logger.info(
        "HNSW RNG config -> requested forward=%s, reverse=%s | applied forward=%s, reverse=%s",
        disable_forward_rng,
        disable_reverse_rng,
        applied_forward,
        applied_reverse,
    )

    base_id = index.ntotal
    for offset, chunk in enumerate(valid_chunks):
        new_id = str(base_id + offset)
        chunk.setdefault("metadata", {})["id"] = new_id
        chunk["id"] = new_id

    rollback_size = passages_file.stat().st_size if passages_file.exists() else 0
    offset_map_backup = offset_map.copy()

    start_time = time.time()

    try:
        with open(passages_file, "a", encoding="utf-8") as f:
            for chunk in valid_chunks:
                offset = f.tell()
                json.dump(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
                offset_map[chunk["id"]] = offset

        with open(offset_file, "wb") as f:
            pickle.dump(offset_map, f)

        server_manager = EmbeddingServerManager(
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
        )
        server_started, actual_port = server_manager.start_server(
            port=server_port,
            model_name=model_name,
            embedding_mode=embedding_mode,
            passages_file=str(meta_path),
            distance_metric=distance_metric,
        )
        if not server_started:
            raise RuntimeError("Failed to start embedding server.")

        if hasattr(index.hnsw, "set_zmq_port"):
            index.hnsw.set_zmq_port(actual_port)
        elif hasattr(index, "set_zmq_port"):
            index.set_zmq_port(actual_port)

        total_start = time.time()
        add_elapsed = 0.0

        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("incremental add timed out")

            if add_timeout > 0:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(add_timeout)

            add_start = time.time()
            for i in range(embeddings.shape[0]):
                index.add(1, faiss.swig_ptr(embeddings[i : i + 1]))
            add_elapsed = time.time() - add_start
            if add_timeout > 0:
                signal.alarm(0)
            faiss.write_index(index, str(index_file))
        finally:
            server_manager.stop_server()

    except TimeoutError:
        raise
    except Exception:
        if passages_file.exists():
            with open(passages_file, "rb+") as f:
                f.truncate(rollback_size)
        with open(offset_file, "wb") as f:
            pickle.dump(offset_map_backup, f)
        raise

    prune_hnsw_embeddings_inplace(str(index_file))

    meta["total_passages"] = len(offset_map)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Reset toggles so the index on disk returns to baseline behaviour.
    try:
        index.hnsw.set_disable_rng_during_add(False)
        index.hnsw.set_disable_reverse_prune(False)
    except AttributeError:
        pass
    faiss.write_index(index, str(index_file))

    total_elapsed = time.time() - total_start

    return total_elapsed, add_elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(".leann/bench/leann-demo.leann"),
        help="Output index base path (without extension).",
    )
    parser.add_argument(
        "--initial-files",
        nargs="*",
        type=Path,
        default=DEFAULT_INITIAL_FILES,
        help="Files used to build the initial index.",
    )
    parser.add_argument(
        "--update-files",
        nargs="*",
        type=Path,
        default=DEFAULT_UPDATE_FILES,
        help="Files appended during the benchmark.",
    )
    parser.add_argument("--runs", type=int, default=1, help="How many times to repeat each scenario.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used for build/update.",
    )
    parser.add_argument(
        "--embedding-mode",
        default="sentence-transformers",
        help="Embedding mode passed to LeannBuilder/embedding server.",
    )
    parser.add_argument(
        "--distance-metric",
        default="mips",
        choices=["mips", "l2", "cosine"],
        help="Distance metric for HNSW backend.",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="efConstruction setting for initial build.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=5557,
        help="Port for the real embedding server.",
    )
    parser.add_argument(
        "--max-initial",
        type=int,
        default=None,
        help="Optional cap on initial passages (after chunking).",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Optional cap on update passages (after chunking).",
    )
    parser.add_argument(
        "--add-timeout",
        type=int,
        default=900,
        help="Timeout in seconds for the incremental add loop (0 = no timeout).",
    )

    args = parser.parse_args()

    register_project_directory(REPO_ROOT)

    initial_paragraphs = load_chunks_from_files(args.initial_files, args.max_initial)
    update_paragraphs = load_chunks_from_files(args.update_files, args.max_updates)
    if not update_paragraphs:
        raise ValueError("No update passages found; please provide --update-files with content.")

    update_chunks = prepare_new_chunks(update_paragraphs)
    ensure_index_dir(args.index_path)

    scenarios = [
        ("baseline", False, False),
        ("disable_forward_rng", True, False),
        ("disable_forward_and_reverse_rng", True, True),
    ]

    results_total: dict[str, list[float]] = {name: [] for name, _, _ in scenarios}
    results_add: dict[str, list[float]] = {name: [] for name, _, _ in scenarios}

    for run in range(args.runs):
        print(f"\n=== Benchmark run {run + 1}/{args.runs} ===")
        for name, disable_forward, disable_reverse in scenarios:
            print(f"\nScenario: {name}")
            cleanup_index_files(args.index_path)
            build_initial_index(
                args.index_path,
                initial_paragraphs,
                args.model_name,
                args.embedding_mode,
                args.distance_metric,
                args.ef_construction,
            )

            try:
                total_elapsed, add_elapsed = benchmark_update_with_mode(
                    args.index_path,
                    update_chunks,
                    args.model_name,
                    args.embedding_mode,
                    args.distance_metric,
                    disable_forward,
                    disable_reverse,
                    args.server_port,
                    args.add_timeout,
                )
            except TimeoutError as exc:
                print(f"Scenario {name} timed out: {exc}")
                continue

            per_chunk = add_elapsed / len(update_chunks)
            print(
                f"Total time: {total_elapsed:.3f} s | add-only: {add_elapsed:.3f} s "
                f"for {len(update_chunks)} passages => {per_chunk * 1e3:.3f} ms/passage"
            )
            results_total[name].append(total_elapsed)
            results_add[name].append(add_elapsed)

    print("\n=== Summary ===")
    for name in results_add:
        add_values = results_add[name]
        total_values = results_total[name]
        if not add_values:
            print(f"{name}: no successful runs")
            continue
        avg_add = sum(add_values) / len(add_values)
        avg_total = sum(total_values) / len(total_values)
        runs = len(add_values)
        print(
            f"{name}: add-only avg {avg_add:.3f} s | total avg {avg_total:.3f} s "
            f"over {runs} run(s)"
        )

    # leave the last build on disk for inspection


if __name__ == "__main__":
    main()
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
