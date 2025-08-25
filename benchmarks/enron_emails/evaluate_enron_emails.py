"""
Enron Emails Benchmark Evaluation - Retrieval Recall@3 (Stages 2/3/4)
Follows the style of FinanceBench/LAION: Stage 2 recall vs FAISS baseline,
Stage 3 complexity sweep to target recall, Stage 4 index comparison.
On errors, fail fast without fallbacks.
"""

import argparse
import json
import os
import pickle

import numpy as np
from leann import LeannBuilder, LeannSearcher
from leann_backend_hnsw import faiss


class RecallEvaluator:
    """Stage 2: Evaluate Recall@3 (LEANN vs FAISS)"""

    def __init__(self, index_path: str, baseline_dir: str):
        self.index_path = index_path
        self.baseline_dir = baseline_dir
        self.searcher = LeannSearcher(index_path)

        baseline_index_path = os.path.join(baseline_dir, "faiss_flat.index")
        metadata_path = os.path.join(baseline_dir, "metadata.pkl")

        self.faiss_index = faiss.read_index(baseline_index_path)
        with open(metadata_path, "rb") as f:
            self.passage_ids = pickle.load(f)

        print(f"📚 Loaded FAISS flat baseline with {self.faiss_index.ntotal} vectors")

        # No fallbacks here; if embedding server is needed but fails, the caller will see the error.

    def evaluate_recall_at_3(
        self, queries: list[str], complexity: int = 64, recompute_embeddings: bool = True
    ) -> float:
        """Evaluate recall@3 using FAISS Flat as ground truth"""
        from leann.api import compute_embeddings

        recompute_str = "with recompute" if recompute_embeddings else "no recompute"
        print(f"🔍 Evaluating recall@3 with complexity={complexity} ({recompute_str})...")

        total_recall = 0.0
        for i, query in enumerate(queries):
            # Compute query embedding with the same model/mode as the index
            q_emb = compute_embeddings(
                [query],
                self.searcher.embedding_model,
                mode=self.searcher.embedding_mode,
                use_server=False,
            ).astype(np.float32)

            # Search FAISS Flat ground truth
            n = q_emb.shape[0]
            k = 3
            distances = np.zeros((n, k), dtype=np.float32)
            labels = np.zeros((n, k), dtype=np.int64)
            self.faiss_index.search(
                n,
                faiss.swig_ptr(q_emb),
                k,
                faiss.swig_ptr(distances),
                faiss.swig_ptr(labels),
            )

            baseline_ids = {self.passage_ids[idx] for idx in labels[0]}

            # Search with LEANN (may require embedding server depending on index configuration)
            results = self.searcher.search(
                query,
                top_k=3,
                complexity=complexity,
                recompute_embeddings=recompute_embeddings,
            )
            test_ids = {r.id for r in results}

            intersection = test_ids.intersection(baseline_ids)
            recall = len(intersection) / 3.0
            total_recall += recall

            if i < 3:
                print(f"  Q{i + 1}: '{query[:60]}...' -> Recall@3: {recall:.3f}")
                print(f"    FAISS: {list(baseline_ids)}")
                print(f"    LEANN: {list(test_ids)}")
                print(f"    ∩: {list(intersection)}")

        avg = total_recall / max(1, len(queries))
        print(f"📊 Average Recall@3: {avg:.3f} ({avg * 100:.1f}%)")
        return avg

    def cleanup(self):
        if hasattr(self, "searcher"):
            self.searcher.cleanup()


class EnronEvaluator:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.searcher = LeannSearcher(index_path)

    def load_queries(self, queries_file: str) -> list[str]:
        queries: list[str] = []
        with open(queries_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if "query" in data:
                    queries.append(data["query"])
        print(f"📊 Loaded {len(queries)} queries from {queries_file}")
        return queries

    def cleanup(self):
        if self.searcher:
            self.searcher.cleanup()

    def analyze_index_sizes(self) -> dict:
        """Analyze index sizes (.index only), similar to LAION bench."""
        from pathlib import Path

        print("📏 Analyzing index sizes (.index only)...")
        index_path = Path(self.index_path)
        index_dir = index_path.parent
        index_name = index_path.stem

        sizes: dict[str, float] = {}
        index_file = index_dir / f"{index_name}.index"
        meta_file = index_dir / f"{index_path.name}.meta.json"
        passages_file = index_dir / f"{index_path.name}.passages.jsonl"
        passages_idx_file = index_dir / f"{index_path.name}.passages.idx"

        sizes["index_only_mb"] = (
            index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0.0
        )
        sizes["metadata_mb"] = (
            meta_file.stat().st_size / (1024 * 1024) if meta_file.exists() else 0.0
        )
        sizes["passages_text_mb"] = (
            passages_file.stat().st_size / (1024 * 1024) if passages_file.exists() else 0.0
        )
        sizes["passages_index_mb"] = (
            passages_idx_file.stat().st_size / (1024 * 1024) if passages_idx_file.exists() else 0.0
        )

        print(f"  📁 .index size: {sizes['index_only_mb']:.1f} MB")
        return sizes

    def create_non_compact_index_for_comparison(self, non_compact_index_path: str) -> dict:
        """Create a non-compact index for comparison using current passages and embeddings."""
        from pathlib import Path

        current_index_path = Path(self.index_path)
        current_index_dir = current_index_path.parent
        current_index_name = current_index_path.name

        # Read metadata to get passage source and embedding model
        meta_path = current_index_dir / f"{current_index_name}.meta.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        passage_source = meta["passage_sources"][0]
        passage_file = passage_source["path"]

        # Convert relative path to absolute
        if not Path(passage_file).is_absolute():
            passage_file = current_index_dir / Path(passage_file).name

        # Load all passages and ids
        ids: list[str] = []
        texts: list[str] = []
        with open(passage_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    ids.append(str(data["id"]))
                    texts.append(data["text"])

        # Compute embeddings using the same method as LEANN
        from leann.api import compute_embeddings

        embeddings = compute_embeddings(
            texts,
            meta["embedding_model"],
            mode=meta.get("embedding_mode", "sentence-transformers"),
            use_server=False,
        ).astype(np.float32)

        # Build non-compact index with same passages and embeddings
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=meta["embedding_model"],
            embedding_mode=meta.get("embedding_mode", "sentence-transformers"),
            is_recompute=False,
            is_compact=False,
            **{
                k: v
                for k, v in meta.get("backend_kwargs", {}).items()
                if k not in ["is_recompute", "is_compact"]
            },
        )

        # Persist a pickle for build_index_from_embeddings
        pkl_path = current_index_dir / f"{Path(non_compact_index_path).stem}_embeddings.pkl"
        with open(pkl_path, "wb") as pf:
            pickle.dump((ids, embeddings), pf)

        print(
            f"🔨 Building non-compact index at {non_compact_index_path} from precomputed embeddings..."
        )
        builder.build_index_from_embeddings(non_compact_index_path, str(pkl_path))

        # Analyze the non-compact index size
        temp_evaluator = EnronEvaluator(non_compact_index_path)
        non_compact_sizes = temp_evaluator.analyze_index_sizes()
        non_compact_sizes["index_type"] = "non_compact"

        return non_compact_sizes

    def compare_index_performance(
        self, non_compact_path: str, compact_path: str, test_queries: list[str], complexity: int
    ) -> dict:
        """Compare search speed for non-compact vs compact indexes."""
        import time

        results: dict = {
            "non_compact": {"search_times": []},
            "compact": {"search_times": []},
            "avg_search_times": {},
            "speed_ratio": 0.0,
        }

        print("⚡ Comparing search performance between indexes...")
        # Non-compact (no recompute)
        print("  🔍 Testing non-compact index (no recompute)...")
        non_compact_searcher = LeannSearcher(non_compact_path)
        for q in test_queries:
            t0 = time.time()
            _ = non_compact_searcher.search(
                q, top_k=3, complexity=complexity, recompute_embeddings=False
            )
            results["non_compact"]["search_times"].append(time.time() - t0)

        # Compact (with recompute). Fail fast if it cannot run.
        print("  🔍 Testing compact index (with recompute)...")
        compact_searcher = LeannSearcher(compact_path)
        for q in test_queries:
            t0 = time.time()
            _ = compact_searcher.search(
                q, top_k=3, complexity=complexity, recompute_embeddings=True
            )
            results["compact"]["search_times"].append(time.time() - t0)
        compact_searcher.cleanup()

        if results["non_compact"]["search_times"]:
            results["avg_search_times"]["non_compact"] = sum(
                results["non_compact"]["search_times"]
            ) / len(results["non_compact"]["search_times"])
        if results["compact"]["search_times"]:
            results["avg_search_times"]["compact"] = sum(results["compact"]["search_times"]) / len(
                results["compact"]["search_times"]
            )
        if results["avg_search_times"].get("compact", 0) > 0:
            results["speed_ratio"] = (
                results["avg_search_times"]["non_compact"] / results["avg_search_times"]["compact"]
            )
        else:
            results["speed_ratio"] = 0.0

        non_compact_searcher.cleanup()
        return results

    def evaluate_complexity(
        self,
        recall_eval: "RecallEvaluator",
        queries: list[str],
        target: float = 0.90,
        c_min: int = 8,
        c_max: int = 256,
        max_iters: int = 10,
        recompute: bool = False,
    ) -> dict:
        """Binary search minimal complexity achieving target recall (monotonic assumption)."""

        def round_c(x: int) -> int:
            # snap to multiple of 8 like other benches typically do
            return max(1, int((x + 7) // 8) * 8)

        metrics: list[dict] = []

        lo = round_c(c_min)
        hi = round_c(c_max)

        print(
            f"🧪 Binary search complexity in [{lo}, {hi}] for target Recall@3>={int(target * 100)}%..."
        )

        # Ensure upper bound can reach target; expand if needed (up to a cap)
        r_lo = recall_eval.evaluate_recall_at_3(
            queries, complexity=lo, recompute_embeddings=recompute
        )
        metrics.append({"complexity": lo, "recall_at_3": r_lo})
        r_hi = recall_eval.evaluate_recall_at_3(
            queries, complexity=hi, recompute_embeddings=recompute
        )
        metrics.append({"complexity": hi, "recall_at_3": r_hi})

        cap = 1024
        while r_hi < target and hi < cap:
            lo = hi
            r_lo = r_hi
            hi = round_c(hi * 2)
            r_hi = recall_eval.evaluate_recall_at_3(
                queries, complexity=hi, recompute_embeddings=recompute
            )
            metrics.append({"complexity": hi, "recall_at_3": r_hi})

        if r_hi < target:
            print(f"⚠️ Max complexity {hi} did not reach target recall {target:.2f}.")
            print("📈 Observations:")
            for m in metrics:
                print(f"  C={m['complexity']:>4} -> Recall@3={m['recall_at_3'] * 100:.1f}%")
            return {"metrics": metrics, "best_complexity": None, "target_recall": target}

        # Binary search within [lo, hi]
        best = hi
        iters = 0
        while lo < hi and iters < max_iters:
            mid = round_c((lo + hi) // 2)
            r_mid = recall_eval.evaluate_recall_at_3(
                queries, complexity=mid, recompute_embeddings=recompute
            )
            metrics.append({"complexity": mid, "recall_at_3": r_mid})
            if r_mid >= target:
                best = mid
                hi = mid
            else:
                lo = mid + 8  # move past mid, respecting multiple-of-8 step
            iters += 1

        print("📈 Binary search results (sampled points):")
        # Print unique complexity entries ordered by complexity
        for m in sorted(
            {m["complexity"]: m for m in metrics}.values(), key=lambda x: x["complexity"]
        ):
            print(f"  C={m['complexity']:>4} -> Recall@3={m['recall_at_3'] * 100:.1f}%")
        print(f"✅ Minimal complexity achieving {int(target * 100)}% recall: {best}")
        return {"metrics": metrics, "best_complexity": best, "target_recall": target}


def main():
    parser = argparse.ArgumentParser(description="Enron Emails Benchmark Evaluation")
    parser.add_argument("--index", required=True, help="Path to LEANN index")
    parser.add_argument(
        "--queries", default="data/evaluation_queries.jsonl", help="Path to evaluation queries"
    )
    parser.add_argument(
        "--stage",
        choices=["2", "3", "4", "all"],
        default="all",
        help="Which stage to run (2=recall, 3=complexity, 4=index comparison)",
    )
    parser.add_argument("--complexity", type=int, default=None, help="LEANN search complexity")
    parser.add_argument("--baseline-dir", default="baseline", help="Baseline output directory")
    parser.add_argument(
        "--max-queries", type=int, help="Limit number of queries to evaluate", default=1000
    )
    parser.add_argument(
        "--target-recall", type=float, default=0.90, help="Target Recall@3 for Stage 3"
    )
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Resolve queries file: if default path not found, fall back to index's directory
    if not os.path.exists(args.queries):
        from pathlib import Path

        idx_dir = Path(args.index).parent
        fallback_q = idx_dir / "evaluation_queries.jsonl"
        if fallback_q.exists():
            args.queries = str(fallback_q)

    baseline_index_path = os.path.join(args.baseline_dir, "faiss_flat.index")
    if not os.path.exists(baseline_index_path):
        print(f"❌ FAISS baseline not found at {baseline_index_path}")
        print("💡 Please run setup_enron_emails.py first to build the baseline")
        raise SystemExit(1)

    results_out: dict = {}

    if args.stage in ("2", "all"):
        print("🚀 Starting Stage 2: Recall@3 evaluation")
        evaluator = RecallEvaluator(args.index, args.baseline_dir)

        enron_eval = EnronEvaluator(args.index)
        queries = enron_eval.load_queries(args.queries)
        queries = queries[:10]
        print(f"🧪 Using first {len(queries)} queries")

        complexity = args.complexity or 64
        r = evaluator.evaluate_recall_at_3(queries, complexity)
        results_out["stage2"] = {"complexity": complexity, "recall_at_3": r}
        evaluator.cleanup()
        enron_eval.cleanup()
        print("✅ Stage 2 completed!\n")

    if args.stage in ("3", "all"):
        print("🚀 Starting Stage 3: Binary search for target recall (no recompute)")
        enron_eval = EnronEvaluator(args.index)
        queries = enron_eval.load_queries(args.queries)
        queries = queries[: args.max_queries]
        print(f"🧪 Using first {len(queries)} queries")

        # Build non-compact index for fast binary search (recompute_embeddings=False)
        from pathlib import Path

        index_path = Path(args.index)
        non_compact_index_path = str(index_path.parent / f"{index_path.stem}_noncompact.leann")
        enron_eval.create_non_compact_index_for_comparison(non_compact_index_path)

        # Use non-compact evaluator for binary search with recompute=False
        evaluator_nc = RecallEvaluator(non_compact_index_path, args.baseline_dir)
        sweep = enron_eval.evaluate_complexity(
            evaluator_nc, queries, target=args.target_recall, recompute=False
        )
        results_out["stage3"] = sweep
        # Persist default stage 3 results near the index for Stage 4 auto-pickup
        from pathlib import Path

        default_stage3_path = Path(args.index).parent / "enron_stage3_results.json"
        with open(default_stage3_path, "w", encoding="utf-8") as f:
            json.dump({"stage3": sweep}, f, indent=2)
        print(f"📝 Saved Stage 3 summary to {default_stage3_path}")
        evaluator_nc.cleanup()
        enron_eval.cleanup()
        print("✅ Stage 3 completed!\n")

    if args.stage in ("4", "all"):
        print("🚀 Starting Stage 4: Index size + performance comparison")
        evaluator = RecallEvaluator(args.index, args.baseline_dir)
        enron_eval = EnronEvaluator(args.index)
        queries = enron_eval.load_queries(args.queries)
        test_q = queries[: min(args.max_queries, len(queries))]

        current_sizes = enron_eval.analyze_index_sizes()
        # Build non-compact index for comparison (no fallback)
        from pathlib import Path

        index_path = Path(args.index)
        non_compact_path = str(index_path.parent / f"{index_path.stem}_noncompact.leann")
        non_compact_sizes = enron_eval.create_non_compact_index_for_comparison(non_compact_path)
        nc_eval = EnronEvaluator(non_compact_path)

        if (
            current_sizes.get("index_only_mb", 0) > 0
            and non_compact_sizes.get("index_only_mb", 0) > 0
        ):
            storage_saving_percent = max(
                0.0,
                100.0 * (1.0 - current_sizes["index_only_mb"] / non_compact_sizes["index_only_mb"]),
            )
        else:
            storage_saving_percent = 0.0

        if args.complexity is None:
            # Prefer in-session Stage 3 result
            if "stage3" in results_out and results_out["stage3"].get("best_complexity") is not None:
                complexity = results_out["stage3"]["best_complexity"]
                print(f"📥 Using best complexity from Stage 3 in-session: {complexity}")
            else:
                # Try to load last saved Stage 3 result near index
                default_stage3_path = Path(args.index).parent / "enron_stage3_results.json"
                if default_stage3_path.exists():
                    with open(default_stage3_path, encoding="utf-8") as f:
                        prev = json.load(f)
                    complexity = prev.get("stage3", {}).get("best_complexity")
                    if complexity is None:
                        raise SystemExit(
                            "❌ Stage 4: No --complexity and no best_complexity found in saved Stage 3 results"
                        )
                    print(f"📥 Using best complexity from saved Stage 3: {complexity}")
                else:
                    raise SystemExit(
                        "❌ Stage 4 requires --complexity if Stage 3 hasn't been run. Run stage 3 first or pass --complexity."
                    )
        else:
            complexity = args.complexity

        comp = enron_eval.compare_index_performance(
            non_compact_path, args.index, test_q, complexity=complexity
        )
        results_out["stage4"] = {
            "current_index": current_sizes,
            "non_compact_index": non_compact_sizes,
            "storage_saving_percent": storage_saving_percent,
            "performance_comparison": comp,
        }
        nc_eval.cleanup()
        evaluator.cleanup()
        enron_eval.cleanup()
        print("✅ Stage 4 completed!\n")

    if args.output and results_out:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results_out, f, indent=2)
        print(f"📝 Saved results to {args.output}")


if __name__ == "__main__":
    main()
