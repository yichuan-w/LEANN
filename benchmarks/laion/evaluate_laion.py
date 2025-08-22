"""
LAION Multimodal Benchmark Evaluation Script - Modular Recall-based Evaluation
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from leann import LeannSearcher
from leann_backend_hnsw import faiss
from sentence_transformers import SentenceTransformer


class RecallEvaluator:
    """Stage 2: Evaluate Recall@3 (LEANN vs FAISS baseline for multimodal retrieval)"""

    def __init__(self, index_path: str, baseline_dir: str):
        self.index_path = index_path
        self.baseline_dir = baseline_dir
        self.searcher = LeannSearcher(index_path)

        # Load FAISS flat baseline (image embeddings)
        baseline_index_path = os.path.join(baseline_dir, "faiss_flat.index")
        metadata_path = os.path.join(baseline_dir, "metadata.pkl")

        self.faiss_index = faiss.read_index(baseline_index_path)
        with open(metadata_path, "rb") as f:
            self.image_ids = pickle.load(f)
        print(f"📚 Loaded FAISS flat baseline with {self.faiss_index.ntotal} image vectors")

        # Load sentence-transformers CLIP for text embedding (ViT-L/14)
        self.st_clip = SentenceTransformer("clip-ViT-L-14")

    def evaluate_recall_at_3(
        self, captions: list[str], complexity: int = 64, recompute_embeddings: bool = True
    ) -> float:
        """Evaluate recall@3 for multimodal retrieval: caption queries -> image results"""
        recompute_str = "with recompute" if recompute_embeddings else "no recompute"
        print(f"🔍 Evaluating recall@3 with complexity={complexity} ({recompute_str})...")

        total_recall = 0.0
        num_queries = len(captions)

        for i, caption in enumerate(captions):
            # Get ground truth: search with FAISS flat using caption text embedding
            # Generate CLIP text embedding for caption via sentence-transformers (normalized)
            query_embedding = self.st_clip.encode(
                [caption], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)

            # Search FAISS flat for ground truth using LEANN's modified faiss API
            n = query_embedding.shape[0]  # Number of queries
            k = 3  # Number of nearest neighbors
            distances = np.zeros((n, k), dtype=np.float32)
            labels = np.zeros((n, k), dtype=np.int64)

            self.faiss_index.search(
                n,
                faiss.swig_ptr(query_embedding),
                k,
                faiss.swig_ptr(distances),
                faiss.swig_ptr(labels),
            )

            # Extract the results (image IDs from FAISS)
            baseline_ids = {self.image_ids[idx] for idx in labels[0]}

            # Search with LEANN at specified complexity (using caption as text query)
            test_results = self.searcher.search(
                caption,
                top_k=3,
                complexity=complexity,
                recompute_embeddings=recompute_embeddings,
            )
            test_ids = {result.id for result in test_results}

            # Calculate recall@3 = |intersection| / |ground_truth|
            intersection = test_ids.intersection(baseline_ids)
            recall = len(intersection) / 3.0  # Ground truth size is 3
            total_recall += recall

            if i < 3:  # Show first few examples
                print(f"  Query {i + 1}: '{caption[:50]}...' -> Recall@3: {recall:.3f}")
                print(f"    FAISS ground truth: {list(baseline_ids)}")
                print(f"    LEANN results (C={complexity}, {recompute_str}): {list(test_ids)}")
                print(f"    Intersection: {list(intersection)}")

        avg_recall = total_recall / num_queries
        print(f"📊 Average Recall@3: {avg_recall:.3f} ({avg_recall * 100:.1f}%)")
        return avg_recall

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "searcher"):
            self.searcher.cleanup()


class LAIONEvaluator:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.searcher = LeannSearcher(index_path)

    def load_queries(self, queries_file: str) -> list[str]:
        """Load caption queries from evaluation file"""
        captions = []
        with open(queries_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    query_data = json.loads(line)
                    captions.append(query_data["query"])

        print(f"📊 Loaded {len(captions)} caption queries")
        return captions

    def analyze_index_sizes(self) -> dict:
        """Analyze index sizes, emphasizing .index only (exclude passages)."""
        print("📏 Analyzing index sizes (.index only)...")

        # Get all index-related files
        index_path = Path(self.index_path)
        index_dir = index_path.parent
        index_name = index_path.stem  # Remove .leann extension

        sizes: dict[str, float] = {}

        # Core index files
        index_file = index_dir / f"{index_name}.index"
        meta_file = index_dir / f"{index_path.name}.meta.json"  # Keep .leann for meta file
        passages_file = index_dir / f"{index_path.name}.passages.jsonl"  # Keep .leann for passages
        passages_idx_file = index_dir / f"{index_path.name}.passages.idx"  # Keep .leann for idx

        # Core index size (.index only)
        index_mb = index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0.0
        sizes["index_only_mb"] = index_mb

        # Other files for reference (not counted in index_only_mb)
        sizes["metadata_mb"] = (
            meta_file.stat().st_size / (1024 * 1024) if meta_file.exists() else 0.0
        )
        sizes["passages_text_mb"] = (
            passages_file.stat().st_size / (1024 * 1024) if passages_file.exists() else 0.0
        )
        sizes["passages_index_mb"] = (
            passages_idx_file.stat().st_size / (1024 * 1024) if passages_idx_file.exists() else 0.0
        )

        print(f"  📁 .index size: {index_mb:.1f} MB")
        if sizes["metadata_mb"]:
            print(f"  🧾 metadata: {sizes['metadata_mb']:.3f} MB")
        if sizes["passages_text_mb"] or sizes["passages_index_mb"]:
            print(
                f"  (passages excluded) text: {sizes['passages_text_mb']:.1f} MB, idx: {sizes['passages_index_mb']:.1f} MB"
            )

        return sizes

    def create_non_compact_index_for_comparison(self, non_compact_index_path: str) -> dict:
        """Create a non-compact index for comparison purposes"""
        print("🏗️ Building non-compact index from existing passages...")

        # Load existing passages from current index
        from leann import LeannBuilder

        current_index_path = Path(self.index_path)
        current_index_dir = current_index_path.parent
        current_index_name = current_index_path.name

        # Read metadata to get passage source
        meta_path = current_index_dir / f"{current_index_name}.meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        passage_source = meta["passage_sources"][0]
        passage_file = passage_source["path"]

        # Convert relative path to absolute
        if not Path(passage_file).is_absolute():
            passage_file = current_index_dir / Path(passage_file).name

        print(f"📄 Loading passages from {passage_file}...")

        # Load CLIP embeddings
        embeddings_file = current_index_dir / "clip_image_embeddings.npy"
        embeddings = np.load(embeddings_file)
        print(f"📐 Loaded embeddings shape: {embeddings.shape}")

        # Build non-compact index with same passages and embeddings
        builder = LeannBuilder(
            backend_name="hnsw",
            # Use CLIP text encoder (ViT-L/14) to match image embeddings (768-dim)
            embedding_model="clip-ViT-L-14",
            embedding_mode="sentence-transformers",
            is_recompute=False,  # Disable recompute (store embeddings)
            is_compact=False,  # Disable compact storage
            distance_metric="cosine",
            **{
                k: v
                for k, v in meta.get("backend_kwargs", {}).items()
                if k not in ["is_recompute", "is_compact", "distance_metric"]
            },
        )

        # Prepare ids and add passages
        ids: list[str] = []
        with open(passage_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    ids.append(str(data["id"]))
                    # Ensure metadata contains the id used by the vector index
                    metadata = {**data.get("metadata", {}), "id": data["id"]}
                    builder.add_text(text=data["text"], metadata=metadata)

        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"IDs count ({len(ids)}) does not match embeddings ({embeddings.shape[0]})."
            )

        # Persist a pickle for build_index_from_embeddings
        pkl_path = current_index_dir / "clip_image_embeddings.pkl"
        with open(pkl_path, "wb") as pf:
            pickle.dump((ids, embeddings.astype(np.float32)), pf)

        print(
            f"🔨 Building non-compact index at {non_compact_index_path} from precomputed embeddings..."
        )
        builder.build_index_from_embeddings(non_compact_index_path, str(pkl_path))

        # Analyze the non-compact index size
        temp_evaluator = LAIONEvaluator(non_compact_index_path)
        non_compact_sizes = temp_evaluator.analyze_index_sizes()
        non_compact_sizes["index_type"] = "non_compact"

        return non_compact_sizes

    def compare_index_performance(
        self, non_compact_path: str, compact_path: str, test_captions: list, complexity: int
    ) -> dict:
        """Compare performance between non-compact and compact indexes"""
        print("⚡ Comparing search performance between indexes...")

        # Test queries
        test_queries = test_captions[:5]

        results = {
            "non_compact": {"search_times": []},
            "compact": {"search_times": []},
            "avg_search_times": {},
            "speed_ratio": 0.0,
        }

        # Test non-compact index (no recompute)
        print("  🔍 Testing non-compact index (no recompute)...")
        non_compact_searcher = LeannSearcher(non_compact_path)

        for caption in test_queries:
            start_time = time.time()
            search_results = non_compact_searcher.search(
                caption, top_k=3, complexity=complexity, recompute_embeddings=False
            )
            search_time = time.time() - start_time
            results["non_compact"]["search_times"].append(search_time)

        # Test compact index (with recompute)
        print("  🔍 Testing compact index (with recompute)...")
        compact_searcher = LeannSearcher(compact_path)

        for caption in test_queries:
            start_time = time.time()
            search_results = compact_searcher.search(
                caption, top_k=3, complexity=complexity, recompute_embeddings=True
            )
            search_time = time.time() - start_time
            results["compact"]["search_times"].append(search_time)

        # Calculate averages
        results["avg_search_times"]["non_compact"] = sum(
            results["non_compact"]["search_times"]
        ) / len(results["non_compact"]["search_times"])
        results["avg_search_times"]["compact"] = sum(results["compact"]["search_times"]) / len(
            results["compact"]["search_times"]
        )

        # Performance ratio
        if results["avg_search_times"]["compact"] > 0:
            results["speed_ratio"] = (
                results["avg_search_times"]["non_compact"] / results["avg_search_times"]["compact"]
            )
        else:
            results["speed_ratio"] = float("inf")

        print(
            f"    Non-compact (no recompute): {results['avg_search_times']['non_compact']:.3f}s avg"
        )
        print(f"    Compact (with recompute): {results['avg_search_times']['compact']:.3f}s avg")
        print(f"    Speed ratio: {results['speed_ratio']:.2f}x")

        # Cleanup
        non_compact_searcher.cleanup()
        compact_searcher.cleanup()

        return results

    def _print_results(self, timing_metrics: dict):
        """Print evaluation results"""
        print("\n🎯 LAION MULTIMODAL BENCHMARK RESULTS")
        print("=" * 60)

        # Index comparison analysis (prefer .index-only view if present)
        if "current_index" in timing_metrics and "non_compact_index" in timing_metrics:
            current = timing_metrics["current_index"]
            non_compact = timing_metrics["non_compact_index"]

            if "index_only_mb" in current and "index_only_mb" in non_compact:
                print("\n📏 Index Comparison Analysis (.index only):")
                print(f"  Compact index (current): {current.get('index_only_mb', 0):.1f} MB")
                print(f"  Non-compact index: {non_compact.get('index_only_mb', 0):.1f} MB")
                print(
                    f"  Storage saving by compact: {timing_metrics.get('storage_saving_percent', 0):.1f}%"
                )
                # Show excluded components for reference if available
                if any(k in non_compact for k in ("passages_text_mb", "passages_index_mb", "metadata_mb")):
                    print("  (passages excluded in totals, shown for reference):")
                    print(
                        f"    - Passages text: {non_compact.get('passages_text_mb', 0):.1f} MB, "
                        f"Passages index: {non_compact.get('passages_index_mb', 0):.1f} MB, "
                        f"Metadata: {non_compact.get('metadata_mb', 0):.3f} MB"
                    )
            else:
                # Fallback to legacy totals if running with older metrics
                print("\n📏 Index Comparison Analysis:")
                print(f"  Compact index (current): {current.get('total_with_embeddings', 0):.1f} MB")
                print(
                    f"  Non-compact index (with embeddings): {non_compact.get('total_with_embeddings', 0):.1f} MB"
                )
                print(
                    f"  Storage saving by compact: {timing_metrics.get('storage_saving_percent', 0):.1f}%"
                )
                print("  Component breakdown (non-compact):")
                print(f"    - Main index: {non_compact.get('index', 0):.1f} MB")
                print(f"    - Passages text: {non_compact.get('passages_text', 0):.1f} MB")
                print(f"    - Passages index: {non_compact.get('passages_index', 0):.1f} MB")
                print(f"    - Metadata: {non_compact.get('metadata', 0):.1f} MB")

        # Performance comparison
        if "performance_comparison" in timing_metrics:
            perf = timing_metrics["performance_comparison"]
            print("\n⚡ Performance Comparison:")
            print(
                f"  Non-compact (no recompute): {perf.get('avg_search_times', {}).get('non_compact', 0):.3f}s avg"
            )
            print(
                f"  Compact (with recompute): {perf.get('avg_search_times', {}).get('compact', 0):.3f}s avg"
            )
            print(f"  Speed ratio: {perf.get('speed_ratio', 0):.2f}x")

        # Legacy single index analysis (fallback)
        if "total_with_embeddings" in timing_metrics and "current_index" not in timing_metrics:
            print("\n📏 Index Size Analysis:")
            print(
                f"  Index with embeddings: {timing_metrics.get('total_with_embeddings', 0):.1f} MB"
            )
            print(
                f"  Estimated pruned index: {timing_metrics.get('total_without_embeddings', 0):.1f} MB"
            )
            print(f"  Compression ratio: {timing_metrics.get('compression_ratio', 0):.2f}x")

    def cleanup(self):
        """Cleanup resources"""
        if self.searcher:
            self.searcher.cleanup()


def main():
    parser = argparse.ArgumentParser(description="LAION Multimodal Benchmark Evaluation")
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
    parser.add_argument("--complexity", type=int, default=None, help="Complexity for search")
    parser.add_argument("--baseline-dir", default="baseline", help="Baseline output directory")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    try:
        # Check if baseline exists
        baseline_index_path = os.path.join(args.baseline_dir, "faiss_flat.index")
        if not os.path.exists(baseline_index_path):
            print(f"❌ FAISS baseline not found at {baseline_index_path}")
            print("💡 Please run setup_laion.py first to build the baseline")
            exit(1)

        if args.stage == "2" or args.stage == "all":
            # Stage 2: Recall@3 evaluation
            print("🚀 Starting Stage 2: Recall@3 evaluation for multimodal retrieval")

            evaluator = RecallEvaluator(args.index, args.baseline_dir)

            # Load caption queries for testing
            laion_evaluator = LAIONEvaluator(args.index)
            captions = laion_evaluator.load_queries(args.queries)

            # Test with queries for robust measurement
            test_captions = captions[:100]  # Use subset for speed
            print(f"🧪 Testing with {len(test_captions)} caption queries")

            # Test with complexity 64
            complexity = 64
            recall = evaluator.evaluate_recall_at_3(test_captions, complexity)
            print(f"📈 Recall@3 at complexity {complexity}: {recall * 100:.1f}%")

            evaluator.cleanup()
            print("✅ Stage 2 completed!\n")

        # Shared non-compact index path for Stage 3 and 4
        non_compact_index_path = args.index.replace(".leann", "_noncompact.leann")
        complexity = args.complexity

        if args.stage == "3" or args.stage == "all":
            # Stage 3: Binary search for 90% recall complexity
            print("🚀 Starting Stage 3: Binary search for 90% recall complexity")
            print(
                "💡 Creating non-compact index for fast binary search with recompute_embeddings=False"
            )

            # Create non-compact index for binary search
            print("🏗️ Creating non-compact index for binary search...")
            evaluator = LAIONEvaluator(args.index)
            evaluator.create_non_compact_index_for_comparison(non_compact_index_path)

            # Use non-compact index for binary search
            binary_search_evaluator = RecallEvaluator(non_compact_index_path, args.baseline_dir)

            # Load caption queries for testing
            captions = evaluator.load_queries(args.queries)

            # Use subset for robust measurement
            test_captions = captions[:50]  # Smaller subset for binary search speed
            print(f"🧪 Testing with {len(test_captions)} caption queries")

            # Binary search for 90% recall complexity
            target_recall = 0.9
            min_complexity, max_complexity = 1, 128

            print(f"🔍 Binary search for {target_recall * 100}% recall complexity...")
            print(f"Search range: {min_complexity} to {max_complexity}")

            best_complexity = None
            best_recall = 0.0

            while min_complexity <= max_complexity:
                mid_complexity = (min_complexity + max_complexity) // 2

                print(
                    f"\n🧪 Testing complexity {mid_complexity} (no recompute, non-compact index)..."
                )
                # Use recompute_embeddings=False on non-compact index for fast binary search
                recall = binary_search_evaluator.evaluate_recall_at_3(
                    test_captions, mid_complexity, recompute_embeddings=False
                )

                print(
                    f"  Complexity {mid_complexity}: Recall@3 = {recall:.3f} ({recall * 100:.1f}%)"
                )

                if recall >= target_recall:
                    best_complexity = mid_complexity
                    best_recall = recall
                    max_complexity = mid_complexity - 1
                    print("  ✅ Target reached! Searching for lower complexity...")
                else:
                    min_complexity = mid_complexity + 1
                    print("  ❌ Below target. Searching for higher complexity...")

            if best_complexity is not None:
                print("\n🎯 Optimal complexity found!")
                print(f"  Complexity: {best_complexity}")
                print(f"  Recall@3: {best_recall:.3f} ({best_recall * 100:.1f}%)")

                # Test a few complexities around the optimal one for verification
                print("\n🔬 Verification test around optimal complexity:")
                verification_complexities = [
                    max(1, best_complexity - 2),
                    max(1, best_complexity - 1),
                    best_complexity,
                    best_complexity + 1,
                    best_complexity + 2,
                ]

                for complexity in verification_complexities:
                    if complexity <= 512:  # reasonable upper bound
                        recall = binary_search_evaluator.evaluate_recall_at_3(
                            test_captions, complexity, recompute_embeddings=False
                        )
                        status = "✅" if recall >= target_recall else "❌"
                        print(f"  {status} Complexity {complexity:3d}: {recall * 100:5.1f}%")

                # Now test the optimal complexity with compact index and recompute for comparison
                print(
                    f"\n🔄 Testing optimal complexity {best_complexity} on compact index WITH recompute..."
                )
                compact_evaluator = RecallEvaluator(args.index, args.baseline_dir)
                recall_with_recompute = compact_evaluator.evaluate_recall_at_3(
                    test_captions[:10], best_complexity, recompute_embeddings=True
                )
                print(
                    f"  ✅ Complexity {best_complexity} (compact index with recompute): {recall_with_recompute * 100:.1f}%"
                )
                complexity = best_complexity
                print(
                    f"  📊 Recall difference: {abs(best_recall - recall_with_recompute) * 100:.2f}%"
                )
                compact_evaluator.cleanup()
            else:
                print(f"\n❌ Could not find complexity achieving {target_recall * 100}% recall")
                print("All tested complexities were below target.")

            # Cleanup evaluators (keep non-compact index for Stage 4)
            binary_search_evaluator.cleanup()
            evaluator.cleanup()

            print("✅ Stage 3 completed! Non-compact index saved for Stage 4.\n")

        if args.stage == "4" or args.stage == "all":
            # Stage 4: Index comparison (without LLM generation)
            print("🚀 Starting Stage 4: Index comparison analysis")

            # Use LAION evaluator for index comparison
            evaluator = LAIONEvaluator(args.index)

            # Load caption queries
            captions = evaluator.load_queries(args.queries)

            # Step 1: Analyze current (compact) index
            print("\n📏 Analyzing current index (compact, pruned)...")
            compact_size_metrics = evaluator.analyze_index_sizes()
            compact_size_metrics["index_type"] = "compact"

            # Step 2: Use existing non-compact index or create if needed
            if Path(non_compact_index_path).exists():
                print(
                    f"\n📁 Using existing non-compact index from Stage 3: {non_compact_index_path}"
                )
                temp_evaluator = LAIONEvaluator(non_compact_index_path)
                non_compact_size_metrics = temp_evaluator.analyze_index_sizes()
                non_compact_size_metrics["index_type"] = "non_compact"
            else:
                print("\n🏗️ Creating non-compact index (with embeddings) for comparison...")
                non_compact_size_metrics = evaluator.create_non_compact_index_for_comparison(
                    non_compact_index_path
                )

            # Step 3: Compare index sizes (.index only)
            print("\n📊 Index size comparison (.index only):")
            print(
                f"  Compact index (current): {compact_size_metrics.get('index_only_mb', 0):.1f} MB"
            )
            print(f"  Non-compact index: {non_compact_size_metrics.get('index_only_mb', 0):.1f} MB")

            storage_saving = 0.0
            if non_compact_size_metrics.get("index_only_mb", 0) > 0:
                storage_saving = (
                    (
                        non_compact_size_metrics.get("index_only_mb", 0)
                        - compact_size_metrics.get("index_only_mb", 0)
                    )
                    / non_compact_size_metrics.get("index_only_mb", 1)
                    * 100
                )
            print(f"  Storage saving by compact: {storage_saving:.1f}%")

            # Step 4: Performance comparison between the two indexes
            if complexity is None:
                raise ValueError("Complexity is required for index comparison")

            print("\n⚡ Performance comparison between indexes...")
            performance_metrics = evaluator.compare_index_performance(
                non_compact_index_path, args.index, captions[:10], complexity=complexity
            )

            # Combine all metrics
            combined_metrics = {
                "current_index": compact_size_metrics,
                "non_compact_index": non_compact_size_metrics,
                "performance_comparison": performance_metrics,
                "storage_saving_percent": storage_saving,
            }

            # Print comprehensive results
            evaluator._print_results(combined_metrics)

            # Save results if requested
            if args.output:
                print(f"\n💾 Saving results to {args.output}...")
                with open(args.output, "w") as f:
                    json.dump(combined_metrics, f, indent=2, default=str)
                print(f"✅ Results saved to {args.output}")

            evaluator.cleanup()
            print("✅ Stage 4 completed!\n")

        if args.stage == "all":
            print("🎉 All evaluation stages completed successfully!")
            print("\n📋 Summary:")
            print("  Stage 2: ✅ Multimodal Recall@3 evaluation completed")
            print("  Stage 3: ✅ Optimal complexity found")
            print("  Stage 4: ✅ Index comparison analysis completed")
            print("\n🔧 Recommended next steps:")
            print("  - Use optimal complexity for best speed/accuracy balance")
            print("  - Review index comparison for storage vs performance tradeoffs")

            # Clean up non-compact index after all stages complete
            print("\n🧹 Cleaning up temporary non-compact index...")
            if Path(non_compact_index_path).exists():
                temp_index_dir = Path(non_compact_index_path).parent
                temp_index_name = Path(non_compact_index_path).name
                for temp_file in temp_index_dir.glob(f"{temp_index_name}*"):
                    temp_file.unlink()
                print(f"✅ Cleaned up {non_compact_index_path}")
            else:
                print("📝 No temporary index to clean up")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Stage {args.stage} failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
