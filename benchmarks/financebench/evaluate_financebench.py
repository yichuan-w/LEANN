"""
FinanceBench Evaluation Script - Modular Recall-based Evaluation
"""

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import openai
from leann import LeannChat, LeannSearcher
from leann_backend_hnsw import faiss

from ..llm_utils import evaluate_rag, generate_hf, generate_vllm, load_hf_model, load_vllm_model

# Setup logging to reduce verbose output
logging.basicConfig(level=logging.WARNING)
logging.getLogger("leann.api").setLevel(logging.WARNING)
logging.getLogger("leann_backend_hnsw").setLevel(logging.WARNING)


class RecallEvaluator:
    """Stage 2: Evaluate Recall@3 (searcher vs baseline)"""

    def __init__(self, index_path: str, baseline_dir: str):
        self.index_path = index_path
        self.baseline_dir = baseline_dir
        self.searcher = LeannSearcher(index_path)

        # Load FAISS flat baseline
        baseline_index_path = os.path.join(baseline_dir, "faiss_flat.index")
        metadata_path = os.path.join(baseline_dir, "metadata.pkl")

        self.faiss_index = faiss.read_index(baseline_index_path)
        with open(metadata_path, "rb") as f:
            self.passage_ids = pickle.load(f)
        print(f"📚 Loaded FAISS flat baseline with {self.faiss_index.ntotal} vectors")

    def evaluate_recall_at_3(
        self, queries: list[str], complexity: int = 64, recompute_embeddings: bool = True
    ) -> float:
        """Evaluate recall@3 for given queries at specified complexity"""
        recompute_str = "with recompute" if recompute_embeddings else "no recompute"
        print(f"🔍 Evaluating recall@3 with complexity={complexity} ({recompute_str})...")

        total_recall = 0.0
        num_queries = len(queries)

        for i, query in enumerate(queries):
            # Get ground truth: search with FAISS flat
            from leann.api import compute_embeddings

            query_embedding = compute_embeddings(
                [query],
                self.searcher.embedding_model,
                mode=self.searcher.embedding_mode,
                use_server=False,
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

            # Extract the results
            baseline_ids = {self.passage_ids[idx] for idx in labels[0]}

            # Search with LEANN at specified complexity
            test_results = self.searcher.search(
                query,
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
                print(f"  Query {i + 1}: '{query[:50]}...' -> Recall@3: {recall:.3f}")
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


class FinanceBenchEvaluator:
    def __init__(self, index_path: str, openai_api_key: Optional[str] = None):
        self.index_path = index_path
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None

        self.searcher = LeannSearcher(index_path)
        self.chat = LeannChat(index_path) if openai_api_key else None

    def load_dataset(self, dataset_path: str = "data/financebench_merged.jsonl"):
        """Load FinanceBench dataset"""
        data = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        print(f"📊 Loaded {len(data)} FinanceBench examples")
        return data

    def analyze_index_sizes(self) -> dict:
        """Analyze index sizes with and without embeddings"""

        print("📏 Analyzing index sizes...")

        # Get all index-related files
        index_path = Path(self.index_path)
        index_dir = index_path.parent
        index_name = index_path.stem  # Remove .leann extension

        sizes = {}
        total_with_embeddings = 0

        # Core index files
        index_file = index_dir / f"{index_name}.index"
        meta_file = index_dir / f"{index_path.name}.meta.json"  # Keep .leann for meta file
        passages_file = index_dir / f"{index_path.name}.passages.jsonl"  # Keep .leann for passages
        passages_idx_file = index_dir / f"{index_path.name}.passages.idx"  # Keep .leann for idx

        for file_path, name in [
            (index_file, "index"),
            (meta_file, "metadata"),
            (passages_file, "passages_text"),
            (passages_idx_file, "passages_index"),
        ]:
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                sizes[name] = size_mb
                total_with_embeddings += size_mb

            else:
                sizes[name] = 0

        sizes["total_with_embeddings"] = total_with_embeddings
        sizes["index_only_mb"] = sizes["index"]  # Just the .index file for fair comparison

        print(f"  📁 Total index size: {total_with_embeddings:.1f} MB")
        print(f"  📁 Index file only: {sizes['index']:.1f} MB")

        return sizes

    def create_compact_index_for_comparison(self, compact_index_path: str) -> dict:
        """Create a compact index for comparison purposes"""
        print("🏗️ Building compact index from existing passages...")

        # Load existing passages from current index

        from leann import LeannBuilder

        current_index_path = Path(self.index_path)
        current_index_dir = current_index_path.parent
        current_index_name = current_index_path.name

        # Read metadata to get passage source
        meta_path = current_index_dir / f"{current_index_name}.meta.json"
        with open(meta_path) as f:
            import json

            meta = json.load(f)

        passage_source = meta["passage_sources"][0]
        passage_file = passage_source["path"]

        # Convert relative path to absolute
        if not Path(passage_file).is_absolute():
            passage_file = current_index_dir / Path(passage_file).name

        print(f"📄 Loading passages from {passage_file}...")

        # Build compact index with same passages
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=meta["embedding_model"],
            embedding_mode=meta.get("embedding_mode", "sentence-transformers"),
            is_recompute=True,  # Enable recompute (no stored embeddings)
            is_compact=True,  # Enable compact storage
            **meta.get("backend_kwargs", {}),
        )

        # Load all passages
        with open(passage_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    builder.add_text(data["text"], metadata=data.get("metadata", {}))

        print(f"🔨 Building compact index at {compact_index_path}...")
        builder.build_index(compact_index_path)

        # Analyze the compact index size
        temp_evaluator = FinanceBenchEvaluator(compact_index_path)
        compact_sizes = temp_evaluator.analyze_index_sizes()
        compact_sizes["index_type"] = "compact"

        return compact_sizes

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
            import json

            meta = json.load(f)

        passage_source = meta["passage_sources"][0]
        passage_file = passage_source["path"]

        # Convert relative path to absolute
        if not Path(passage_file).is_absolute():
            passage_file = current_index_dir / Path(passage_file).name

        print(f"📄 Loading passages from {passage_file}...")

        # Build non-compact index with same passages
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=meta["embedding_model"],
            embedding_mode=meta.get("embedding_mode", "sentence-transformers"),
            is_recompute=False,  # Disable recompute (store embeddings)
            is_compact=False,  # Disable compact storage
            **{
                k: v
                for k, v in meta.get("backend_kwargs", {}).items()
                if k not in ["is_recompute", "is_compact"]
            },
        )

        # Load all passages
        with open(passage_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    builder.add_text(data["text"], metadata=data.get("metadata", {}))

        print(f"🔨 Building non-compact index at {non_compact_index_path}...")
        builder.build_index(non_compact_index_path)

        # Analyze the non-compact index size
        temp_evaluator = FinanceBenchEvaluator(non_compact_index_path)
        non_compact_sizes = temp_evaluator.analyze_index_sizes()
        non_compact_sizes["index_type"] = "non_compact"

        return non_compact_sizes

    def compare_index_performance(
        self, non_compact_path: str, compact_path: str, test_data: list, complexity: int
    ) -> dict:
        """Compare performance between non-compact and compact indexes"""
        print("⚡ Comparing search performance between indexes...")

        import time

        from leann import LeannSearcher

        # Test queries
        test_queries = [item["question"] for item in test_data[:5]]

        results = {
            "non_compact": {"search_times": []},
            "compact": {"search_times": []},
            "avg_search_times": {},
            "speed_ratio": 0.0,
        }

        # Test non-compact index (no recompute)
        print("  🔍 Testing non-compact index (no recompute)...")
        non_compact_searcher = LeannSearcher(non_compact_path)

        for query in test_queries:
            start_time = time.time()
            _ = non_compact_searcher.search(
                query, top_k=3, complexity=complexity, recompute_embeddings=False
            )
            search_time = time.time() - start_time
            results["non_compact"]["search_times"].append(search_time)

        # Test compact index (with recompute)
        print("  🔍 Testing compact index (with recompute)...")
        compact_searcher = LeannSearcher(compact_path)

        for query in test_queries:
            start_time = time.time()
            _ = compact_searcher.search(
                query, top_k=3, complexity=complexity, recompute_embeddings=True
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

    def evaluate_timing_breakdown(
        self, data: list[dict], max_samples: Optional[int] = None
    ) -> dict:
        """Evaluate timing breakdown and accuracy by hacking LeannChat.ask() for separated timing"""
        if not self.chat or not self.openai_client:
            print("⚠️  Skipping timing evaluation (no OpenAI API key provided)")
            return {
                "total_questions": 0,
                "avg_search_time": 0.0,
                "avg_generation_time": 0.0,
                "avg_total_time": 0.0,
                "accuracy": 0.0,
            }

        print("🔍🤖 Evaluating timing breakdown and accuracy (search + generation)...")

        if max_samples:
            data = data[:max_samples]
            print(f"📝 Using first {max_samples} samples for timing evaluation")

        search_times = []
        generation_times = []
        total_times = []
        correct_answers = 0

        for i, item in enumerate(data):
            question = item["question"]
            ground_truth = item["answer"]

            try:
                # Hack: Monkey-patch the ask method to capture internal timing
                original_ask = self.chat.ask
                captured_search_time = None
                captured_generation_time = None

                def patched_ask(*args, **kwargs):
                    nonlocal captured_search_time, captured_generation_time

                    # Time the search part
                    search_start = time.time()
                    results = self.chat.searcher.search(args[0], top_k=3, complexity=64)
                    captured_search_time = time.time() - search_start

                    # Time the generation part
                    context = "\n\n".join([r.text for r in results])
                    prompt = (
                        "Here is some retrieved context that might help answer your question:\n\n"
                        f"{context}\n\n"
                        f"Question: {args[0]}\n\n"
                        "Please provide the best answer you can based on this context and your knowledge."
                    )

                    generation_start = time.time()
                    answer = self.chat.llm.ask(prompt)
                    captured_generation_time = time.time() - generation_start

                    return answer

                # Apply the patch
                self.chat.ask = patched_ask

                # Time the total QA
                total_start = time.time()
                generated_answer = self.chat.ask(question)
                total_time = time.time() - total_start

                # Restore original method
                self.chat.ask = original_ask

                # Store the timings
                search_times.append(captured_search_time)
                generation_times.append(captured_generation_time)
                total_times.append(total_time)

                # Check accuracy using LLM as judge
                is_correct = self._check_answer_accuracy(generated_answer, ground_truth, question)
                if is_correct:
                    correct_answers += 1

                status = "✅" if is_correct else "❌"
                print(
                    f"Question {i + 1}/{len(data)}: {status} Search={captured_search_time:.3f}s, Gen={captured_generation_time:.3f}s, Total={total_time:.3f}s"
                )
                print(f"  GT: {ground_truth}")
                print(f"  Gen: {generated_answer[:100]}...")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                search_times.append(0.0)
                generation_times.append(0.0)
                total_times.append(0.0)

        accuracy = correct_answers / len(data) if data else 0.0

        metrics = {
            "total_questions": len(data),
            "avg_search_time": sum(search_times) / len(search_times) if search_times else 0.0,
            "avg_generation_time": sum(generation_times) / len(generation_times)
            if generation_times
            else 0.0,
            "avg_total_time": sum(total_times) / len(total_times) if total_times else 0.0,
            "accuracy": accuracy,
            "correct_answers": correct_answers,
            "search_times": search_times,
            "generation_times": generation_times,
            "total_times": total_times,
        }

        return metrics

    def _check_answer_accuracy(
        self, generated_answer: str, ground_truth: str, question: str
    ) -> bool:
        """Check if generated answer matches ground truth using LLM as judge"""
        judge_prompt = f"""You are an expert judge evaluating financial question answering.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {generated_answer}

Task: Determine if the generated answer is factually correct compared to the ground truth. Focus on:
1. Numerical accuracy (exact values, units, currency)
2. Key financial concepts and terminology
3. Overall factual correctness

For financial data, small formatting differences are OK (e.g., "$1,577" vs "1577 million" vs "$1.577 billion"), but the core numerical value must match.

Respond with exactly one word: "CORRECT" if the generated answer is factually accurate, or "INCORRECT" if it's wrong or significantly different."""

        try:
            judge_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=10,
                temperature=0,
            )
            if not judge_response.choices or judge_response.choices[0].message is None:
                raise ValueError("LLM returned empty or filtered response")
            judgment = judge_response.choices[0].message.content.strip().upper()
            return judgment == "CORRECT"
        except Exception as e:
            print(f"  ⚠️  Judge error: {e}, falling back to string matching")
            # Fallback to simple string matching
            gen_clean = generated_answer.strip().lower().replace("$", "").replace(",", "")
            gt_clean = ground_truth.strip().lower().replace("$", "").replace(",", "")
            return gt_clean in gen_clean

    def _print_results(self, timing_metrics: dict):
        """Print evaluation results"""
        print("\n🎯 EVALUATION RESULTS")
        print("=" * 50)

        # Index comparison analysis
        if "current_index" in timing_metrics and "non_compact_index" in timing_metrics:
            print("\n📏 Index Comparison Analysis:")
            current = timing_metrics["current_index"]
            non_compact = timing_metrics["non_compact_index"]

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
            print(f"  Total index size: {timing_metrics.get('total_with_embeddings', 0):.1f} MB")

        print("\n📊 Accuracy:")
        print(f"  Accuracy: {timing_metrics.get('accuracy', 0) * 100:.1f}%")
        print(
            f"  Correct Answers: {timing_metrics.get('correct_answers', 0)}/{timing_metrics.get('total_questions', 0)}"
        )

        print("\n📊 Timing Breakdown:")
        print(f"  Total Questions: {timing_metrics.get('total_questions', 0)}")
        print(f"  Avg Search Time: {timing_metrics.get('avg_search_time', 0):.3f}s")
        print(f"  Avg Generation Time: {timing_metrics.get('avg_generation_time', 0):.3f}s")
        print(f"  Avg Total Time: {timing_metrics.get('avg_total_time', 0):.3f}s")

        if timing_metrics.get("avg_total_time", 0) > 0:
            search_pct = (
                timing_metrics.get("avg_search_time", 0)
                / timing_metrics.get("avg_total_time", 1)
                * 100
            )
            gen_pct = (
                timing_metrics.get("avg_generation_time", 0)
                / timing_metrics.get("avg_total_time", 1)
                * 100
            )
            print("\n📈 Time Distribution:")
            print(f"  Search: {search_pct:.1f}%")
            print(f"  Generation: {gen_pct:.1f}%")

    def cleanup(self):
        """Cleanup resources"""
        if self.searcher:
            self.searcher.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Modular FinanceBench Evaluation")
    parser.add_argument("--index", required=True, help="Path to LEANN index")
    parser.add_argument("--dataset", default="data/financebench_merged.jsonl", help="Dataset path")
    parser.add_argument(
        "--stage",
        choices=["2", "3", "4", "all"],
        default="all",
        help="Which stage to run (2=recall, 3=complexity, 4=generation)",
    )
    parser.add_argument("--complexity", type=int, default=None, help="Complexity for search")
    parser.add_argument("--baseline-dir", default="baseline", help="Baseline output directory")
    parser.add_argument("--openai-api-key", help="OpenAI API key for generation evaluation")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument(
        "--llm-backend", choices=["openai", "hf", "vllm"], default="openai", help="LLM backend"
    )
    parser.add_argument("--model-name", default="Qwen3-8B", help="Model name for HF/vLLM")

    args = parser.parse_args()

    try:
        # Check if baseline exists
        baseline_index_path = os.path.join(args.baseline_dir, "faiss_flat.index")
        if not os.path.exists(baseline_index_path):
            print(f"❌ FAISS baseline not found at {baseline_index_path}")
            print("💡 Please run setup_financebench.py first to build the baseline")
            exit(1)

        if args.stage == "2" or args.stage == "all":
            # Stage 2: Recall@3 evaluation
            print("🚀 Starting Stage 2: Recall@3 evaluation")

            evaluator = RecallEvaluator(args.index, args.baseline_dir)

            # Load FinanceBench queries for testing
            print("📖 Loading FinanceBench dataset...")
            queries = []
            with open(args.dataset, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        queries.append(data["question"])

            # Test with more queries for robust measurement
            test_queries = queries[:2000]
            print(f"🧪 Testing with {len(test_queries)} queries")

            # Test with complexity 64
            complexity = 64
            recall = evaluator.evaluate_recall_at_3(test_queries, complexity)
            print(f"📈 Recall@3 at complexity {complexity}: {recall * 100:.1f}%")

            evaluator.cleanup()
            print("✅ Stage 2 completed!\n")

        # Shared non-compact index path for Stage 3 and 4
        non_compact_index_path = args.index.replace(".leann", "_noncompact.leann")
        complexity = args.complexity

        if args.stage == "3" or args.stage == "all":
            # Stage 3: Binary search for 90% recall complexity (using non-compact index for speed)
            print("🚀 Starting Stage 3: Binary search for 90% recall complexity")
            print(
                "💡 Creating non-compact index for fast binary search with recompute_embeddings=False"
            )

            # Create non-compact index for binary search (will be reused in Stage 4)
            print("🏗️ Creating non-compact index for binary search...")
            evaluator = FinanceBenchEvaluator(args.index)
            evaluator.create_non_compact_index_for_comparison(non_compact_index_path)

            # Use non-compact index for binary search
            binary_search_evaluator = RecallEvaluator(non_compact_index_path, args.baseline_dir)

            # Load queries for testing
            print("📖 Loading FinanceBench dataset...")
            queries = []
            with open(args.dataset, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        queries.append(data["question"])

            # Use more queries for robust measurement
            test_queries = queries[:200]
            print(f"🧪 Testing with {len(test_queries)} queries")

            # Binary search for 90% recall complexity (without recompute for speed)
            target_recall = 0.9
            min_complexity, max_complexity = 1, 32

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
                    test_queries, mid_complexity, recompute_embeddings=False
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
                            test_queries, complexity, recompute_embeddings=False
                        )
                        status = "✅" if recall >= target_recall else "❌"
                        print(f"  {status} Complexity {complexity:3d}: {recall * 100:5.1f}%")

                # Now test the optimal complexity with compact index and recompute for comparison
                print(
                    f"\n🔄 Testing optimal complexity {best_complexity} on compact index WITH recompute..."
                )
                compact_evaluator = RecallEvaluator(args.index, args.baseline_dir)
                recall_with_recompute = compact_evaluator.evaluate_recall_at_3(
                    test_queries[:10], best_complexity, recompute_embeddings=True
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
            # Stage 4: Comprehensive evaluation with dual index comparison
            print("🚀 Starting Stage 4: Comprehensive evaluation with dual index comparison")

            # Use FinanceBench evaluator for QA evaluation
            evaluator = FinanceBenchEvaluator(
                args.index, args.openai_api_key if args.llm_backend == "openai" else None
            )

            print("📖 Loading FinanceBench dataset...")
            data = evaluator.load_dataset(args.dataset)

            # Step 1: Analyze current (compact) index
            print("\n📏 Analyzing current index (compact, pruned)...")
            compact_size_metrics = evaluator.analyze_index_sizes()
            compact_size_metrics["index_type"] = "compact"

            # Step 2: Use existing non-compact index or create if needed
            from pathlib import Path

            if Path(non_compact_index_path).exists():
                print(
                    f"\n📁 Using existing non-compact index from Stage 3: {non_compact_index_path}"
                )
                temp_evaluator = FinanceBenchEvaluator(non_compact_index_path)
                non_compact_size_metrics = temp_evaluator.analyze_index_sizes()
                non_compact_size_metrics["index_type"] = "non_compact"
            else:
                print("\n🏗️ Creating non-compact index (with embeddings) for comparison...")
                non_compact_size_metrics = evaluator.create_non_compact_index_for_comparison(
                    non_compact_index_path
                )

            # Step 3: Compare index sizes
            print("\n📊 Index size comparison:")
            print(
                f"  Compact index (current): {compact_size_metrics['total_with_embeddings']:.1f} MB"
            )
            print(
                f"  Non-compact index: {non_compact_size_metrics['total_with_embeddings']:.1f} MB"
            )
            print("\n📊 Index-only size comparison (.index file only):")
            print(f"  Compact index: {compact_size_metrics['index_only_mb']:.1f} MB")
            print(f"  Non-compact index: {non_compact_size_metrics['index_only_mb']:.1f} MB")
            # Use index-only size for fair comparison (same as Enron emails)
            storage_saving = (
                (non_compact_size_metrics["index_only_mb"] - compact_size_metrics["index_only_mb"])
                / non_compact_size_metrics["index_only_mb"]
                * 100
            )
            print(f"  Storage saving by compact: {storage_saving:.1f}%")

            # Step 4: Performance comparison between the two indexes
            if complexity is None:
                raise ValueError("Complexity is required for performance comparison")

            print("\n⚡ Performance comparison between indexes...")
            performance_metrics = evaluator.compare_index_performance(
                non_compact_index_path, args.index, data[:10], complexity=complexity
            )

            # Step 5: Generation evaluation
            test_samples = 20
            print(f"\n🧪 Testing with first {test_samples} samples for generation analysis")

            if args.llm_backend == "openai" and args.openai_api_key:
                print("🔍🤖 Running OpenAI-based generation evaluation...")
                evaluation_start = time.time()
                timing_metrics = evaluator.evaluate_timing_breakdown(data[:test_samples])
                evaluation_time = time.time() - evaluation_start
            else:
                print(
                    f"🔍🤖 Running {args.llm_backend} generation evaluation with {args.model_name}..."
                )
                try:
                    # Load LLM
                    if args.llm_backend == "hf":
                        tokenizer, model = load_hf_model(args.model_name)

                        def llm_func(prompt):
                            return generate_hf(tokenizer, model, prompt)
                    else:  # vllm
                        llm, sampling_params = load_vllm_model(args.model_name)

                        def llm_func(prompt):
                            return generate_vllm(llm, sampling_params, prompt)

                    # Simple generation evaluation
                    queries = [item["question"] for item in data[:test_samples]]
                    gen_results = evaluate_rag(
                        evaluator.searcher,
                        llm_func,
                        queries,
                        domain="finance",
                        complexity=complexity,
                    )

                    timing_metrics = {
                        "total_questions": len(queries),
                        "avg_search_time": gen_results["avg_search_time"],
                        "avg_generation_time": gen_results["avg_generation_time"],
                        "results": gen_results["results"],
                    }
                    evaluation_time = time.time()

                except Exception as e:
                    print(f"❌ Generation evaluation failed: {e}")
                    timing_metrics = {
                        "total_questions": 0,
                        "avg_search_time": 0,
                        "avg_generation_time": 0,
                    }
                    evaluation_time = 0

            # Combine all metrics
            combined_metrics = {
                **timing_metrics,
                "total_evaluation_time": evaluation_time,
                "current_index": compact_size_metrics,
                "non_compact_index": non_compact_size_metrics,
                "performance_comparison": performance_metrics,
                "storage_saving_percent": storage_saving,
            }

            # Print results
            print("\n📊 Generation Results:")
            print(f"  Total Questions: {timing_metrics.get('total_questions', 0)}")
            print(f"  Avg Search Time: {timing_metrics.get('avg_search_time', 0):.3f}s")
            print(f"  Avg Generation Time: {timing_metrics.get('avg_generation_time', 0):.3f}s")

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
            print("  Stage 2: ✅ Recall@3 evaluation completed")
            print("  Stage 3: ✅ Optimal complexity found")
            print("  Stage 4: ✅ Generation accuracy & timing evaluation completed")
            print("\n🔧 Recommended next steps:")
            print("  - Use optimal complexity for best speed/accuracy balance")
            print("  - Review accuracy and timing breakdown for performance optimization")
            print("  - Run full evaluation on complete dataset if needed")

            # Clean up non-compact index after all stages complete
            print("\n🧹 Cleaning up temporary non-compact index...")
            from pathlib import Path

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
        exit(1)


if __name__ == "__main__":
    main()
