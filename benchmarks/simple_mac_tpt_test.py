import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel

# Add MLX imports
try:
    import mlx.core as mx
    from mlx_lm.utils import load

    MLX_AVAILABLE = True
except ImportError:
    print("MLX not available. Install with: uv pip install mlx mlx-lm")
    MLX_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    model_path: str = "facebook/contriever-msmarco"
    batch_sizes: list[int] = None
    seq_length: int = 256
    num_runs: int = 5
    use_fp16: bool = True
    use_int4: bool = False
    use_int8: bool = False
    use_cuda_graphs: bool = False
    use_flash_attention: bool = False
    use_linear8bitlt: bool = False
    use_mlx: bool = False  # New flag for MLX testing

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


class MLXBenchmark:
    """MLX-specific benchmark for embedding models"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        """Load MLX model and tokenizer following the API pattern"""
        print(f"Loading MLX model from {self.config.model_path}...")
        try:
            model, tokenizer = load(self.config.model_path)
            print("MLX model loaded successfully")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading MLX model: {e}")
            raise

    def _create_random_batch(self, batch_size: int):
        """Create random input batches for MLX testing - same as PyTorch"""
        return torch.randint(0, 1000, (batch_size, self.config.seq_length), dtype=torch.long)

    def _run_inference(self, input_ids: torch.Tensor) -> float:
        """Run MLX inference with same input as PyTorch"""
        start_time = time.time()
        try:
            # Convert PyTorch tensor to MLX array
            input_ids_mlx = mx.array(input_ids.numpy())

            # Get embeddings
            embeddings = self.model(input_ids_mlx)

            # Mean pooling (following the API pattern)
            pooled = embeddings.mean(axis=1)

            # Convert to numpy (following the API pattern)
            pooled_numpy = np.array(pooled.tolist(), dtype=np.float32)

            # Force computation
            _ = pooled_numpy.shape

        except Exception as e:
            print(f"MLX inference error: {e}")
            return float("inf")
        end_time = time.time()

        return end_time - start_time

    def run(self) -> dict[int, dict[str, float]]:
        """Run the MLX benchmark across all batch sizes"""
        results = {}

        print(f"Starting MLX benchmark with model: {self.config.model_path}")
        print(f"Testing batch sizes: {self.config.batch_sizes}")

        for batch_size in self.config.batch_sizes:
            print(f"\n=== Testing MLX batch size: {batch_size} ===")
            times = []

            # Create input batch (same as PyTorch)
            input_ids = self._create_random_batch(batch_size)

            # Warm up
            print("Warming up...")
            for _ in range(3):
                try:
                    self._run_inference(input_ids[:2])  # Warm up with smaller batch
                except Exception as e:
                    print(f"Warmup error: {e}")
                    break

            # Run benchmark
            for _i in tqdm(range(self.config.num_runs), desc=f"MLX Batch size {batch_size}"):
                try:
                    elapsed_time = self._run_inference(input_ids)
                    if elapsed_time != float("inf"):
                        times.append(elapsed_time)
                except Exception as e:
                    print(f"Error during MLX inference: {e}")
                    break

            if not times:
                print(f"Skipping batch size {batch_size} due to errors")
                continue

            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time

            results[batch_size] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "throughput": throughput,
                "min_time": np.min(times),
                "max_time": np.max(times),
            }

            print(f"MLX Results for batch size {batch_size}:")
            print(f"  Avg Time: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"  Min Time: {np.min(times):.4f}s")
            print(f"  Max Time: {np.max(times):.4f}s")
            print(f"  Throughput: {throughput:.2f} sequences/second")

        return results


class Benchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        print(f"Loading model from {self.config.model_path}...")

        model = AutoModel.from_pretrained(self.config.model_path)
        if self.config.use_fp16:
            model = model.half()
        model = torch.compile(model)
        model = model.to(self.device)

        model.eval()
        return model

    def _create_random_batch(self, batch_size: int) -> torch.Tensor:
        return torch.randint(
            0,
            1000,
            (batch_size, self.config.seq_length),
            device=self.device,
            dtype=torch.long,
        )

    def _run_inference(self, input_ids: torch.Tensor) -> float:
        attention_mask = torch.ones_like(input_ids)
        # print shape of input_ids and attention_mask
        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")
        start_time = time.time()
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        end_time = time.time()

        return end_time - start_time

    def run(self) -> dict[int, dict[str, float]]:
        results = {}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for batch_size in self.config.batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            times = []

            input_ids = self._create_random_batch(batch_size)

            for _i in tqdm(range(self.config.num_runs), desc=f"Batch size {batch_size}"):
                try:
                    elapsed_time = self._run_inference(input_ids)
                    times.append(elapsed_time)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    break

            if not times:
                continue

            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time

            results[batch_size] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "throughput": throughput,
            }

            print(f"Avg Time: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"Throughput: {throughput:.2f} sequences/second")

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            peak_memory_gb = 0.0

        for batch_size in results:
            results[batch_size]["peak_memory_gb"] = peak_memory_gb

        return results


def run_benchmark():
    """Main function to run the benchmark with optimized parameters."""
    config = BenchmarkConfig()

    try:
        benchmark = Benchmark(config)
        results = benchmark.run()

        max_throughput = max(results[batch_size]["throughput"] for batch_size in results)
        avg_throughput = np.mean([results[batch_size]["throughput"] for batch_size in results])

        return {
            "max_throughput": max_throughput,
            "avg_throughput": avg_throughput,
            "results": results,
        }

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return {"max_throughput": 0.0, "avg_throughput": 0.0, "error": str(e)}


def run_mlx_benchmark():
    """Run MLX-specific benchmark"""
    if not MLX_AVAILABLE:
        print("MLX not available, skipping MLX benchmark")
        return {
            "max_throughput": 0.0,
            "avg_throughput": 0.0,
            "error": "MLX not available",
        }

    config = BenchmarkConfig(model_path="mlx-community/all-MiniLM-L6-v2-4bit", use_mlx=True)

    try:
        benchmark = MLXBenchmark(config)
        results = benchmark.run()

        if not results:
            return {
                "max_throughput": 0.0,
                "avg_throughput": 0.0,
                "error": "No valid results",
            }

        max_throughput = max(results[batch_size]["throughput"] for batch_size in results)
        avg_throughput = np.mean([results[batch_size]["throughput"] for batch_size in results])

        return {
            "max_throughput": max_throughput,
            "avg_throughput": avg_throughput,
            "results": results,
        }

    except Exception as e:
        print(f"MLX benchmark failed: {e}")
        return {"max_throughput": 0.0, "avg_throughput": 0.0, "error": str(e)}


if __name__ == "__main__":
    print("=== PyTorch Benchmark ===")
    pytorch_result = run_benchmark()
    print(f"PyTorch Max throughput: {pytorch_result['max_throughput']:.2f} sequences/second")
    print(f"PyTorch Average throughput: {pytorch_result['avg_throughput']:.2f} sequences/second")

    print("\n=== MLX Benchmark ===")
    mlx_result = run_mlx_benchmark()
    print(f"MLX Max throughput: {mlx_result['max_throughput']:.2f} sequences/second")
    print(f"MLX Average throughput: {mlx_result['avg_throughput']:.2f} sequences/second")

    # Compare results
    if pytorch_result["max_throughput"] > 0 and mlx_result["max_throughput"] > 0:
        speedup = mlx_result["max_throughput"] / pytorch_result["max_throughput"]
        print("\n=== Comparison ===")
        print(f"MLX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch")
