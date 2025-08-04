# python embedd_micro.py --use_int8 Fastest

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, BitsAndBytesConfig


@dataclass
class BenchmarkConfig:
    model_path: str
    batch_sizes: list[int]
    seq_length: int
    num_runs: int
    use_fp16: bool = True
    use_int4: bool = False
    use_int8: bool = False  # Add this parameter
    use_cuda_graphs: bool = False
    use_flash_attention: bool = False
    use_linear8bitlt: bool = False


class GraphContainer:
    """Container for managing graphs for different batch sizes (CUDA graphs on NVIDIA, regular on others)."""

    def __init__(self, model: nn.Module, seq_length: int):
        self.model = model
        self.seq_length = seq_length
        self.graphs: dict[int, GraphWrapper] = {}

    def get_or_create(self, batch_size: int) -> "GraphWrapper":
        if batch_size not in self.graphs:
            self.graphs[batch_size] = GraphWrapper(self.model, batch_size, self.seq_length)
        return self.graphs[batch_size]


class GraphWrapper:
    """Wrapper for graph capture and replay (CUDA graphs on NVIDIA, regular on others)."""

    def __init__(self, model: nn.Module, batch_size: int, seq_length: int):
        self.model = model
        self.device = self._get_device()
        self.static_input = self._create_random_batch(batch_size, seq_length)
        self.static_attention_mask = torch.ones_like(self.static_input)

        # Warm up
        self._warmup()

        # Only use CUDA graphs on NVIDIA GPUs
        if torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph"):
            # Capture graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self.model(
                    input_ids=self.static_input,
                    attention_mask=self.static_attention_mask,
                )
            self.use_cuda_graph = True
        else:
            # For MPS or CPU, just store the model
            self.use_cuda_graph = False
            self.static_output = None

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _create_random_batch(self, batch_size: int, seq_length: int) -> torch.Tensor:
        return torch.randint(
            0, 1000, (batch_size, seq_length), device=self.device, dtype=torch.long
        )

    def _warmup(self, num_warmup: int = 3):
        with torch.no_grad():
            for _ in range(num_warmup):
                self.model(
                    input_ids=self.static_input,
                    attention_mask=self.static_attention_mask,
                )

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_graph:
            self.static_input.copy_(input_ids)
            self.static_attention_mask.copy_(attention_mask)
            self.graph.replay()
            return self.static_output
        else:
            # For MPS/CPU, just run normally
            return self.model(input_ids=input_ids, attention_mask=attention_mask)


class ModelOptimizer:
    """Applies various optimizations to the model."""

    @staticmethod
    def optimize(model: nn.Module, config: BenchmarkConfig) -> nn.Module:
        print("\nApplying model optimizations:")

        if model is None:
            raise ValueError("Cannot optimize None model")

        # Move to GPU
        if torch.cuda.is_available():
            model = model.cuda()
            device = "cuda"
        elif torch.backends.mps.is_available():
            model = model.to("mps")
            device = "mps"
        else:
            model = model.cpu()
            device = "cpu"
        print(f"- Model moved to {device}")

        # FP16
        if config.use_fp16 and not config.use_int4:
            model = model.half()
            # use torch compile
            model = torch.compile(model)
            print("- Using FP16 precision")

        # Check if using SDPA (only on CUDA)
        if (
            torch.cuda.is_available()
            and torch.version.cuda
            and float(torch.version.cuda[:3]) >= 11.6
        ):
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                print("- Using PyTorch SDPA (scaled_dot_product_attention)")
            else:
                print("- PyTorch SDPA not available")

        # Flash Attention (only on CUDA)
        if config.use_flash_attention and torch.cuda.is_available():
            try:
                from flash_attn.flash_attention import FlashAttention  # noqa: F401

                print("- Flash Attention 2 available")
                if hasattr(model.config, "attention_mode"):
                    model.config.attention_mode = "flash_attention_2"
                    print("  - Enabled Flash Attention 2 mode")
            except ImportError:
                print("- Flash Attention not available")

        # Memory efficient attention (only on CUDA)
        if torch.cuda.is_available():
            try:
                from xformers.ops import memory_efficient_attention  # noqa: F401

                if hasattr(model, "enable_xformers_memory_efficient_attention"):
                    model.enable_xformers_memory_efficient_attention()
                    print("- Enabled xformers memory efficient attention")
                else:
                    print("- Model doesn't support xformers")
            except (ImportError, AttributeError):
                print("- Xformers not available")

        model.eval()
        print("- Model set to eval mode")

        return model


class Timer:
    """Handles accurate GPU timing using GPU events or CPU timing."""

    def __init__(self):
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.use_gpu_timing = True
        elif torch.backends.mps.is_available():
            # MPS doesn't have events, use CPU timing
            self.use_gpu_timing = False
        else:
            # CPU timing
            self.use_gpu_timing = False

    @contextmanager
    def timing(self):
        if self.use_gpu_timing:
            self.start_event.record()
            yield
            self.end_event.record()
            self.end_event.synchronize()
        else:
            # Use CPU timing for MPS/CPU
            start_time = time.time()
            yield
            self.cpu_elapsed = time.time() - start_time

    def elapsed_time(self) -> float:
        if self.use_gpu_timing:
            return self.start_event.elapsed_time(self.end_event) / 1000  # ms to seconds
        else:
            return self.cpu_elapsed


class Benchmark:
    """Main benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        try:
            self.model = self._load_model()
            if self.model is None:
                raise ValueError("Model initialization failed - model is None")

            # Only use CUDA graphs on NVIDIA GPUs
            if config.use_cuda_graphs and torch.cuda.is_available():
                self.graphs = GraphContainer(self.model, config.seq_length)
            else:
                self.graphs = None
            self.timer = Timer()
        except Exception as e:
            print(f"ERROR in benchmark initialization: {e!s}")
            raise

    def _load_model(self) -> nn.Module:
        print(f"Loading model from {self.config.model_path}...")

        try:
            # Int4 quantization using HuggingFace integration
            if self.config.use_int4:
                import bitsandbytes as bnb

                print(f"- bitsandbytes version: {bnb.__version__}")

                # Check if using custom 8bit quantization
                if hasattr(self.config, "use_linear8bitlt") and self.config.use_linear8bitlt:
                    print("- Using custom Linear8bitLt replacement for all linear layers")

                    # Load original model (without quantization config)
                    import bitsandbytes as bnb
                    import torch

                    # set default to half
                    torch.set_default_dtype(torch.float16)
                    compute_dtype = torch.float16 if self.config.use_fp16 else torch.float32
                    model = AutoModel.from_pretrained(
                        self.config.model_path,
                        torch_dtype=compute_dtype,
                    )

                    # Define replacement function
                    def replace_linear_with_linear8bitlt(model):
                        """Recursively replace all nn.Linear layers with Linear8bitLt"""
                        for name, module in list(model.named_children()):
                            if isinstance(module, nn.Linear):
                                # Get original linear layer parameters
                                in_features = module.in_features
                                out_features = module.out_features
                                bias = module.bias is not None

                                # Create 8bit linear layer
                                # print size
                                print(f"in_features: {in_features}, out_features: {out_features}")
                                new_module = bnb.nn.Linear8bitLt(
                                    in_features,
                                    out_features,
                                    bias=bias,
                                    has_fp16_weights=False,
                                )

                                # Copy weights and bias
                                new_module.weight.data = module.weight.data
                                if bias:
                                    new_module.bias.data = module.bias.data

                                # Replace module
                                setattr(model, name, new_module)
                            else:
                                # Process child modules recursively
                                replace_linear_with_linear8bitlt(module)

                        return model

                    # Replace all linear layers
                    model = replace_linear_with_linear8bitlt(model)
                    # add torch compile
                    model = torch.compile(model)

                    # Move model to GPU (quantization happens here)
                    device = (
                        "cuda"
                        if torch.cuda.is_available()
                        else "mps"
                        if torch.backends.mps.is_available()
                        else "cpu"
                    )
                    model = model.to(device)

                    print("- All linear layers replaced with Linear8bitLt")

                else:
                    # Use original Int4 quantization method
                    print("- Using bitsandbytes for Int4 quantization")

                    # Create quantization config

                    compute_dtype = torch.float16 if self.config.use_fp16 else torch.float32
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )

                    print("- Quantization config:", quantization_config)

                    # Load model directly with quantization config
                    model = AutoModel.from_pretrained(
                        self.config.model_path,
                        quantization_config=quantization_config,
                        torch_dtype=compute_dtype,
                        device_map="auto",  # Let HF decide on device mapping
                    )

                # Check if model loaded successfully
                if model is None:
                    raise ValueError("Model loading returned None")

                print(f"- Model type: {type(model)}")

                # Apply optimizations directly here
                print("\nApplying model optimizations:")

                if hasattr(self.config, "use_linear8bitlt") and self.config.use_linear8bitlt:
                    print("- Model moved to GPU with Linear8bitLt quantization")
                else:
                    # Skip moving to GPU since device_map="auto" already did that
                    print("- Model already on GPU due to device_map='auto'")

                # Skip FP16 conversion since we specified compute_dtype
                print(f"- Using {compute_dtype} for compute dtype")

                # Check CUDA and SDPA
                if (
                    torch.cuda.is_available()
                    and torch.version.cuda
                    and float(torch.version.cuda[:3]) >= 11.6
                ):
                    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                        print("- Using PyTorch SDPA (scaled_dot_product_attention)")
                    else:
                        print("- PyTorch SDPA not available")

                # Try xformers if available (only on CUDA)
                if torch.cuda.is_available():
                    try:
                        if hasattr(model, "enable_xformers_memory_efficient_attention"):
                            model.enable_xformers_memory_efficient_attention()
                            print("- Enabled xformers memory efficient attention")
                        else:
                            print("- Model doesn't support xformers")
                    except (ImportError, AttributeError):
                        print("- Xformers not available")

                # Set to eval mode
                model.eval()
                print("- Model set to eval mode")
            # Int8 quantization using HuggingFace integration
            elif self.config.use_int8:
                print("- Using INT8 quantization")
                # For now, just use standard loading with INT8 config
                compute_dtype = torch.float16 if self.config.use_fp16 else torch.float32
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )

                model = AutoModel.from_pretrained(
                    self.config.model_path,
                    quantization_config=quantization_config,
                    torch_dtype=compute_dtype,
                    device_map="auto",
                )

                if model is None:
                    raise ValueError("Model loading returned None")

                print(f"- Model type: {type(model)}")
                model.eval()
                print("- Model set to eval mode")

            else:
                # Standard loading for FP16/FP32
                model = AutoModel.from_pretrained(self.config.model_path)
                print("- Model loaded in standard precision")
                print(f"- Model type: {type(model)}")

                # Apply standard optimizations
                # set default to half
                import torch

                torch.set_default_dtype(torch.bfloat16)
                model = ModelOptimizer.optimize(model, self.config)
                model = model.half()
                # add torch compile
                model = torch.compile(model)

            # Final check to ensure model is not None
            if model is None:
                raise ValueError("Model is None after optimization")

            print(f"- Final model type: {type(model)}")
            return model

        except Exception as e:
            print(f"ERROR loading model: {e!s}")
            import traceback

            traceback.print_exc()
            raise

    def _create_random_batch(self, batch_size: int) -> torch.Tensor:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return torch.randint(
            0,
            1000,
            (batch_size, self.config.seq_length),
            device=device,
            dtype=torch.long,
        )

    def _run_inference(
        self, input_ids: torch.Tensor, graph_wrapper: GraphWrapper | None = None
    ) -> tuple[float, torch.Tensor]:
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad(), self.timer.timing():
            if graph_wrapper is not None:
                output = graph_wrapper(input_ids, attention_mask)
            else:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return self.timer.elapsed_time(), output

    def run(self) -> dict[int, dict[str, float]]:
        results = {}

        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elif torch.backends.mps.is_available():
            # MPS doesn't have reset_peak_memory_stats, skip it
            pass
        else:
            print("- No GPU memory stats available")

        for batch_size in self.config.batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            times = []

            # Get or create graph for this batch size
            graph_wrapper = (
                self.graphs.get_or_create(batch_size) if self.graphs is not None else None
            )

            # Pre-allocate input tensor
            input_ids = self._create_random_batch(batch_size)
            print(f"Input shape: {input_ids.shape}")

            # Run benchmark
            for i in tqdm(range(self.config.num_runs), desc=f"Batch size {batch_size}"):
                try:
                    elapsed_time, output = self._run_inference(input_ids, graph_wrapper)
                    if i == 0:  # Only print on first run
                        print(f"Output shape: {output.last_hidden_state.shape}")
                    times.append(elapsed_time)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    break

            if not times:
                print(f"No successful runs for batch size {batch_size}, skipping")
                continue

            # Calculate statistics
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

        # Log memory usage
        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        elif torch.backends.mps.is_available():
            # MPS doesn't have max_memory_allocated, use 0
            peak_memory_gb = 0.0
        else:
            peak_memory_gb = 0.0
            print("- No GPU memory usage available")

        if peak_memory_gb > 0:
            print(f"\nPeak GPU memory usage: {peak_memory_gb:.2f} GB")
        else:
            print("\n- GPU memory usage not available")

        # Add memory info to results
        for batch_size in results:
            results[batch_size]["peak_memory_gb"] = peak_memory_gb

        return results


def main():
    parser = argparse.ArgumentParser(description="Model Inference Benchmark")
    parser.add_argument(
        "--model_path",
        type=str,
        default="facebook/contriever",
        help="Path to the model",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of batch sizes",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=256,
        help="Sequence length for input",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs for each batch size",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Enable FP16 inference",
    )
    parser.add_argument(
        "--use_int4",
        action="store_true",
        help="Enable INT4 quantization using bitsandbytes",
    )
    parser.add_argument(
        "--use_int8",
        action="store_true",
        help="Enable INT8 quantization for both activations and weights using bitsandbytes",
    )
    parser.add_argument(
        "--use_cuda_graphs",
        action="store_true",
        help="Enable CUDA Graphs optimization (only on NVIDIA GPUs)",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Enable Flash Attention 2 if available (only on NVIDIA GPUs)",
    )
    parser.add_argument(
        "--use_linear8bitlt",
        action="store_true",
        help="Enable Linear8bitLt quantization for all linear layers",
    )

    args = parser.parse_args()

    # Print arguments for debugging
    print("\nCommand line arguments:")
    for arg, value in vars(args).items():
        print(f"- {arg}: {value}")

    config = BenchmarkConfig(
        model_path=args.model_path,
        batch_sizes=[int(bs) for bs in args.batch_sizes.split(",")],
        seq_length=args.seq_length,
        num_runs=args.num_runs,
        use_fp16=args.use_fp16,
        use_int4=args.use_int4,
        use_int8=args.use_int8,  # Add this line
        use_cuda_graphs=args.use_cuda_graphs,
        use_flash_attention=args.use_flash_attention,
        use_linear8bitlt=args.use_linear8bitlt,
    )

    # Print configuration for debugging
    print("\nBenchmark configuration:")
    for field, value in vars(config).items():
        print(f"- {field}: {value}")

    try:
        benchmark = Benchmark(config)
        results = benchmark.run()

        # Save results to file
        import json
        import os

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Generate filename based on configuration
        precision_type = (
            "int4"
            if config.use_int4
            else "int8"
            if config.use_int8
            else "fp16"
            if config.use_fp16
            else "fp32"
        )
        model_name = os.path.basename(config.model_path)
        output_file = f"results/benchmark_{model_name}_{precision_type}.json"

        # Save results
        with open(output_file, "w") as f:
            json.dump(
                {
                    "config": {
                        k: str(v) if isinstance(v, list) else v for k, v in vars(config).items()
                    },
                    "results": {str(k): v for k, v in results.items()},
                },
                f,
                indent=2,
            )
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
