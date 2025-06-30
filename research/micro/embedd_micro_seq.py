import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoModel
from tqdm import tqdm
from contextlib import contextmanager
import math

@dataclass
class BenchmarkConfig:
    model_path: str
    batch_sizes: List[int]
    seq_length: int
    num_runs: int
    use_fp16: bool = True
    use_cuda_graphs: bool = False
    use_flash_attention: bool = False
    max_batch_size: int = 256  # Maximum batch size before splitting


class CUDAGraphContainer:
    """Container for managing CUDA graphs for different batch sizes."""
    
    def __init__(self, model: nn.Module, seq_length: int, max_batch_size: int):
        self.model = model
        self.seq_length = seq_length
        self.max_batch_size = max_batch_size
        self.graphs: Dict[int, CUDAGraphWrapper] = {}
    
    def get_or_create(self, batch_size: int) -> 'CUDAGraphWrapper':
        # For CUDA graphs, we always use the actual batch size or max_batch_size
        effective_batch_size = min(batch_size, self.max_batch_size)
        
        if effective_batch_size not in self.graphs:
            self.graphs[effective_batch_size] = CUDAGraphWrapper(
                self.model, effective_batch_size, self.seq_length
            )
        return self.graphs[effective_batch_size]


class CUDAGraphWrapper:
    """Wrapper for CUDA graph capture and replay."""
    
    def __init__(self, model: nn.Module, batch_size: int, seq_length: int):
        self.model = model
        self.static_input = self._create_random_batch(batch_size, seq_length)
        self.static_attention_mask = torch.ones_like(self.static_input)
        
        # Warm up
        self._warmup()
        
        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(
                input_ids=self.static_input,
                attention_mask=self.static_attention_mask
            )
    
    def _create_random_batch(self, batch_size: int, seq_length: int) -> torch.Tensor:
        return torch.randint(
            0, 1000, (batch_size, seq_length), 
            device="cuda", 
            dtype=torch.long
        )
    
    def _warmup(self, num_warmup: int = 3):
        with torch.no_grad():
            for _ in range(num_warmup):
                self.model(
                    input_ids=self.static_input,
                    attention_mask=self.static_attention_mask
                )
    
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.static_input.copy_(input_ids)
        self.static_attention_mask.copy_(attention_mask)
        self.graph.replay()
        return self.static_output


class ModelOptimizer:
    """Applies various optimizations to the model."""
    
    @staticmethod
    def optimize(model: nn.Module, config: BenchmarkConfig) -> nn.Module:
        print("\nApplying model optimizations:")
        
        # Move to GPU
        model = model.cuda()
        print("- Model moved to GPU")
        
        # FP16
        if config.use_fp16:
            model = model.half()
            print("- Using FP16 precision")
        
        # Check if using SDPA
        if torch.version.cuda and float(torch.version.cuda[:3]) >= 11.6:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("- Using PyTorch SDPA (scaled_dot_product_attention)")
                # No need to do anything as it's automatically enabled
            else:
                print("- PyTorch SDPA not available")
        
        # Flash Attention
        if config.use_flash_attention:
            try:
                from flash_attn.flash_attention import FlashAttention
                print("- Flash Attention 2 available")
                if hasattr(model.config, "attention_mode"):
                    model.config.attention_mode = "flash_attention_2"
                    print("  - Enabled Flash Attention 2 mode")
            except ImportError:
                print("- Flash Attention not available")
        
        # Optimize LayerNorm
        try:
            num_layernorms = 0
            for module in model.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.forward = torch.jit.script(module.forward)
                    num_layernorms += 1
            if num_layernorms > 0:
                print(f"- Optimized {num_layernorms} LayerNorm modules with TorchScript")
        except Exception as e:
            print(f"- LayerNorm optimization failed: {e}")
        
        # Memory efficient attention
        try:
            from xformers.ops import memory_efficient_attention
            model.enable_xformers_memory_efficient_attention()
            print("- Enabled xformers memory efficient attention")
        except (ImportError, AttributeError):
            print("- Xformers not available")
        
        model.eval()
        print("- Model set to eval mode")
        
        return model


class Timer:
    """Handles accurate GPU timing using CUDA events."""
    
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    @contextmanager
    def timing(self):
        self.start_event.record()
        yield
        self.end_event.record()
        self.end_event.synchronize()
    
    def elapsed_time(self) -> float:
        return self.start_event.elapsed_time(self.end_event) / 1000  # ms to seconds


class Benchmark:
    """Main benchmark runner."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = self._load_model()
        self.cuda_graphs = (
            CUDAGraphContainer(self.model, config.seq_length, config.max_batch_size)
            if config.use_cuda_graphs
            else None
        )
        self.timer = Timer()
    
    def _load_model(self) -> nn.Module:
        print(f"Loading model from {self.config.model_path}...")
        model = AutoModel.from_pretrained(self.config.model_path)
        return ModelOptimizer.optimize(model, self.config)
    
    def _create_random_batch(self, batch_size: int) -> torch.Tensor:
        return torch.randint(
            0, 1000,
            (batch_size, self.config.seq_length),
            device="cuda",
            dtype=torch.long
        )
    
    def _run_inference(
        self,
        input_ids: torch.Tensor,
        cuda_graph_wrapper: Optional[CUDAGraphWrapper] = None
    ) -> Tuple[float, torch.Tensor]:
        attention_mask = torch.ones_like(input_ids)
        original_batch_size = input_ids.shape[0]
        print(f"Original input_ids shape: {input_ids.shape}")
        
        # Split large batches to avoid OOM
        max_batch_size = self.config.max_batch_size
        if original_batch_size > max_batch_size:
            print(f"Splitting batch of size {original_batch_size} into chunks of {max_batch_size}")
            total_time = 0
            outputs = []
            
            with torch.no_grad():
                for i in range(0, original_batch_size, max_batch_size):
                    end_idx = min(i + max_batch_size, original_batch_size)
                    batch_slice = input_ids[i:end_idx]
                    mask_slice = attention_mask[i:end_idx]
                    
                    print(f"Processing chunk {i//max_batch_size + 1}: shape {batch_slice.shape}")
                    
                    # Use CUDA graph if available (with the smaller batch size)
                    chunk_cuda_graph = None
                    if cuda_graph_wrapper is not None:
                        chunk_cuda_graph = self.cuda_graphs.get_or_create(batch_slice.shape[0])
                    
                    with self.timer.timing():
                        if chunk_cuda_graph is not None:
                            chunk_output = chunk_cuda_graph(batch_slice, mask_slice)
                        else:
                            chunk_output = self.model(input_ids=batch_slice, attention_mask=mask_slice)
                    
                    total_time += self.timer.elapsed_time()
                    outputs.append(chunk_output.last_hidden_state)
                
                # Combine outputs
                combined_output = torch.cat(outputs, dim=0)
                print(f"Combined output shape: {combined_output.shape}")
                
                # Create a wrapper object similar to model output to maintain consistency
                class DummyOutput:
                    def __init__(self, hidden_states):
                        self.last_hidden_state = hidden_states
                
                output = DummyOutput(combined_output)
                return total_time, output
        else:
            # Process normally for small batches
            with torch.no_grad(), self.timer.timing():
                if cuda_graph_wrapper is not None:
                    output = cuda_graph_wrapper(input_ids, attention_mask)
                else:
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            print(f"Output shape: {output.last_hidden_state.shape}")
            return self.timer.elapsed_time(), output
    
    def run(self) -> Dict[int, Dict[str, float]]:
        results = {}
        
        for batch_size in self.config.batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            times = []
            
            # Get or create CUDA graph for this batch size
            cuda_graph_wrapper = None
            if self.cuda_graphs is not None:
                if batch_size <= self.config.max_batch_size:
                    cuda_graph_wrapper = self.cuda_graphs.get_or_create(batch_size)
                else:
                    # For large batches, we'll use the max_batch_size graph in chunks
                    cuda_graph_wrapper = True  # Just a flag to indicate we want to use CUDA graphs
            
            # Pre-allocate input tensor
            input_ids = self._create_random_batch(batch_size)
            
            # Run benchmark
            for run_idx in tqdm(range(self.config.num_runs), desc=f"Batch size {batch_size}"):
                elapsed_time, _ = self._run_inference(input_ids, cuda_graph_wrapper)
                times.append(elapsed_time)
                print(f"Run {run_idx+1}: {elapsed_time:.4f}s")
            
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
        default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096",
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
        "--no_fp16",
        action="store_true",
        help="Disable FP16 inference",
    )
    parser.add_argument(
        "--use_cuda_graphs",
        action="store_true",
        help="Enable CUDA Graphs optimization",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Enable Flash Attention 2 if available",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=256,
        help="Maximum batch size before splitting to prevent OOM",
    )
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_path=args.model_path,
        batch_sizes=[int(bs) for bs in args.batch_sizes.split(",")],
        seq_length=args.seq_length,
        num_runs=args.num_runs,
        use_fp16=not args.no_fp16,
        use_cuda_graphs=args.use_cuda_graphs,
        use_flash_attention=args.use_flash_attention,
        max_batch_size=args.max_batch_size,
    )
    
    benchmark = Benchmark(config)
    results = benchmark.run()
    
    # Print overall summary
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Model: {config.model_path}")
    print(f"Sequence Length: {config.seq_length}")
    print(f"FP16: {config.use_fp16}")
    print(f"CUDA Graphs: {config.use_cuda_graphs}")
    print(f"Flash Attention: {config.use_flash_attention}")
    print(f"Max Batch Size: {config.max_batch_size}")
    print("\nResults:")
    
    print("\nBatch Size | Avg Time (s) | Throughput (seq/s)")
    print("-" * 50)
    for bs in sorted(results.keys()):
        r = results[bs]
        print(f"{bs:^10} | {r['avg_time']:^12.4f} | {r['throughput']:^17.2f}")


if __name__ == "__main__":
    main()