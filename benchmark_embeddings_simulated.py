import time

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODEL_NAME_TORCH = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]
NUM_RUNS = 10
WARMUP_RUNS = 2
SEQ_LENGTH = 256
EMBED_DIM = 768  # Dimension for all-mpnet-base-v2

# --- Generate Dummy Data ---
DUMMY_SENTENCES = ["This is a test sentence for benchmarking." * 5] * max(BATCH_SIZES)


# --- PyTorch Benchmark Function ---
def benchmark_torch(model, sentences):
    start_time = time.time()
    model.encode(sentences, convert_to_numpy=True)
    torch.mps.synchronize()  # Ensure computation is finished on MPS
    end_time = time.time()
    return (end_time - start_time) * 1000  # Return time in ms


# --- Simulated MLX Benchmark Function ---
def benchmark_mlx_simulated(dummy_embedding_table, sentences):
    # 1. Simulate tokenization (result is just shape)
    batch_size = len(sentences)
    input_ids = mx.random.randint(0, 30000, (batch_size, SEQ_LENGTH))
    attention_mask = mx.ones((batch_size, SEQ_LENGTH))

    start_time = time.time()
    # 2. Simulate embedding lookup
    embeddings = dummy_embedding_table[input_ids]

    # 3. Simulate mean pooling
    mask = mx.expand_dims(attention_mask, -1)
    sum_embeddings = (embeddings * mask).sum(axis=1)
    sum_mask = mask.sum(axis=1)
    _ = sum_embeddings / sum_mask

    mx.eval()  # Ensure all MLX computations are finished
    end_time = time.time()
    return (end_time - start_time) * 1000  # Return time in ms


# --- Main Execution ---
def main():
    print("--- Initializing Models ---")
    # Load real PyTorch model
    print(f"Loading PyTorch model: {MODEL_NAME_TORCH}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        print("Warning: MPS not available for PyTorch. Benchmark will run on CPU.")
    model_torch = SentenceTransformer(MODEL_NAME_TORCH, device=device)
    print(f"PyTorch model loaded on: {device}")

    # Create dummy MLX embedding table
    print("Creating simulated MLX model...")
    dummy_vocab_size = 30522  # Typical BERT vocab size
    dummy_embedding_table_mlx = mx.random.normal((dummy_vocab_size, EMBED_DIM))
    mx.eval()  # Ensure table is created
    print("Simulated MLX model created.")

    # --- Warm-up ---
    print("\n--- Performing Warm-up Runs ---")
    for _ in range(WARMUP_RUNS):
        benchmark_torch(model_torch, DUMMY_SENTENCES[:1])
        benchmark_mlx_simulated(dummy_embedding_table_mlx, DUMMY_SENTENCES[:1])
    print("Warm-up complete.")

    # --- Benchmarking ---
    print("\n--- Starting Benchmark ---")
    results_torch = []
    results_mlx = []

    for batch_size in BATCH_SIZES:
        print(f"Benchmarking batch size: {batch_size}")
        sentences_batch = DUMMY_SENTENCES[:batch_size]

        # Benchmark PyTorch
        torch_times = [benchmark_torch(model_torch, sentences_batch) for _ in range(NUM_RUNS)]
        results_torch.append(np.mean(torch_times))

        # Benchmark MLX
        mlx_times = [
            benchmark_mlx_simulated(dummy_embedding_table_mlx, sentences_batch)
            for _ in range(NUM_RUNS)
        ]
        results_mlx.append(np.mean(mlx_times))

    print("\n--- Benchmark Results (Average time per batch in ms) ---")
    print(f"Batch Sizes: {BATCH_SIZES}")
    print(f"PyTorch (mps): {[f'{t:.2f}' for t in results_torch]}")
    print(f"MLX (simulated): {[f'{t:.2f}' for t in results_mlx]}")

    # --- Plotting ---
    print("\n--- Generating Plot ---")
    plt.figure(figsize=(10, 6))
    plt.plot(BATCH_SIZES, results_torch, marker="o", linestyle="-", label=f"PyTorch ({device})")
    plt.plot(BATCH_SIZES, results_mlx, marker="s", linestyle="-", label="MLX (Simulated)")

    plt.title("Simulated Embedding Performance: MLX vs PyTorch")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Time per Batch (ms)")
    plt.xticks(BATCH_SIZES)
    plt.grid(True)
    plt.legend()

    output_filename = "embedding_benchmark_simulated.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
    main()
