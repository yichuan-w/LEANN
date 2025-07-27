import time

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import torch
from mlx_lm import load
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODEL_NAME_TORCH = "Qwen/Qwen3-Embedding-0.6B"
MODEL_NAME_MLX = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
BATCH_SIZES = [1, 8, 16, 32, 64, 128]
NUM_RUNS = 10  # Number of runs to average for each batch size
WARMUP_RUNS = 2  # Number of warm-up runs

# --- Generate Dummy Data ---
DUMMY_SENTENCES = ["This is a test sentence for benchmarking." * 5] * max(BATCH_SIZES)

# --- Benchmark Functions ---b


def benchmark_torch(model, sentences):
    start_time = time.time()
    model.encode(sentences, convert_to_numpy=True)
    end_time = time.time()
    return (end_time - start_time) * 1000  # Return time in ms


def benchmark_mlx(model, tokenizer, sentences):
    start_time = time.time()

    # Tokenize sentences using MLX tokenizer
    tokens = []
    for sentence in sentences:
        token_ids = tokenizer.encode(sentence)
        tokens.append(token_ids)

    # Pad sequences to the same length
    max_len = max(len(t) for t in tokens)
    input_ids = []
    attention_mask = []

    for token_seq in tokens:
        # Pad sequence
        padded = token_seq + [tokenizer.eos_token_id] * (max_len - len(token_seq))
        input_ids.append(padded)
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = [1] * len(token_seq) + [0] * (max_len - len(token_seq))
        attention_mask.append(mask)

    # Convert to MLX arrays
    input_ids = mx.array(input_ids)
    attention_mask = mx.array(attention_mask)

    # Get embeddings
    embeddings = model(input_ids)

    # Mean pooling
    mask = mx.expand_dims(attention_mask, -1)
    sum_embeddings = (embeddings * mask).sum(axis=1)
    sum_mask = mask.sum(axis=1)
    _ = sum_embeddings / sum_mask

    mx.eval()  # Ensure computation is finished
    end_time = time.time()
    return (end_time - start_time) * 1000  # Return time in ms


# --- Main Execution ---
def main():
    print("--- Initializing Models ---")
    # Load PyTorch model
    print(f"Loading PyTorch model: {MODEL_NAME_TORCH}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_torch = SentenceTransformer(MODEL_NAME_TORCH, device=device)
    print(f"PyTorch model loaded on: {device}")

    # Load MLX model
    print(f"Loading MLX model: {MODEL_NAME_MLX}")
    model_mlx, tokenizer_mlx = load(MODEL_NAME_MLX)
    print("MLX model loaded.")

    # --- Warm-up ---
    print("\n--- Performing Warm-up Runs ---")
    for _ in range(WARMUP_RUNS):
        benchmark_torch(model_torch, DUMMY_SENTENCES[:1])
        benchmark_mlx(model_mlx, tokenizer_mlx, DUMMY_SENTENCES[:1])
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
            benchmark_mlx(model_mlx, tokenizer_mlx, sentences_batch) for _ in range(NUM_RUNS)
        ]
        results_mlx.append(np.mean(mlx_times))

    print("\n--- Benchmark Results (Average time per batch in ms) ---")
    print(f"Batch Sizes: {BATCH_SIZES}")
    print(f"PyTorch (mps): {[f'{t:.2f}' for t in results_torch]}")
    print(f"MLX:           {[f'{t:.2f}' for t in results_mlx]}")

    # --- Plotting ---
    print("\n--- Generating Plot ---")
    plt.figure(figsize=(10, 6))
    plt.plot(
        BATCH_SIZES,
        results_torch,
        marker="o",
        linestyle="-",
        label=f"PyTorch ({device})",
    )
    plt.plot(BATCH_SIZES, results_mlx, marker="s", linestyle="-", label="MLX")

    plt.title(f"Embedding Performance: MLX vs PyTorch\nModel: {MODEL_NAME_TORCH}")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Time per Batch (ms)")
    plt.xticks(BATCH_SIZES)
    plt.grid(True)
    plt.legend()

    # Save the plot
    output_filename = "embedding_benchmark.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
    main()
