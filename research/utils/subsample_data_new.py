import os
import time
import json
import numpy as np
import multiprocessing


def subsample_jsonl_random(input_file_path, output_file_path, ratio=0.1, seed=42):
    """
    Subsamples 10% of the data from a JSONL file efficiently.

    Args:
    input_file_path (str): Path to the input JSONL file.
    output_file_path (str): Path to the output JSONL file where the subsample will be saved.
    seed (int): Seed for the random number generator to ensure reproducibility.
    """
    start_time = time.time()

    # First pass: count the number of lines in the file
    line_count = 0
    with open(input_file_path, "r", encoding="utf-8") as file:
        for _ in file:
            line_count += 1
    print(f"Total lines: {line_count}")

    # Calculate indices for 10% sample
    np.random.seed(seed)
    sample_size = int(line_count * ratio)
    selected_indices = set(np.random.choice(line_count, sample_size, replace=False))

    # Second pass: write the selected lines to the output file
    print(f"Subsampling {sample_size} lines")
    current_index = 0
    with (
        open(input_file_path, "r", encoding="utf-8") as input_file,
        open(output_file_path, "w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            if current_index in selected_indices:
                output_file.write(line)
            current_index += 1

    end_time = time.time()
    print(
        f"Time: {(end_time - start_time) / 60:.2f} minutes\tRaw Size: {line_count}\t Sampled Size: {sample_size}"
    )


if __name__ == "__main__":
    # input_dir = '/mnt/md-256k/redpajama_v1/common_crawl_2023_06'
    # output_dir = '/mnt/md-256k/massive_ds_data/subsampled_0.1/rpj_common_crawl_2023_06'
    input_dir = "/mnt/md-256k/massive_ds_data/full/dpr_wiki"
    output_dir = "/mnt/md-256k/massive_ds_data/subsampled_0.1/dpr_wiki"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        subsample_jsonl_random(input_path, output_path)
