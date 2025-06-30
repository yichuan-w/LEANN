import sys
import time
import faiss
import numpy as np
import pickle
import os
import json
import time
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
import subprocess

# Add argument parsing
parser = argparse.ArgumentParser(description='Build and evaluate HNSW index')
parser.add_argument('--config', type=str, default="0.02per_M6-7_degree_based",
                    help='Configuration name for the index (default: 0.02per_M6-7_degree_based)')
parser.add_argument('--M', type=int, default=32,
                    help='HNSW M parameter (default: 32)')
parser.add_argument('--efConstruction', type=int, default=256,
                    help='HNSW efConstruction parameter (default: 256)')
parser.add_argument('--K_NEIGHBORS', type=int, default=3,
                    help='Number of neighbors to retrieve (default: 3)')
parser.add_argument('--max_queries', type=int, default=1000,
                    help='Maximum number of queries to load (default: 1000)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for query encoding (default: 64)')
parser.add_argument('--db_embedding_file', type=str, 
                    default="/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/passages_00.pkl",
                    help='Path to database embedding file')
parser.add_argument('--index_saving_dir', type=str,
                    default="/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices",
                    help='Directory to save the index')
parser.add_argument('--task_name', type=str, default="nq",
                    help='Task name from TASK_CONFIGS (default: nq)')
parser.add_argument('--embedder_model', type=str, default="facebook/contriever-msmarco",
                    help='Model name for query embedding (default: facebook/contriever-msmarco)')

args = parser.parse_args()

# Replace hardcoded constants with arguments
M = args.M
efConstruction = args.efConstruction
K_NEIGHBORS = args.K_NEIGHBORS
DB_EMBEDDING_FILE = args.db_embedding_file
INDEX_SAVING_FILE = args.index_saving_dir
TASK_NAME = args.task_name
EMBEDDER_MODEL_NAME = args.embedder_model
MAX_QUERIES_TO_LOAD = args.max_queries
QUERY_ENCODING_BATCH_SIZE = args.batch_size
CONFIG_NAME = args.config

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "demo"))
from config import SCALING_OUT_DIR, get_example_path, TASK_CONFIGS
sys.path.append(project_root)
from contriever.src.contriever import Contriever, load_retriever

# 1M samples
print(f"Loading embeddings from {DB_EMBEDDING_FILE}...")
with open(DB_EMBEDDING_FILE, 'rb') as f:
    data = pickle.load(f)

xb = data[1]
print(f"Original dtype: {xb.dtype}")

if xb.dtype != np.float32:
    print("Converting embeddings to float32.")
    xb = xb.astype(np.float32)
else:
    print("Embeddings are already float32.")
print(f"Loaded database embeddings (xb), shape: {xb.shape}")
d = xb.shape[1] # Get dimension

query_file_path = TASK_CONFIGS[TASK_NAME].query_path
print(f"Using query path from TASK_CONFIGS: {query_file_path}")

query_texts = []
print(f"Reading queries from: {query_file_path}")
with open(query_file_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= MAX_QUERIES_TO_LOAD:
            print(f"Stopped loading queries at limit: {MAX_QUERIES_TO_LOAD}")
            break
        record = json.loads(line)
        query_texts.append(record["query"])
print(f"Loaded {len(query_texts)} query texts.")

print("\nInitializing retriever model for encoding queries...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, tokenizer, _ = load_retriever(EMBEDDER_MODEL_NAME)
model.to(device)
model.eval() # Set to evaluation mode
print("Retriever model loaded.")


def embed_queries(queries, model, tokenizer, model_name_or_path, per_gpu_batch_size=64):
    """Embed queries using the model with batching"""
    model = model.half()
    model.eval()
    embeddings = []
    batch_question = []

    with torch.no_grad():
        for k, query in tqdm(enumerate(queries), desc="Encoding queries"):
            batch_question.append(query)

            # Process when batch is full or at the end
            if len(batch_question) == per_gpu_batch_size or k == len(queries) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=512,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                output = model(**encoded_batch)

                # Contriever typically uses output.last_hidden_state pooling or something specialized
                if "contriever" not in model_name_or_path:
                    output = output.last_hidden_state[:, 0, :]

                embeddings.append(output.cpu())
                batch_question = []  # Reset batch

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(f"Query embeddings shape: {embeddings.shape}")
    return embeddings

print(f"\nEncoding {len(query_texts)} queries (batch size: {QUERY_ENCODING_BATCH_SIZE})...")
xq_full = embed_queries(query_texts, model, tokenizer, EMBEDDER_MODEL_NAME, per_gpu_batch_size=QUERY_ENCODING_BATCH_SIZE)

# Ensure float32 for Faiss compatibility after encoding
if xq_full.dtype != np.float32:
    print(f"Converting encoded queries from {xq_full.dtype} to float32.")
    xq_full = xq_full.astype(np.float32)

print(f"Encoded queries (xq_full), shape: {xq_full.shape}, dtype: {xq_full.dtype}")

# Check dimension consistency
if xq_full.shape[1] != d:
     raise ValueError(f"Query embedding dimension ({xq_full.shape[1]}) does not match database dimension ({d})")

# recall_idx = []

print("\nBuilding FlatIP index for ground truth...")
index_flat = faiss.IndexFlatIP(d)  # Use Inner Product
index_flat.add(xb)
print(f"Searching FlatIP index with {MAX_QUERIES_TO_LOAD} queries (k={K_NEIGHBORS})...")
D_flat, recall_idx_flat = index_flat.search(xq_full, k=K_NEIGHBORS)

# print(recall_idx_flat)

# Create a specific directory for this index configuration
index_dir = f"{INDEX_SAVING_FILE}/{CONFIG_NAME}_hnsw_IP_M{M}_efC{efConstruction}"
if CONFIG_NAME == "origin":
    index_dir = f"{INDEX_SAVING_FILE}/hnsw_IP_M{M}_efC{efConstruction}"
os.makedirs(index_dir, exist_ok=True)
index_filename = f"{index_dir}/index.faiss"

# Check if index already exists
if os.path.exists(index_filename):
    print(f"Found existing index at {index_filename}, loading...")
    index = faiss.read_index(index_filename)
    print("Index loaded successfully.")
else:
    print('Building HNSW index (IP)...')
    # add build time
    start_time = time.time()
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.set_percentile_thresholds()
    index.add(xb)
    end_time = time.time()
    print(f'time: {end_time - start_time}')
    print('HNSW index built.')
    
    # Save the HNSW index
    print(f"Saving index to {index_filename}...")
    faiss.write_index(index, index_filename)
    print("Index saved successfully.")
# Analyze the HNSW index
print("\nAnalyzing HNSW index...")
print(f"Total number of nodes: {index.ntotal}")
print("Neighbor statistics:")
print(index.hnsw.print_neighbor_stats(0))

# Save degree distribution
distribution_filename = f"{index_dir}/degree_distribution.txt"
print(f"Saving degree distribution to {distribution_filename}...")
index.hnsw.save_degree_distribution(0, distribution_filename)
print("Degree distribution saved successfully.")

# Plot the degree distribution
plot_output_path = f"{index_dir}/degree_distribution.png"
print(f"Generating degree distribution plot to {plot_output_path}...")
try:
    subprocess.run(
        ["python", "/home/ubuntu/Power-RAG/utils/plot_degree_distribution.py", distribution_filename, "-o", plot_output_path],
        check=True
    )
    print(f"Degree distribution plot saved to {plot_output_path}")
except subprocess.CalledProcessError as e:
    print(f"Error generating degree distribution plot: {e}")
except FileNotFoundError:
    print("Warning: plot_degree_distribution.py script not found in current directory")

print('Searching HNSW index...')



# for efSearch in [2, 4, 8, 16, 32, 64,128,256,512,1024]:
#     print(f'*************efSearch: {efSearch}*************')
#     for i in range(10):
#         index.hnsw.efSearch = efSearch
#         D, I = index.search(xq_full[i:i+1], K_NEIGHBORS)
# exit()


recall_result_file = f"{index_dir}/recall_result.txt"
time_list = []
recall_list = []
recompute_list = []
with open(recall_result_file, 'w') as f:
    for efSearch in [2, 4, 8, 16, 24, 32, 48, 64, 96,114,128,144,160,176,192,208,224,240,256,384,420,440,460,480,512,768,1024,1152,1536,1792,2048,2230,2408,2880]:
        index.hnsw.efSearch = efSearch
        # calculate the time of searching
        start_time = time.time()
        faiss.cvar.hnsw_stats.reset()
        # print faiss.cvar.hnsw_stats.ndis
        print(f'ndis: {faiss.cvar.hnsw_stats.ndis}')
        D, I = index.search(xq_full, K_NEIGHBORS)
        print('D[0]:', D[0])
        end_time = time.time()
        print(f'time: {end_time - start_time}')
        time_list.append(end_time - start_time)
        print("recompute:", faiss.cvar.hnsw_stats.ndis/len(I))
        recompute_list.append(faiss.cvar.hnsw_stats.ndis/len(I))
        # print(I)

        # calculate the recall using the flat index the formula:
        # recall = sum(recall_idx == recall_idx_flat) / len(recall_idx)
        recall=[]
        for i in range(len(I)):
            acc = 0
            for j in range(len(I[i])):
                if I[i][j] in recall_idx_flat[i]:
                    acc += 1
            recall.append(acc / len(I[i]))
        recall = sum(recall) / len(recall)
        recall_list.append(recall)
        print(f'efSearch: {efSearch}')
        print(f'recall: {recall}')
        f.write(f'efSearch: {efSearch}, recall: {recall}\n')
print(f'Done and result saved to {recall_result_file}')
print(f'time_list: {time_list}')
print(f'recall_list: {recall_list}')
print(f'recompute_list: {recompute_list}')
exit()
# Analyze edge stats
print("\nAnalyzing edge statistics...")
edge_stats_file = f"{index_dir}/edge_stats.txt"
if not os.path.exists(edge_stats_file):
    index.save_edge_stats(edge_stats_file)
    print(f'Edge stats saved to {edge_stats_file}')
else:
    print(f'Edge stats already exists at {edge_stats_file}')


def analyze_edge_stats(filename):
    """
    Analyze edge statistics from a CSV file and print thresholds at various percentiles.
    
    Args:
        filename: Path to the edge statistics CSV file
    """
    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist")
        return
    
    print(f"Analyzing edge statistics from {filename}...")
    
    # Read the file
    distances = []
    with open(filename, 'r') as f:
        # Skip header
        header = f.readline()
        
        # Read all edges
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    src = int(parts[0])
                    dst = int(parts[1])
                    level = int(parts[2])
                    distance = float(parts[3])
                    distances.append(distance)
                except ValueError:
                    continue
    
    if not distances:
        print("No valid edges found in file")
        return
    
    # Sort distances
    distances = np.array(distances)
    distances.sort()
    
    # Calculate and print statistics
    print(f"Total edges: {len(distances)}")
    print(f"Min distance: {distances[0]:.6f}")
    print(f"Max distance: {distances[-1]:.6f}")
    
    # Print thresholds at specified percentiles
    percentiles = [0.5, 1, 2, 3, 5, 8, 10, 15, 20,30,40,50,60,70]
    print("\nDistance thresholds at percentiles:")
    for p in percentiles:
        idx = int(len(distances) * p / 100)
        if idx < len(distances):
            print(f"{p:.1f}%: {distances[idx]:.6f}")
    
    return distances

analyze_edge_stats(edge_stats_file)