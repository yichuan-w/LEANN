import numpy as np
import os

# --- Configuration for Data Paths and Labels (Mirrors plotting script for consistency) ---
BIG_GRAPH_PATHS = [
    "/opt/dlami/nvme/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/hnsw/",
    "/opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/indices/99_4_degree_based_hnsw_IP_M32_efC256/",
    "/opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/indices/d9_hnsw_IP_M8_efC128/",
    "/opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/indices/half_edges_IP_M32_efC128/"
]
STATS_FILE_NAME = "degree_distribution.txt"
BIG_GRAPH_LABELS = [  # These will be used as keys in the cached file
    "HNSW-Base",
    "DegreeGuide",
    "HNSW-D9", 
    "RandCut",
]
# Average degrees are static and can be directly used in the plotting script or also cached.
# For simplicity here, we'll focus on caching the dynamic degree arrays.
# BIG_GRAPH_AVG_DEG = [18, 9, 9, 9] 

# --- Cache File Configuration ---
DATA_CACHE_DIR = "./paper_plot/data/"
CACHE_FILE_NAME = "big_graph_degree_data.npz" # Using .npz for multiple arrays

def create_degree_data_cache():
    """
    Reads degree distribution data from specified text files and saves it
    into a compressed NumPy (.npz) cache file.
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    cache_file_path = os.path.join(DATA_CACHE_DIR, CACHE_FILE_NAME)

    cached_data = {}
    print(f"Starting data caching process for {len(BIG_GRAPH_PATHS)} graph types...")

    for i, base_path in enumerate(BIG_GRAPH_PATHS):
        method_label = BIG_GRAPH_LABELS[i]
        degree_file_path = os.path.join(base_path, STATS_FILE_NAME)
        
        print(f"Processing: {method_label} from {degree_file_path}")
        
        try:
            # Load degrees as integers
            degrees = np.loadtxt(degree_file_path, dtype=int)
            
            if degrees.size == 0:
                print(f"  [WARN] Degree file is empty: {degree_file_path}. Storing as empty array for {method_label}.")
                # Store an empty array or handle as needed. For npz, an empty array is fine.
                cached_data[method_label] = np.array([], dtype=int) 
            else:
                # Store the loaded degrees array with the method label as the key
                cached_data[method_label] = degrees
                print(f"  [INFO] Loaded {len(degrees)} degrees for {method_label}. Max degree: {np.max(degrees) if degrees.size > 0 else 'N/A'}")
                
        except FileNotFoundError:
            print(f"  [ERROR] Degree file not found: {degree_file_path}. Skipping {method_label}.")
            # Optionally store a placeholder or skip. For robustness, store None or an empty array.
            # Storing None might require special handling when loading. Empty array is safer for np.load.
            cached_data[method_label] = np.array([], dtype=int) # Store empty array if file not found
        except Exception as e:
            print(f"  [ERROR] An error occurred loading {degree_file_path} for {method_label}: {e}")
            cached_data[method_label] = np.array([], dtype=int) # Store empty array on other errors

    if not cached_data:
        print("[ERROR] No data was successfully processed or loaded. Cache file will not be created.")
        return

    try:
        # Save all collected degree arrays into a single .npz file.
        # Using savez_compressed for potentially smaller file size.
        np.savez_compressed(cache_file_path, **cached_data)
        print(f"\n[SUCCESS] Degree distribution data successfully cached to: {os.path.abspath(cache_file_path)}")
        print("Cached arrays (keys):", list(cached_data.keys()))
    except Exception as e:
        print(f"\n[ERROR] Failed to save data to cache file {cache_file_path}: {e}")

if __name__ == "__main__":
    print("--- Degree Distribution Data Caching Script ---")
    create_degree_data_cache()
    print("--- Caching script finished. ---")
