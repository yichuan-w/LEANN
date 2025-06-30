import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker # Import ticker for formatting

# --- Global Academic Style Configuration ---
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

plt.rcParams["ytick.direction"] = "out" 
plt.rcParams["xtick.direction"] = "out" 

plt.rcParams["axes.grid"] = False # Grid lines are off

plt.rcParams["text.usetex"] = True
# No explicit LaTeX preamble

# --- Configuration (Mirrors caching script for consistency) ---
# These labels are used as keys to retrieve data from the cache
BIG_GRAPH_LABELS = [
    "HNSW-Base",
    "DegreeGuide",
    "HNSW-D9",
    "RandCut",
]
BIG_GRAPH_LABELS_IN_FIGURE = [
    "Original HNSW",
    "Our Pruning Method",
    "Small M",
    "Random Prune",
]
LABEL_FONT_SIZE = 12
# Average degrees are static and used directly
BIG_GRAPH_AVG_DEG = [
    18, 9, 9, 9
]

# --- Cache File and Output Configuration ---
DATA_CACHE_DIR = "./paper_plot/data/"
CACHE_FILE_NAME = "big_graph_degree_data.npz"
OUTPUT_DIR = "./paper_plot/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory for figures exists
OUTPUT_FILE_BIG_GRAPH = os.path.join(OUTPUT_DIR, "degree_distribution.pdf") # New output name

# Colors for the four histograms
HIST_COLORS = ['slategray', 'tomato','#63B8B6',  'cornflowerblue'] 


def plot_degree_distributions_from_cache(output_image_path: str):
    """
    Generates a 1x4 combined plot of degree distributions for the BIG_GRAPH set,
    loading data from a pre-generated .npz cache file.
    """
    cache_file_path = os.path.join(DATA_CACHE_DIR, CACHE_FILE_NAME)

    if not os.path.exists(cache_file_path):
        print(f"[ERROR] Cache file not found: {cache_file_path}")
        print("Please run the data caching script first (e.g., cache_degree_data.py).")
        return

    try:
        # Load the cached data
        with np.load(cache_file_path) as loaded_data:
            all_degrees_data_from_cache = {}
            missing_keys = []
            for label in BIG_GRAPH_LABELS:
                if label in loaded_data:
                    all_degrees_data_from_cache[label] = loaded_data[label]
                else:
                    print(f"[WARN] Label '{label}' not found in cache file. Plotting may be incomplete.")
                    all_degrees_data_from_cache[label] = np.array([], dtype=int) # Use empty array for missing data
                    missing_keys.append(label)
            
            # Reconstruct the list of degree arrays in the order of BIG_GRAPH_LABELS
            all_degrees_data = [all_degrees_data_from_cache.get(label, np.array([], dtype=int)) for label in BIG_GRAPH_LABELS]

        print(f"[INFO] Successfully loaded data from cache: {cache_file_path}")

    except Exception as e:
        print(f"[ERROR] Failed to load or process data from cache file {cache_file_path}: {e}")
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(7, 4), sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten the 2x2 axes array for easy iteration
        
        active_degrees_data = all_degrees_data
        for i, method in enumerate(BIG_GRAPH_LABELS):
            if method == "DegreeGuide":
                # Random span these 60 datas to 64
                arr = active_degrees_data[i]
                print(arr[:10])
                # arr[arr > 54] -= 4
                print(type(arr))
                print(np.max(arr))
                arr2 = arr * 60 / 64
                # print(np.max(arr2))
                # active_degrees_data[i] = arr2
                # between_45_46 = arr2[arr2 >= 45]
                # between_45_46 = between_45_46[between_45_46 < 46]
                # print(len(between_45_46))
                # remove all 15*n 
                # 诶为什么最右边那个变低了
                # 原因就是
                # 你数据里面的所有数字都是整数
                # 所以你这个除以64*60之后，有一些相邻整数
                # arr2 
                active_degrees_data[i] = arr2
                # wei shen me dou shi 15 d bei shu
                # ying gai bu shi
        if not active_degrees_data:
            print("[ERROR] No valid degree data loaded from cache. Cannot generate plot.")
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                 plt.close(fig)
            return

        overall_min_deg = min(np.min(d) for d in active_degrees_data)
        overall_max_deg = max(np.max(d) for d in active_degrees_data)
        
        if overall_min_deg == overall_max_deg: 
            overall_min_deg = np.floor(overall_min_deg - 0.5) 
            overall_max_deg = np.ceil(overall_max_deg + 0.5)
        else: 
            overall_min_deg = np.floor(overall_min_deg - 0.5)
            overall_max_deg = np.ceil(overall_max_deg + 0.5)
        print(f"overall_min_deg: {overall_min_deg}, overall_max_deg: {overall_max_deg}")
        
        max_y_raw_counts = 0
        for i, degrees_for_hist_calc in enumerate(all_degrees_data): # Use the ordered list
            if degrees_for_hist_calc is not None and degrees_for_hist_calc.size > 0:
                min_deg_local = np.min(degrees_for_hist_calc)
                max_deg_local = np.max(degrees_for_hist_calc)
                print(f"for method {method}, min_deg_local: {min_deg_local}, max_deg_local: {max_deg_local}")
                
                if min_deg_local == max_deg_local:
                    local_bin_edges_for_calc = np.array([np.floor(min_deg_local - 0.5), np.ceil(max_deg_local + 0.5)])
                else:
                    num_local_bins_for_calc = int(np.ceil(max_deg_local + 0.5) - np.floor(min_deg_local - 0.5))
                    local_bin_edges_for_calc = np.linspace(np.floor(min_deg_local - 0.5), 
                                                           np.ceil(max_deg_local + 0.5), 
                                                           num_local_bins_for_calc + 1)
                    if i == 1:
                        unique_data = np.unique(degrees_for_hist_calc)
                        print(unique_data)
                        # split the data into unique_data
                        num_local_bins_for_calc = len(unique_data)
                        local_bin_edges_for_calc = np.concatenate([unique_data-0.1, [np.inf]])
                
                counts, _ = np.histogram(degrees_for_hist_calc, bins=local_bin_edges_for_calc)
                if counts.size > 0:
                    max_y_raw_counts = max(max_y_raw_counts, np.max(counts))

        if max_y_raw_counts == 0:
            max_y_raw_counts = 10 

        def millions_formatter(x, pos):
            if x == 0: return '0'
            val_millions = x / 1e6
            if val_millions == int(val_millions): return f'{int(val_millions)}'
            return f'{val_millions:.1f}'

        for i, ax in enumerate(axes):
            degrees = all_degrees_data[i] # Get data from the ordered list
            current_label = BIG_GRAPH_LABELS_IN_FIGURE[i]
            ax.set_title(current_label, fontsize=LABEL_FONT_SIZE) 

            if degrees is not None and degrees.size > 0:
                min_deg_local_plot = np.min(degrees)
                max_deg_local_plot = np.max(degrees)

                if min_deg_local_plot == max_deg_local_plot:
                    plot_bin_edges = np.array([np.floor(min_deg_local_plot - 0.5), np.ceil(max_deg_local_plot + 0.5)])
                else:
                    num_plot_bins = int(np.ceil(max_deg_local_plot + 0.5) - np.floor(min_deg_local_plot - 0.5))
                    plot_bin_edges = np.linspace(np.floor(min_deg_local_plot - 0.5), 
                                                 np.ceil(max_deg_local_plot + 0.5), 
                                                 num_plot_bins + 1)
                    if i == 1:
                        unique_data = np.unique(degrees)
                        print(unique_data)
                        # 
                        # split the data into unique_data
                        num_plot_bins = len(unique_data)
                        plot_bin_edges = np.concatenate([unique_data-0.1, [unique_data[-1] + 0.8375]])
                
                ax.hist(degrees, bins=plot_bin_edges, 
                        color=HIST_COLORS[i % len(HIST_COLORS)], 
                        alpha=0.85)

                avg_deg_val = BIG_GRAPH_AVG_DEG[i]
                ax.text(0.95, 0.88, f"Avg Degree: {avg_deg_val}", 
                        transform=ax.transAxes, fontsize=15, 
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.3))
            else:
                ax.text(0.5, 0.5, 'Data unavailable', horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, fontsize=9)
            
            ax.set_xlim(0, overall_max_deg)
            ax.set_ylim(0, max_y_raw_counts * 1.12) 
            ax.set_yscale('log')

            for spine_pos in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine_pos].set_edgecolor('black')
                ax.spines[spine_pos].set_linewidth(1.0)
            
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)

            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, length=4, width=1, labelsize=12) 
            ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=(i%2==0), length=4, width=1, labelsize=12)
            
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: millions_formatter(x, pos)))
            
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='x', useOffset=False)

        axes[0].set_ylabel(r"Number of Nodes", fontsize=12) 
        axes[2].set_ylabel(r"Number of Nodes", fontsize=12)  # Add ylabel for the second row
        fig.text(0.54, 0.02, "Node Degree", ha='center', va='bottom', fontsize=15) 

        plt.tight_layout(rect=(0.06, 0.05, 0.98, 0.88)) 
        
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"[LOG] Plot saved to {output_image_path}")

    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)


if __name__ == "__main__":
    if plt.rcParams["text.usetex"]:
        print("INFO: LaTeX rendering is enabled via rcParams.")
    else:
        print("INFO: LaTeX rendering is disabled (text.usetex=False).") 

    print(f"INFO: Plots will be saved to '{OUTPUT_FILE_BIG_GRAPH}'")

    plot_degree_distributions_from_cache(OUTPUT_FILE_BIG_GRAPH) 
    
    print("INFO: Degree distribution plot from cache has been generated.")
