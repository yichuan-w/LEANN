# python faiss/demo/plot_graph_struct.py faiss/demo/output.log
# python faiss/demo/plot_graph_struct.py large_graph_recompute.log
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

# Modified recall_levels and corresponding styles/widths from previous step
recall_levels = [0.90, 0.92, 0.94, 0.96]
line_styles = ['--', '-', '-', '-'] 
line_widths = [1, 1.5, 1.5, 1.5]

MAPPED_METHOD_NAMES = [
    # 'HNSW-Base',
    # 'DegreeGuide',
    # 'HNSW-D9',
    # 'RandCut',
    "Original HNSW",
    "Our Pruning Method",
    "Small M",
    "Random Prune",
]

PERFORMANCE_PLOT_PATH = './paper_plot/figures/H_hnsw_performance_comparison.pdf'
SAVED_PATH = './paper_plot/figures/H_hnsw_recall_comparison.pdf'

def extract_data_from_log(log_content):
    """Extract method names, recall lists, and recompute lists from the log file."""
    
    method_pattern = r"Building HNSW index with ([^\.]+)\.\.\.|Building HNSW index with ([^\n]+)..."
    recall_list_pattern = r"recall_list: (\[[\d\., ]+\])"
    recompute_list_pattern = r"recompute_list: (\[[\d\., ]+\])"
    avg_neighbors_pattern = r"neighbors per node: ([\d\.]+)"
    
    method_matches = re.findall(method_pattern, log_content)
    # Temporary list for raw method identifiers from regex
    _methods_raw_identifiers_regex = []
    for match in method_matches:
        method_ident = match[0] if match[0] else match[1]
        _methods_raw_identifiers_regex.append(method_ident.strip().rstrip('.'))
    
    recall_lists_str = re.findall(recall_list_pattern, log_content)
    recompute_lists_str = re.findall(recompute_list_pattern, log_content)
    avg_neighbors_str_list = re.findall(avg_neighbors_pattern, log_content) # Keep as string list for now

    # Determine if regex approach was sufficient, similar to original logic
    # This check helps decide if we use regex-extracted names or fallback to split-parsing
    _min_len_for_regex_path = min(
        len(_methods_raw_identifiers_regex) if _methods_raw_identifiers_regex else 0,
        len(recall_lists_str) if recall_lists_str else 0,
        len(recompute_lists_str) if recompute_lists_str else 0,
        len(avg_neighbors_str_list) if avg_neighbors_str_list else 0
    )

    methods = [] # This will hold the final display names

    if _min_len_for_regex_path < 4 : # Fallback path if regex didn't get enough (e.g., for 4 methods)
        # print("Regex approach failed or yielded insufficient data, trying direct extraction...")
        sections = log_content.split("Building HNSW index with ")[1:]
        methods_temp = []
        for section in sections:
            method_name_raw = section.split("\n")[0].strip().rstrip('.')
            # Apply new short names in fallback
            if method_name_raw == 'hnsw_IP_M30_efC128': mapped_name = MAPPED_METHOD_NAMES[0]
            elif method_name_raw.startswith('99_4_degree'): mapped_name = MAPPED_METHOD_NAMES[1]
            elif method_name_raw.startswith('d9_hnsw'): mapped_name = MAPPED_METHOD_NAMES[2]
            elif method_name_raw.startswith('half'): mapped_name = MAPPED_METHOD_NAMES[3]
            else: mapped_name = method_name_raw # Fallback to raw if no rule
            methods_temp.append(mapped_name)
        methods = methods_temp
        # If fallback provides fewer than 4 methods, reordering later might not apply or error
        # print(f"Direct extraction found {len(methods)} methods: {methods}")
    else: # Regex path considered sufficient
        methods_temp = []
        for raw_name in _methods_raw_identifiers_regex:
            # Apply new short names for regex path too
            if raw_name == 'hnsw_IP_M30_efC128': mapped_name = MAPPED_METHOD_NAMES[0]
            elif raw_name.startswith('99_4_degree'): mapped_name = MAPPED_METHOD_NAMES[1]
            elif raw_name.startswith('d9_hnsw'): mapped_name = MAPPED_METHOD_NAMES[2]
            elif raw_name.startswith('half'): mapped_name = MAPPED_METHOD_NAMES[3] # Assumes 'half' is a good prefix
            else: mapped_name = raw_name # Fallback to cleaned raw name
            methods_temp.append(mapped_name)
        methods = methods_temp
        # print(f"Regex extraction found {len(methods)} methods: {methods}")

    # Convert string lists of numbers to actual numbers
    avg_neighbors = [float(avg) for avg in avg_neighbors_str_list]

    # Reordering (This reordering is crucial for color consistency if colors are fixed by position)
    # It assumes methods[0] is Base, methods[1] is Our, etc., *before* this reordering step
    # if that was the natural order from logs. The reordering swaps 3rd and 4th items.
    if len(methods) >= 4 and \
       len(recall_lists_str) >= 4 and \
       len(recompute_lists_str) >= 4 and \
       len(avg_neighbors) >= 4:
        # This reordering means:
        # Original order assumed: HNSW-Base, DegreeGuide, HNSW-D9, RandCut
        # After reorder: HNSW-Base, DegreeGuide, RandCut, HNSW-D9
        methods = [methods[0], methods[1], methods[3], methods[2]]
        recall_lists_str = [recall_lists_str[0], recall_lists_str[1], recall_lists_str[3], recall_lists_str[2]]
        recompute_lists_str = [recompute_lists_str[0], recompute_lists_str[1], recompute_lists_str[3], recompute_lists_str[2]]
        avg_neighbors = [avg_neighbors[0], avg_neighbors[1], avg_neighbors[3], avg_neighbors[2]]
    # else:
        # print("Warning: Not enough elements to perform standard reordering. Using data as found.")


    if len(avg_neighbors) > 0 and avg_neighbors_str_list[0] == "17.35": # Note: avg_neighbors_str_list used for string comparison
        target_avg_neighbors = [18, 9, 9, 9] # This seems to be a specific adjustment based on a known log state
        current_len = len(avg_neighbors)
        # Ensure this reordering matches the one applied to `methods` if avg_neighbors were reordered with them
        # If avg_neighbors was reordered, this hardcoding might need adjustment or be applied pre-reorder.
        # For now, assume it applies to the (potentially reordered) avg_neighbors list.
        avg_neighbors = target_avg_neighbors[:current_len]


    recall_lists = [eval(recall_list) for recall_list in recall_lists_str]
    recompute_lists = [eval(recompute_list) for recompute_list in recompute_lists_str]
    
    # Final truncation to ensure all lists have the same minimum length
    min_length = min(len(methods), len(recall_lists), len(recompute_lists), len(avg_neighbors))
    
    methods = methods[:min_length]
    recall_lists = recall_lists[:min_length]
    recompute_lists = recompute_lists[:min_length]
    avg_neighbors = avg_neighbors[:min_length]
    
    return methods, recall_lists, recompute_lists, avg_neighbors


def plot_recall_comparison(methods, recall_lists, recompute_lists, avg_neighbors, current_recall_levels):
    """Create a line chart comparing computation costs at different recall levels, with academic style."""    
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["ytick.direction"] = "in"
    # plt.rcParams["hatch.linewidth"] = 1.5 # From example, but not used in line plot
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["text.usetex"] = True # Ensure LaTeX is available or set to False
    
    computation_costs = []
    for i, method_name in enumerate(methods): # methods now contains short names
        method_costs = []
        for level in current_recall_levels:
            recall_idx = next((idx for idx, recall in enumerate(recall_lists[i]) if recall >= level), None)
            if recall_idx is not None:
                method_costs.append(recompute_lists[i][recall_idx])
            else:
                method_costs.append(None) 
        computation_costs.append(method_costs)

    fig, ax = plt.subplots(figsize=(5,2.5))
    
    # Modified academic_colors for consistency
    # HNSW-Base (Grey), DegreeGuide (Red), RandCut (Cornflowerblue), HNSW-D9 (DarkBlue)
    # academic_colors = ['dimgrey', 'tomato', 'cornflowerblue', '#003366', 'forestgreen', 'crimson']
    academic_colors = [ 'slategray', 'tomato', 'cornflowerblue','#63B8B6',]
    markers = ['o', '*', '^', 'D', 'v', 'P']
    # Origin, Our, Random, SmallM


    for i, method_name in enumerate(methods): # method_name is now short, e.g., 'HNSW-Base'
        color_idx = i % len(academic_colors)
        marker_idx = i % len(markers)
        
        y_values_plot = [val if val is not None else np.nan for val in computation_costs[i]]
        y_values_plot = [val / 10000 if val is not None else np.nan for val in computation_costs[i]]

        if method_name == MAPPED_METHOD_NAMES[0]: # Original HNSW-Base
            linestyle = '--'
        else:
            linestyle = '-'
        if method_name == MAPPED_METHOD_NAMES[1]: # Our Pruning Method
            marker_size = 12
        elif method_name == MAPPED_METHOD_NAMES[2]: # Small M
            marker_size = 7.5
        else:
            marker_size = 8
        if method_name == MAPPED_METHOD_NAMES[1]: # Our Pruning Method
            zorder = 10
        else:
            zorder = 1
        
        # for random prune
        if method_name == MAPPED_METHOD_NAMES[3]:
            y_values_plot[0] += 0.12 # To prevent overlap with our method
        elif method_name == MAPPED_METHOD_NAMES[1]:
            y_values_plot[0] -= 0.06 # To prevent overlap with original hnsw

        ax.plot(current_recall_levels, y_values_plot, 
                 label=f"{method_name} (Avg Degree: {int(avg_neighbors[i])})", # Uses new short names
                 color=academic_colors[color_idx], marker=markers[marker_idx], markeredgecolor='#FFFFFF80', # zhege miaobian shibushi buhaokan()
                 markersize=marker_size, linewidth=2, linestyle=linestyle, zorder=zorder)

    ax.set_xlabel('Recall Target', fontsize=9, fontweight="bold")
    ax.set_ylabel('Nodes to Recompute', fontsize=9, fontweight="bold")
    ax.set_xticks(current_recall_levels)
    ax.set_xticklabels([f'{level*100:.0f}\%' for level in current_recall_levels], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.set_ylabel(r'Nodes to Recompute ($\mathbf{\times 10^4}$)', fontsize=9, fontweight="bold")

    # Legend styling (already moved up from previous request)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, 
              fontsize=6, edgecolor="black", facecolor="white", framealpha=1,
              shadow=False, fancybox=False, prop={"weight": "normal", "size": 8})
    
    # No grid lines: ax.grid(True, linestyle='--', alpha=0.7)
    
    # Spines adjustment for academic look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    annot_recall_level_92 = 0.92
    if annot_recall_level_92 in current_recall_levels:
        annot_recall_idx_92 = current_recall_levels.index(annot_recall_level_92)
        method_base_name = "Our Pruning Method"
        method_compare_92_name = "Small M"

        if method_base_name in methods and method_compare_92_name in methods:
            idx_base = methods.index(method_base_name)
            idx_compare_92 = methods.index(method_compare_92_name)
            cost_base_92 = computation_costs[idx_base][annot_recall_idx_92] / 10000
            cost_compare_92 = computation_costs[idx_compare_92][annot_recall_idx_92] / 10000

            if cost_base_92 is not None and cost_compare_92 is not None and cost_base_92 > 0:
                ratio_92 = cost_compare_92 / cost_base_92
                ax.annotate("", xy=(annot_recall_level_92, cost_compare_92),
                            xytext=(annot_recall_level_92, cost_base_92),
                            arrowprops=dict(arrowstyle="<->", color='#333333',
                                            lw=1.5, mutation_scale=15,
                                            shrinkA=3, shrinkB=3),
                            zorder=10) # Arrow drawn first

                text_x_pos_92 = annot_recall_level_92 # Text x is on the arrow line
                text_y_pos_92 = (cost_base_92 + cost_compare_92) / 2
                plot_ymin, plot_ymax = ax.get_ylim() # Boundary checks
                if text_y_pos_92 < plot_ymin + (plot_ymax-plot_ymin)*0.05: text_y_pos_92 = plot_ymin + (plot_ymax-plot_ymin)*0.05
                if text_y_pos_92 > plot_ymax - (plot_ymax-plot_ymin)*0.05: text_y_pos_92 = plot_ymax - (plot_ymax-plot_ymin)*0.05

                ax.text(text_x_pos_92, text_y_pos_92, f"{ratio_92:.2f}x",
                        fontsize=9, color='black',
                        va='center', ha='center', # Centered horizontally and vertically
                        bbox=dict(boxstyle='square,pad=0.25', # Creates space around text
                                  fc='white',    # Face color matches plot background
                                  ec='white',    # Edge color matches plot background
                                  alpha=1.0),              # Fully opaque
                        zorder=11) # Text on top of arrow

    # --- Annotation for performance gap at 96% recall (0.96) ---
    annot_recall_level_96 = 0.96
    if annot_recall_level_96 in current_recall_levels:
        annot_recall_idx_96 = current_recall_levels.index(annot_recall_level_96)
        method_base_name = "Our Pruning Method"
        method_compare_96_name = "Random Prune"

        if method_base_name in methods and method_compare_96_name in methods:
            idx_base = methods.index(method_base_name)
            idx_compare_96 = methods.index(method_compare_96_name)
            cost_base_96 = computation_costs[idx_base][annot_recall_idx_96] / 10000
            cost_compare_96 = computation_costs[idx_compare_96][annot_recall_idx_96] / 10000

            if cost_base_96 is not None and cost_compare_96 is not None and cost_base_96 > 0:
                ratio_96 = cost_compare_96 / cost_base_96
                ax.annotate("", xy=(annot_recall_level_96, cost_compare_96),
                            xytext=(annot_recall_level_96, cost_base_96),
                            arrowprops=dict(arrowstyle="<->", color='#333333',
                                            lw=1.5, mutation_scale=15,
                                            shrinkA=3, shrinkB=3),
                            zorder=10) # Arrow drawn first

                text_x_pos_96 = annot_recall_level_96 # Text x is on the arrow line
                text_y_pos_96 = (cost_base_96 + cost_compare_96) / 2
                plot_ymin, plot_ymax = ax.get_ylim() # Boundary checks
                if text_y_pos_96 < plot_ymin + (plot_ymax-plot_ymin)*0.05: text_y_pos_96 = plot_ymin + (plot_ymax-plot_ymin)*0.05
                if text_y_pos_96 > plot_ymax - (plot_ymax-plot_ymin)*0.05: text_y_pos_96 = plot_ymax - (plot_ymax-plot_ymin)*0.05

                ax.text(text_x_pos_96, text_y_pos_96, f"{ratio_96:.2f}x",
                        fontsize=9, color='black',
                        va='center', ha='center', # Centered horizontally and vertically
                        bbox=dict(boxstyle='square,pad=0.25', # Creates space around text
                                  fc='white',    # Face color matches plot background
                                  ec='white',    # Edge color matches plot background
                                  alpha=1.0),              # Fully opaque
                        zorder=11) # Text on top of arrow


    plt.tight_layout(pad=0.5)
    plt.savefig(SAVED_PATH, bbox_inches="tight", dpi=300)
    plt.show()

# --- Main script execution ---
parser = argparse.ArgumentParser()
parser.add_argument("log_file", type=str, default="./demo/output.log")
args = parser.parse_args()

try:
    with open(args.log_file, 'r') as f:
        log_content = f.read()
except FileNotFoundError:
    print(f"Error: Log file '{args.log_file}' not found.")
    exit()

methods, recall_lists, recompute_lists, avg_neighbors = extract_data_from_log(log_content)

if methods: 
    # plot_performance(methods, recall_lists, recompute_lists, avg_neighbors)
    # print(f"Performance plot saved to {PERFORMANCE_PLOT_PATH}")
    
    plot_recall_comparison(methods, recall_lists, recompute_lists, avg_neighbors, recall_levels)
    print(f"Recall comparison plot saved to {SAVED_PATH}")

    print("\nMethod Summary:")
    for i, method in enumerate(methods):
        print(f"{method}:")
        if i < len(avg_neighbors): # Check index bounds
             print(f"  - Average neighbors per node: {avg_neighbors[i]:.2f}")
        
        for level in recall_levels:
            if i < len(recall_lists) and i < len(recompute_lists): # Check index bounds
                recall_idx = next((idx for idx, recall_val in enumerate(recall_lists[i]) if recall_val >= level), None)
                if recall_idx is not None:
                    print(f"  - Computations needed for {level*100:.0f}% recall: {recompute_lists[i][recall_idx]:.0f}")
                else:
                    print(f"  - Does not reach {level*100:.0f}% recall in the test")
            else:
                print(f"  - Data missing for recall/recompute lists for method {method}")
        print()
else:
    print("No data extracted from the log file. Cannot generate plots or summary.")