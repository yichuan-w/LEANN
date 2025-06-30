import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
from matplotlib.patches import FancyArrowPatch

sns.set_theme(style="ticks", font_scale=1.2)
plt.rcParams['axes.grid'] = True          
plt.rcParams['axes.grid.which'] = 'major'  
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['ytick.minor.visible'] = False
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# Generation(LLama 1B)	Generation(LLama 3B)	Generation(LLama 7B)
# 0.085s	0.217s	0.472s
# llm_inference_time=[0.085, 0.217, 0.472, 0] # Will be replaced by CSV data
# llm_inference_time_for_mac = [0.316, 0.717, 1.468, 0] # Will be replaced by CSV data

def parse_latency_data(csv_path):
    df = pd.read_csv(csv_path)
    latency_data = {}
    llm_gen_times = {}  # To store LLM generation times: (dataset, hardware) -> time
    
    for _, row in df.iterrows():
        dataset = row['Dataset']
        hardware = row['Hardware']
        recall_target_str = row['Recall_target'].replace('%', '')
        try:
            recall_target = float(recall_target_str)
        except ValueError:
            print(f"Warning: Could not parse recall_target '{row['Recall_target']}'. Skipping row.")
            continue

        if (dataset, hardware) not in llm_gen_times: # Read once per (dataset, hardware)
            llm_time_val = pd.to_numeric(row.get('LLM_Gen_Time_1B'), errors='coerce')
            if not pd.isna(llm_time_val):
                llm_gen_times[(dataset, hardware)] = llm_time_val
            else:
                llm_gen_times[(dataset, hardware)] = np.nan # Store NaN if unparsable/missing

        cols_to_skip = ['Dataset', 'Hardware', 'Recall_target',
                        'LLM_Gen_Time_1B', 'LLM_Gen_Time_3B', 'LLM_Gen_Time_7B']

        for col in df.columns:
            if col not in cols_to_skip:
                method_name = col
                key = (dataset, hardware, method_name)
                if key not in latency_data:
                    latency_data[key] = []
                try:
                    latency_value = float(row[method_name])
                    latency_data[key].append((recall_target, latency_value))
                except ValueError:
                    # Handle cases where latency might be non-numeric (e.g., 'N/A' or empty)
                    print(f"Warning: Could not parse latency for {method_name} at {dataset}/{hardware}/Recall {recall_target} ('{row[method_name]}'). Skipping this point.")
                    latency_data[key].append((recall_target, np.nan)) # Or skip appending

    # Sort by recall for consistent plotting
    for key in latency_data:
        latency_data[key].sort(key=lambda x: x[0])
    return latency_data, llm_gen_times

def parse_storage_data(csv_path):
    df = pd.read_csv(csv_path)
    storage_data = {}
    # Assuming the first column is 'MetricType' (RAM/Storage) and subsequent columns are methods
    # And the header row is like: MetricType, Method1, Method2, ...
    # Transpose to make methods as rows for easier lookup might be an option,
    # but let's try direct parsing.
    
    # Find the row for RAM and Storage
    ram_row = df[df.iloc[:, 0] == 'RAM'].iloc[0]
    storage_row = df[df.iloc[:, 0] == 'Storage'].iloc[0]
    
    methods = df.columns[1:] # First column is the metric type label
    for method in methods:
        storage_data[method] = {
            'RAM': pd.to_numeric(ram_row[method], errors='coerce'),
            'Storage': pd.to_numeric(storage_row[method], errors='coerce')
        }
    return storage_data

# Load data
latency_csv_path = 'paper_plot/data/main_latency.csv'
storage_csv_path = 'paper_plot/data/ram_storage.csv'
latency_data, llm_generation_times = parse_latency_data(latency_csv_path)
storage_info = parse_storage_data(storage_csv_path)

# --- Determine unique Datasets and Hardware combinations to plot for ---
unique_dataset_hardware_configs = sorted(list(set((d, h) for d, h, m in latency_data.keys())))

if not unique_dataset_hardware_configs:
    print("Error: No (Dataset, Hardware) combinations found in latency data. Check CSV paths and content.")
    exit()

# --- Define constants for plotting ---
all_method_names = sorted(list(set(m for d,h,m in latency_data.keys())))
if not all_method_names:
    # Fallback if latency_data is empty but storage_info might have method names
    all_method_names = sorted(list(storage_info.keys()))

if not all_method_names:
    print("Error: No method names found in data. Cannot proceed with plotting.")
    exit()
    
method_markers = {
    'HNSW': 'o', 
    'IVF': 'X', 
    'DiskANN': 's', 
    'IVF-Disk': 'P', 
    'IVF-Recompute': '^', 
    'Our': '*',
    'BM25': "v"
    # Add more if necessary, or make it dynamic
}
method_display_names = {
    'IVF-Recompute': 'IVF-Recompute (EdgeRAG)',
    # 其他方法保持原名
}

# Ensure all methods have a marker
default_markers = ['^', 'v', '<', '>', 'H', 'h', '+', 'x', '|', '_']
next_default_marker = 0
for mn in all_method_names:
    if mn not in method_markers:
        print(f"mn: {mn}")
        method_markers[mn] = default_markers[next_default_marker % len(default_markers)]
        next_default_marker +=1

recall_levels_present = sorted(list(set(r for key in latency_data for r, l in latency_data[key])))
# Define colors for up to a few common recall levels, add more if needed
base_recall_colors = {
    85.0: "#1f77b4", # Blue
    90.0: "#ff7f0e", # Orange
    95.0: "#2ca02c", # Green
    # Add more if other recall % values exist
}
recall_colors = {}
color_palette = sns.color_palette("viridis", n_colors=len(recall_levels_present))

for idx, r_level in enumerate(recall_levels_present):
    recall_colors[r_level] = base_recall_colors.get(r_level, color_palette[idx % len(color_palette)])


# --- Determine global x (latency) and y (storage) limits for consistent axes ---
all_latency_values = []
all_storage_values = []
raw_data_size = 76  # Raw data size in GB

for ds_hw_key in unique_dataset_hardware_configs:
    current_ds, current_hw = ds_hw_key
    for method_name in all_method_names:
        # Get storage for this method
        disk_storage = storage_info.get(method_name, {}).get('Storage', np.nan)
        if not np.isnan(disk_storage):
            all_storage_values.append(disk_storage)
        
        # Get latencies for this method under current_ds, current_hw
        latency_key = (current_ds, current_hw, method_name)
        if latency_key in latency_data:
            for recall, latency in latency_data[latency_key]:
                if not np.isnan(latency):
                    all_latency_values.append(latency)

# Add padding to limits
min_lat = min(all_latency_values) if all_latency_values else 0.001
max_lat = max(all_latency_values) if all_latency_values else 1
min_store = min(all_storage_values) if all_storage_values else 0
max_store = max(all_storage_values) if all_storage_values else 1

# Convert storage values to proportion of raw data
min_store_proportion = min_store / raw_data_size if all_storage_values else 0
max_store_proportion = max_store / raw_data_size if all_storage_values else 0.1

# Padding for log scale latency - adjust minimum to be more reasonable
lat_log_min = -1  # Changed from -2 to -1 to set minimum to 10^-1 (0.1s)
lat_log_max = np.log10(max_lat) if max_lat > 0 else 3   # default to 1000 s
lat_padding = (lat_log_max - lat_log_min) * 0.05
global_xlim = [10**(lat_log_min - lat_padding), 10**(lat_log_max + lat_padding)]
if global_xlim[0] <= 0: global_xlim[0] = 0.1  # Changed from 0.01 to 0.1

# Padding for linear scale storage proportion
store_padding = (max_store_proportion - min_store_proportion) * 0.05
global_ylim = [max(0, min_store_proportion - store_padding), max_store_proportion + store_padding]
if global_ylim[0] >= global_ylim[1]: # Avoid inverted or zero range
    global_ylim[1] = global_ylim[0] + 0.1

# After loading the data and before plotting, add this code to reorder the datasets
# Find where you define all_datasets (around line 95)

# Original code:
all_datasets = sorted(list(set(ds for ds, _ in unique_dataset_hardware_configs)))

# Replace with this to specify the exact order:
all_datasets_unsorted = list(set(ds for ds, _ in unique_dataset_hardware_configs))
desired_order = ['NQ', 'TriviaQA',  'GPQA','HotpotQA']
all_datasets = [ds for ds in desired_order if ds in all_datasets_unsorted]
# Add any datasets that might be in the data but not in our desired_order list
all_datasets.extend([ds for ds in all_datasets_unsorted if ds not in desired_order])

# Then the rest of your code remains the same:
a10_configs = [(ds, 'A10') for ds in all_datasets if (ds, 'A10') in unique_dataset_hardware_configs]
mac_configs = [(ds, 'MAC') for ds in all_datasets if (ds, 'MAC') in unique_dataset_hardware_configs]

# Create two figures - one for A10 and one for MAC
hardware_configs = [a10_configs, mac_configs]
hardware_names = ['A10', 'MAC']

for fig_idx, configs_for_this_figure in enumerate(hardware_configs):
    if not configs_for_this_figure:
        continue

    num_cols_this_figure = len(configs_for_this_figure)
    # 1 row, num_cols_this_figure columns
    fig, axs = plt.subplots(1, num_cols_this_figure, figsize=(7 * num_cols_this_figure, 6), sharex=True, sharey=True, squeeze=False)
    
    # fig.suptitle(f"Latency vs. Storage ({hardware_names[fig_idx]})", fontsize=18, y=0.98)

    for subplot_idx, (current_ds, current_hw) in enumerate(configs_for_this_figure):
        ax = axs[0, subplot_idx] # Accessing column in the first row
        ax.set_title(f"{current_ds}", fontsize=25) # No need to show hardware in title since it's in suptitle

        for method_name in all_method_names:
            marker = method_markers.get(method_name, '+') 
            disk_storage = storage_info.get(method_name, {}).get('Storage', np.nan)

            latency_points_key = (current_ds, current_hw, method_name)
            if latency_points_key in latency_data:
                points_for_method = latency_data[latency_points_key]
                print(f"points_for_method: {points_for_method}")
                for recall, latency in points_for_method:
                    # Only skip if latency is invalid (since we need log scale for x-axis)
                    # But allow zero storage since y-axis is now linear
                    if np.isnan(latency) or np.isnan(disk_storage) or latency <= 0:
                        continue
                    
                    # Add LLM generation time from CSV
                    current_llm_add_time = llm_generation_times.get((current_ds, current_hw))
                    if current_llm_add_time is not None and not np.isnan(current_llm_add_time):
                        latency = latency + current_llm_add_time
                    else:
                        raise ValueError(f"No LLM generation time found for {current_ds} on {current_hw}")

                    # Special handling for BM25
                    if method_name == 'BM25':
                        # BM25 is only valid for 85% recall points (other points are 0)
                        if recall != 85.0:
                            continue
                        color = 'grey'
                    else:
                        # Use the color for target recall
                        color = recall_colors.get(recall, 'grey')
                    
                    # Convert storage to proportion
                    disk_storage_proportion = disk_storage / raw_data_size
                    size = 80
                    
                    x_offset = -50
                    if current_ds == 'GPQA':
                        x_offset = -32

                    # Apply a small vertical offset to IVF-Recompute points to make them more visible
                    if method_name == 'IVF-Recompute':
                        # Add a small vertical offset (adjust the 0.05 value as needed)
                        disk_storage_proportion += 0.07
                        size = 80
                    if method_name == 'DiskANN':
                        size = 50
                    if method_name == 'Our':
                        size = 140
                        disk_storage_proportion += 0.05
                        # Add "Pareto Frontier" label to Our method points
                        
                        if recall == 95:
                            ax.annotate('Ours', 
                                    (latency, disk_storage_proportion),
                                    xytext=(x_offset, 25),  # Increased leftward offset from -65 to -120
                                    textcoords='offset points',
                                    fontsize=20,
                                    color='red',
                                   weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))
                    # Increase size for BM25 points
                    if method_name == 'BM25':
                        size = 70
                    size*=5
                    
                    ax.scatter(latency, disk_storage_proportion, marker=marker, color=color, 
                               s=size, alpha=0.85, edgecolors='black', linewidths=0.7)
                    
            

        
        ax.set_xscale("log")
        ax.set_yscale("linear") # CHANGED from log scale to linear scale for Y-axis
        
        # Generate appropriate powers of 10 based on your data range
        min_power = -1
        max_power = 4
        log_ticks = [10**i for i in range(min_power, max_power+1)]

        # Set custom tick positions
        ax.set_xticks(log_ticks)

        # Create custom bold LaTeX labels with 10^n format
        log_tick_labels = [fr'$\mathbf{{10^{{{i}}}}}$' for i in range(min_power, max_power+1)]
        ax.set_xticklabels(log_tick_labels, fontsize=24)

        # Apply global limits
        if subplot_idx == 0:
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)

        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
        # Remove minor grid lines completely
        ax.grid(False, which="minor")
        
        # Remove ticks
        # First set the shared parameters for both axes
        ax.tick_params(axis='both', which='both', length=0, labelsize=24)

        # Then set the padding only for the x-axis
        ax.tick_params(axis='x', which='both', pad=10)
        
        if subplot_idx == 0: # Y-label only for the leftmost subplot
            ax.set_ylabel("Proportional Size", fontsize=24)
        
        # X-label for all subplots in a 1xN layout can be okay, or just the middle/last one.
        # Let's put it on all for now.
        ax.set_xlabel("Latency (s)", fontsize=25)

        # Display 100%, 200%, 300% for yaxis
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['100\%', '200\\%', '300\\%'])

        # Create a custom arrow with "Better" text inside
        # Create the arrow patch with a wider shaft
        arrow = FancyArrowPatch(
            (0.8, 0.8),  # Start point (top-right)
            (0.65, 0.6),  # End point (toward bottom-left)
            transform=ax.transAxes,
            arrowstyle='simple,head_width=40,head_length=35,tail_width=20',  # Increased arrow dimensions
            facecolor='white',
            edgecolor='black',
            linewidth=3,  # Thicker outline
            zorder=5
        )
        
        # Add the arrow to the plot
        ax.add_patch(arrow)
        
        # Calculate the midpoint of the arrow for text placement
        mid_x = (0.8 + 0.65) / 2 + 0.002 + 0.01
        mid_y = (0.8 + 0.6) / 2 + 0.01
        
        # Add the "Better" text at the midpoint of the arrow
        ax.text(mid_x, mid_y, 'Better', 
                transform=ax.transAxes,
                ha='center', 
                va='center',
                fontsize=16,  # Increased font size from 12 to 16
                fontweight='bold',
                rotation=40,  # Rotate to match arrow direction
                zorder=6)  # Ensure text is on top of arrow

    # Create legends (once per figure)
    method_legend_handles = []
    for method, marker_style in method_markers.items():
        if method in all_method_names: 
            print(f"method: {method}")
            # Use black color for BM25 in the legend
            if method == 'BM25':
                method_legend_handles.append(mlines.Line2D([], [], color='black', marker=marker_style, linestyle='None',
                                markersize=10, label=method))
            else:
                if method in method_display_names:
                    method = method_display_names[method]
                method_legend_handles.append(mlines.Line2D([], [], color='black', marker=marker_style, linestyle='None',
                                markersize=10, label=method))
    
    recall_legend_handles = []
    sorted_recall_levels = sorted(recall_colors.keys())
    for r_level in sorted_recall_levels:
        recall_legend_handles.append(mlines.Line2D([], [], color=recall_colors[r_level], marker='o', linestyle='None',
                                   markersize=20, label=f"Target Recall={r_level:.0f}\%"))

    # 将图例分成两行：第一行是方法，第二行是召回率
    if fig_idx == 0:
        # 从方法列表中先排除'Our'
        other_methods = [m for m in all_method_names if m != 'Our']
        # 按照需要的顺序创建方法列表（将'Our'放在最后）
        ordered_methods = other_methods + (['Our'] if 'Our' in all_method_names else [])
        
        # 按照新顺序创建方法图例句柄
        method_legend_handles = []
        for method in ordered_methods:
            if method in method_markers:
                marker_style = method_markers[method]
                # 使用显示名称映射
                display_name = method_display_names.get(method, method)
                color = 'black'
                marker_size = 22
                if method == 'Our':
                    marker_size = 27
                elif 'IVF-Recompute' in method or 'EdgeRAG' in method:
                    marker_size = 17
                elif 'DiskANN' in method:
                    marker_size = 19
                elif 'BM25' in method:
                    marker_size = 20
                method_legend_handles.append(mlines.Line2D([], [], color=color, marker=marker_style, 
                                        linestyle='None', markersize=marker_size, label=display_name))
    
        # 创建召回率图例（第二行）- 注意位置调整，放在方法图例下方
        recall_legend = fig.legend(handles=recall_legend_handles, 
                    loc='upper center', bbox_to_anchor=(0.5, 1.05),  # y坐标降低，放在第一行下方
                    ncol=len(recall_legend_handles), fontsize=28)


        # 创建方法图例（第一行）
        method_legend = fig.legend(handles=method_legend_handles, 
                    loc='upper center', bbox_to_anchor=(0.5, 0.91),
                    ncol=len(method_legend_handles), fontsize=28)
        
        # 添加图例到渲染器
        fig.add_artist(method_legend)
        fig.add_artist(recall_legend)

    # 调整布局，为顶部的两行图例留出更多空间
    plt.tight_layout(rect=(0, 0, 1.0, 0.74))  # 顶部空间从0.9调整到0.85，给两行图例留出更多空间
    
    save_path = f'./paper_plot/figures/main_exp_fig_{fig_idx+1}.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    print(f"Saved figure {fig_idx+1} to {save_path}")
    plt.show()