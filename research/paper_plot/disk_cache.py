import matplotlib
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "sans-serif"  # Use generic sans-serif family
plt.rcParams['text.latex.preamble'] = r"""
\usepackage{helvet} % Use Helvetica font for text
\usepackage{sfmath} % Use sans-serif font for math
\renewcommand{\familydefault}{\sfdefault} % Set sans-serif as default text font
\usepackage[T1]{fontenc} % Recommended for font encoding
"""
# plt.rcParams['mathtext.fontset'] = 'dejavusans'
SAVE_PTH = "./paper_plot/figures"
font_size = 16

# New data in dictionary format
datasets = ["NQ", "TriviaQA", "GPQA", "Hotpot"]

cache_ratios = ["4.2G\n (0\%)", "8.7G\n (2.5\%)", "13.2G\n (5\%)", "18.6G\n (8\%)", "22.2G\n (10\%)"]
latency_data = {
    "NQ": [4.616, 4.133, 3.826, 3.511, 3.323],
    "TriviaQA": [5.777, 4.979, 4.553, 4.141, 3.916],
    "GPQA": [1.733, 1.593, 1.468, 1.336, 1.259],
    "Hotpot": [15.515, 13.479, 12.383, 11.216, 10.606],
}
cache_hit_counts = {
    "NQ": [0, 14.81, 23.36, 31.99, 36.73],
    "TriviaQA": [0, 18.55, 27.99, 37.06, 41.86],
    "GPQA": [0, 10.99, 20.31, 29.71, 35.01],
    "Hotpot": [0, 17.47, 26.91, 36.2, 41.06]
}

# Create the figure with 4 subplots in a 2x2 grid
fig, axes_grid = plt.subplots(2, 2, figsize=(7,6))
axes = axes_grid.flatten()  # Flatten the 2x2 grid to a 1D array

# Bar style settings
width = 0.7
x = np.arange(len(cache_ratios))

# Define hatch patterns for different cache ratios
hatch_patterns = ['//', '//', '//', '//', '//']

# Find max cache hit value across all datasets for unified y-axis
all_hit_counts = []
for dataset in datasets:
    all_hit_counts.extend(cache_hit_counts[dataset])
max_unified_hit = max(all_hit_counts) * 1.13

for i, dataset in enumerate(datasets):
    latencies = latency_data[dataset]
    hit_counts = cache_hit_counts[dataset]

    for j, val in enumerate(latencies):
        container = axes[i].bar(
            x[j],
            val,
            width=width,
            color="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=10,
        )
        axes[i].bar_label(
            container,
            [f"{val:.2f}"],
            fontsize=10,
            zorder=200,
            fontweight="bold",
        )

    axes[i].set_title(dataset, fontsize=font_size)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(cache_ratios, fontsize=12, rotation=0, ha='center', fontweight="bold")

    max_val_ratios = [1.35, 1.65, 1.45, 1.75]
    max_val = max(latencies) * max_val_ratios[i]
    axes[i].set_ylim(0, max_val)
    axes[i].tick_params(axis='y', labelsize=12)

    if i % 2 == 0:
        axes[i].set_ylabel("Latency (s)", fontsize=font_size)
        axes[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    ax2: Axes = axes[i].twinx()
    ax2.plot(x, hit_counts,
        linestyle='--',
        marker='o',
        markersize=6,
        linewidth=1.5,
        color='k',
        markerfacecolor='none',
        zorder=20)

    ax2.set_ylim(0, max_unified_hit)
    ax2.tick_params(axis='y', labelsize=12)
    if i % 2 == 1:
        ax2.set_ylabel(r"Cache Hit (\%)", fontsize=font_size)

    for j, val in enumerate(hit_counts):
        if val > 0:
            ax2.annotate(f"{val:.1f}%",
                         (x[j], val),
                         textcoords="offset points",
                         xytext=(0, 5),
                         ha='center',
                         va='bottom',
                         fontsize=10,
                         fontweight='bold')

# Create legend for both plots
bar_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Latency')
line_patch = Line2D([0], [0], color='black', linestyle='--', label='Cache Hit Rate')

# --- MODIFICATION FOR LEGEND AT THE TOP ---
fig.legend(handles=[bar_patch, line_patch],
           loc='upper center',        # Position the legend at the upper center
           bbox_to_anchor=(0.5, 0.995), # Anchor point (0.5 means horizontal center of figure,
                                      # 0.97 means 97% from the bottom, so near the top)
           ncol=3,
           fontsize=font_size-2)
# --- END OF MODIFICATION ---

# Set common x-axis label - you might want to add this back if needed
# fig.text(0.5, 0.02, "Disk Cache Size", ha='center', fontsize=font_size, fontweight='bold') # Adjusted y for potential bottom label

# --- MODIFICATION FOR TIGHT LAYOUT ---
# Adjust rect to make space for the legend at the top.
# (left, bottom, right, top_for_subplots)
# We want subplots to occupy space from y=0 up to y=0.93 (or similar)
# leaving the top portion (0.93 to 1.0) for the legend.
plt.tight_layout(rect=(0, 0, 1, 0.93)) # Ensure subplots are below the legend
# --- END OF MODIFICATION ---

# Create directory if it doesn't exist (optional, good practice)
import os
if not os.path.exists(SAVE_PTH):
    os.makedirs(SAVE_PTH)

plt.savefig(f"{SAVE_PTH}/disk_cache_latency.pdf", dpi=300) # Changed filename slightly for testing
print(f"Save to {SAVE_PTH}/disk_cache_latency.pdf")
# plt.show() # Optional: to display the plot