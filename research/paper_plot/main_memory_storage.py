from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# Comment Test
 
# om script.settings import DATA_PATH, FIGURE_PATH
# DATA_PATH ="/home/ubuntu/Power-RAG/paper_plot/data"
# FIGURE_PATH = "/home/ubuntu/Power-RAG/paper_plot/figures"
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True

import numpy as np
import pandas as pd

# Load the RAM and Storage data directly from CSV
data = pd.read_csv("./paper_plot/data/ram_storage.csv")

# Explicitly reorder columns to ensure "Our" is at the end
cols = list(data.columns)
if "Our" in cols and cols[-1] != "Our":
    cols.remove("Our")
    cols.append("Our")
    data = data[cols]

# Set up the figure with two columns
fig = plt.figure(figsize=(12, 3))
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])  # Left panel for RAM
ax2 = fig.add_subplot(gs[0, 1])  # Right panel for Storage

# Define the visual style elements
edgecolors = ["dimgrey", "#63B8B6", "tomato", "slategray", "silver", "navy"]
hatches = ["/////", "\\\\\\\\\\"]

# Calculate positions for the bars
methods = data.columns[1:]  # Skip the 'Hardware' column
num_methods = len(methods)
# Reverse the order of methods for display (to have "Our" at the bottom)
methods = list(methods)[::-1]
y_positions = np.arange(num_methods)
bar_width = 0.6

# Plot RAM data in left panel
ram_bars = ax1.barh(
    y_positions,
    data.iloc[0, 1:].values[::-1],  # Reverse the data to match reversed methods
    height=bar_width,
    color="white",
    edgecolor=edgecolors[0],
    hatch=hatches[0],
    linewidth=1.0,
    label="RAM",
    zorder=10,
)
ax1.set_title("RAM Usage", fontsize=14, fontweight='bold')
ax1.set_yticks(y_positions)
ax1.set_yticklabels(methods, fontsize=14)
ax1.set_xlabel("Size (\\textit{GB})", fontsize=14)
ax1.xaxis.set_tick_params(labelsize=14)

# Plot Storage data in right panel
storage_bars = ax2.barh(
    y_positions, 
    data.iloc[1, 1:].values[::-1],  # Reverse the data to match reversed methods
    height=bar_width,
    color="white",
    edgecolor=edgecolors[1],
    hatch=hatches[1],
    linewidth=1.0,
    label="Storage",
    zorder=10,
)
ax2.set_title("Storage Usage", fontsize=14, fontweight='bold')
ax2.set_yticks(y_positions)
ax2.set_yticklabels(methods, fontsize=14)
ax2.set_xlabel("Size (\\textit{GB})", fontsize=14)
ax2.xaxis.set_tick_params(labelsize=14)

plt.tight_layout()
plt.savefig("./paper_plot/figures/ram_storage_double_column.pdf", bbox_inches="tight", dpi=300)
print("Saving the figure to ./paper_plot/figures/ram_storage_double_column.pdf")