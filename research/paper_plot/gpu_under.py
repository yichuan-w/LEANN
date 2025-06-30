#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /gpu_utilization_plot.py
# \brief: Plots GPU throughput vs. batch size to show utilization with equally spaced x-axis.
# Author: AI Assistant

import numpy as np
import pandas as pd # Using pandas for data structuring, similar to example
from matplotlib import pyplot as plt

# Apply styling similar to the example script
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
# plt.rcParams["hatch.linewidth"] = 1.5 # Not used for line plots
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True # Enables LaTeX for text rendering

# New Benchmark data (4th set)
data = {
    'batch_size': [1, 4, 8, 10, 16, 20, 32, 40, 64, 128, 256,],
    'avg_time_s': [
        0.0031, 0.0057, 0.0100, 0.0114, 0.0186, 0.0234,
        0.0359, 0.0422, 0.0626, 0.1259, 0.2454,
    ],
    'throughput_seq_s': [
        318.10, 696.77, 798.95, 874.70, 859.58, 855.19,
        890.80, 946.93, 1022.75, 1017.03, 1043.17,
    ]
}
benchmark_df = pd.DataFrame(data)

# Create the plot
# Increased width slightly for more x-axis labels
fig, ax = plt.subplots()
fig.set_size_inches(8, 5)

# Generate equally spaced x-coordinates (indices)
x_indices = np.arange(len(benchmark_df))

# Plotting throughput vs. batch size (using indices for x-axis)
ax.plot(
    x_indices, # Use equally spaced indices for plotting
    benchmark_df['throughput_seq_s'],
    marker='o',       # Add markers to data points
    linestyle='-',
    color="#63B8B6",  # A color inspired by the example's 'edgecolors'
    linewidth=2,
    markersize=6,
    # label="Model Throughput" # Label for legend if needed, but not showing legend by default
)

# Setting labels for axes
ax.set_xlabel("Batch Size", fontsize=14)
ax.set_ylabel("Throughput (sequences/second)", fontsize=14)

# Customizing Y-axis for the new data range:
# Start Y from 0 to include the anomalous low point and show full scale.
y_min_val = 200
# Round up y_max_val to the nearest 100, as max throughput > 1000
y_max_val = np.ceil(benchmark_df['throughput_seq_s'].max() / 100) * 100
ax.set_ylim((y_min_val, y_max_val))
# Set y-ticks every 100 units, ensuring the top tick is included.
ax.set_yticks(np.arange(y_min_val, y_max_val + 1, 100))

# Customizing X-axis for equally spaced ticks:
# Set tick positions to the indices
ax.set_xticks(x_indices)
# Set tick labels to the actual batch_size values
ax.set_xticklabels(benchmark_df['batch_size'])
ax.tick_params(axis='x', rotation=45, labelsize=10) # Rotate X-axis labels, fontsize 10
ax.tick_params(axis='y', labelsize=12)


# Add a light grid for better readability, common in academic plots
ax.grid(True, linestyle=':', linewidth=0.5, color='grey', alpha=0.7, zorder=0)

# Remove title (as requested)
# ax.set_title("GPU Throughput vs. Batch Size", fontsize=16) # Title would go here

# Optional: Add a legend if you have multiple lines or want to label the single line
# ax.legend(
#     loc="center right", # Location might need adjustment due to data shape
#     edgecolor="black",
#     facecolor="white",
#     framealpha=1.0,
#     shadow=False,
#     fancybox=False,
#     prop={"weight": "bold", "size": 10}
# ).set_zorder(100)

# Adjust layout to prevent labels from being cut off
plt.tight_layout()

# Save the figure
output_filename = "./paper_plot/figures/gpu_throughput_vs_batch_size_equispaced.pdf"
plt.savefig(output_filename, bbox_inches="tight", dpi=300)
print(f"Plot saved to {output_filename}")

# Display the plot (optional, depending on environment)
plt.show()

# %%
# This is just to mimic the '%%' cell structure from the example.
# No actual code needed here for this script.