#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /bottleneck_breakdown.py
# \brief: Illustrates the query time bottleneck on consumer devices (Final Version - Font & Legend Adjust).
# Author: Gemini Assistant (adapted from user's style and feedback)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter # Not strictly needed for just font, but imported if user wants to try

# Set matplotlib styles similar to the example
plt.rcParams["font.family"] = "Helvetica" # Primary font family
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1.0
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

plt.rcParams["text.usetex"] = True
# Attempt to make LaTeX use Helvetica as the main font
plt.rcParams['text.latex.preamble'] = r"""
\usepackage{helvet}       % helvetica font
\usepackage{sansmath}     % helvetica for math
\sansmath                % activate sansmath
\renewcommand{\familydefault}{\sfdefault} % make sans-serif the default family
"""


# Final Data for the breakdown (3 Segments)
labels_raw = [ # Raw labels before potential LaTeX escaping
    'IO: Text + PQ Lookup',
    'CPU: Tokenize + Distance Compute',
    'GPU: Embedding Recompute',
]
# Times in ms, ordered for stacking
times_ms = np.array([
    8.009,   # Quantization
    16.197,  # Search
    76.512,  # Embedding Recomputation
])

total_time_ms = times_ms.sum()
percentages = (times_ms / total_time_ms) * 100

# Prepare labels for legend, escaping for LaTeX if active
labels_legend = []
# st1 = r'&' # Not needed as current labels_raw don't have '&'
for label, time, perc in zip(labels_raw, times_ms, percentages):
    # Construct the percentage string carefully for LaTeX
    perc_str = f"{perc:.1f}" + r"\%" # Correct way to form 'NN.N\%'
    # label_tex = label.replace('&', st1) # Use if '&' is in labels_raw
    label_tex = label # Current labels_raw are clean for LaTeX
    labels_legend.append(
        f"{label_tex}\n({time:.1f}ms, {perc_str})"
    )

# Styling based on user's script
# Using first 3 from the provided lists
edgecolors_list = ["dimgrey", "#63B8B6", "tomato", "silver", "slategray"]
hatches_list = ["/////", "xxxxx", "\\\\\\\\\\"]

edgecolors = edgecolors_list[:3]
hatches = hatches_list[:3]
fill_color = "white"

# Create the figure and axes
# Adjusted figure size to potentially accommodate legend on the right
fig, ax = plt.subplots()
fig.set_size_inches(7, 1.5) # Width increased slightly, height adjusted
# Adjusted right margin for external legend, bottom for x-label
plt.subplots_adjust(left=0.12, right=0.72, top=0.95, bottom=0.25)

# Create the horizontal stacked bar
bar_height = 0.2
y_pos = 0

left_offset = 0
for i in range(len(times_ms)):
    ax.barh(
        y_pos,
        times_ms[i],
        height=bar_height,
        left=left_offset,
        color=fill_color,
        edgecolor=edgecolors[i],
        hatch=hatches[i],
        linewidth=1.5,
        label=labels_legend[i],
        zorder=10
    )
    text_x_pos = left_offset + times_ms[i] / 2
    if times_ms[i] > total_time_ms * 0.03: # Threshold for displaying text
        ax.text(
            text_x_pos,
            y_pos,
            f"{times_ms[i]:.1f}ms",
            ha='center',
            va='center',
            fontsize=8,
            fontweight='bold',
            color='black',
            zorder=20,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.5, alpha=0.8)
        )
    left_offset += times_ms[i]

# Set plot limits and labels
ax.set_xlim([0, total_time_ms * 1.02])
ax.set_xlabel("Time (ms)", fontsize=14, fontweight='bold', x=0.75, )

# Y-axis: Remove y-ticks and labels
ax.set_yticks([])
ax.set_yticklabels([])

# Legend: Placed to the right of the plot
ax.legend(
    # (x, y) for anchor, (0,0) is bottom left, (1,1) is top right of AXES
    # To place outside on the right, x should be > 1
    bbox_to_anchor=(1.03, 0.5), # x > 1 means outside to the right, y=0.5 for vertical center
    ncol=1, # Single column for a taller, narrower legend
    loc="center left", # Anchor the legend's left-center to bbox_to_anchor point
    labelspacing=0.5, # Adjust spacing
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    shadow=False,
    fancybox=False,
    handlelength=1.5,
    handletextpad=0.6,
    columnspacing=1.5,
    prop={"weight": "bold", "size": 9},
).set_zorder(100)

# Save the figure (using the original generic name as requested)
output_filename = "./bottleneck_breakdown.pdf"
# plt.tight_layout() # tight_layout might conflict with external legend; adjust subplots_adjust instead
plt.savefig(output_filename, bbox_inches="tight", dpi=300)
print(f"Saved plot to {output_filename}")

# plt.show() # Uncomment to display plot interactively