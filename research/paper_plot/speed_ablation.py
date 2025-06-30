#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /speed_ablation.py
# \brief:
# Author: raphael hao

# %%
import numpy as np
import pandas as pd

# %%
# from script.settings import DATA_PATH, FIGURE_PATH

# Load the latency ablation data
latency_data = pd.read_csv("./paper_plot/data/latency_ablation.csv")
# Filter for SpeedUp metric only
speedup_data = latency_data[latency_data['Metric'] == 'SpeedUp']

# %%
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1.5
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True

# %%

fig, ax = plt.subplots()
fig.set_size_inches(5, 1.5)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 3, 3
group = len(speedup_data['Dataset'].unique())
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "#63B8B6", "tomato", "silver", "slategray"]
hatches = ["/////", "xxxxx", "\\\\\\\\\\"]
labels = ["Base", "Base + Two-level", "Base + Two-level + Batch"]

datasets = speedup_data['Dataset'].unique()

for i, dataset in enumerate(datasets):
    dataset_data = speedup_data[speedup_data['Dataset'] == dataset]
    
    for j in range(n):
        if j == 0:
            value = dataset_data['Original'].values[0]
        elif j == 1:
            value = dataset_data['original + two_level'].values[0]
        else:
            value = dataset_data['original + two_level + batch'].values[0]
            
        ax.text(
            exit_idx_x[i] + j * width,
            value + 0.05,
            f"{value:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            rotation=0,
            zorder=20,
        )
        
        ax.bar(
            exit_idx_x[i] + j * width,
            value,
            width=width * 0.8,
            color="white",
            edgecolor=edgecolors[j],
            hatch=hatches[j],
            linewidth=1.5,
            label=labels[j] if i == 0 else None,
            zorder=10,
        )



ax.set_ylim([0.5, 2.3])
ax.set_yticks(np.arange(0.5, 2.2, 0.5))
ax.set_yticklabels(np.arange(0.5, 2.2, 0.5), fontsize=12)
ax.set_xticks(exit_idx_x + width)
ax.set_xticklabels(datasets, fontsize=10)
# ax.set_xlabel("Different Datasets", fontsize=14)
ax.legend(
    bbox_to_anchor=(-0.03, 1.4),
    ncol=3,
    loc="upper left",
    labelspacing=0.1,
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    shadow=False,
    fancybox=False,
    handlelength=0.8,
    handletextpad=0.6,
    columnspacing=0.8,
    prop={"weight": "bold", "size": 10},
).set_zorder(100)
ax.set_ylabel("Speedup", fontsize=11)

plt.savefig("./paper_plot/figures/latency_speedup.pdf", bbox_inches="tight", dpi=300)

# %%

print(f"Save to ./paper_plot/figures/latency_speedup.pdf")