import csv

import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
SAVE_PTH = "./paper_plot/figures"
font_size = 16

# Generation(LLama 1B)	Generation(LLama 3B)	Generation(LLama 7B)
# 0.085s	0.217s	0.472s
llm_inference_time=[0.085, 0.217, 0.472, 0]

USE_LLM_INDEX = 3 # +0

file_path = "./paper_plot/data/main_latency.csv"

with open(file_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    data = list(reader)

# 打印原始数据
for row in data:
    print(",".join(row))




models = ["A10", "MAC"]
datasets = ["NQ", "TriviaQA", "GPQA", "HotpotQA"]
data = [[float(cell) if cell.isdigit() else cell for cell in row] for row in data[1:]]
for k, model in enumerate(models):

    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(20, 3)
    plt.subplots_adjust(wspace=0, hspace=0)

    total_width, n = 6, 6
    group = 1
    width = total_width * 0.9 / n
    x = np.arange(group) * n
    exit_idx_x = x + (total_width - width) / n
    edgecolors = ["dimgrey", "#63B8B6", "tomato", "slategray", "mediumpurple", "green", "red", "blue", "yellow", "silver"]
    # hatches = ["", "\\\\", "//", "||", "x", "--", "..", "", "\\\\", "//", "||", "x", "--", ".."]
    hatches =["\\\\\\","\\\\"]

    labels = [
        "HNSW",
        "IVF",
        "DiskANN",
        "IVF-Disk",
        "IVF-Recompute",
        "Our",
        # "DGL-OnDisk",
    ]
    if k == 0:
        x_labels = "GraphSAGE"
    else:
        x_labels = "GAT"

    yticks = [0.01, 0.1, 1, 10, 100, 1000,10000]  # Log scale ticks
    val_limit = 15000  # Upper limit for the plot

    for i in range(4):
        axes[i].set_yscale('log')  # Set y-axis to logarithmic scale
        axes[i].set_yticks(yticks)
        axes[i].set_ylim(0.01, val_limit)  # Lower limit should be > 0 for log scale

        axes[i].tick_params(axis="y", labelsize=10)

        axes[i].set_xticks([])
        # axes[i].set_xticklabels()
        axes[i].set_xlabel(datasets[i], fontsize=font_size)
        axes[i].grid(axis="y", linestyle="--")
        axes[i].set_xlim(exit_idx_x[0] - 0.15 * width - 0.2, exit_idx_x[0] + (n-0.25)* width + 0.2)
        for j in range(n):
            ##TODO add label

            # num = float(data[i * 2 + k][j + 3])
            # plot_label = [num]
            # if j == 6 and i == 3:
            #     plot_label = ["N/A"]
            #     num = 0
            local_hatches=["////","\\\\","xxxx"]
            # here add 3 bars rather than one bar TODO 
            print('exit_idx_x',exit_idx_x)
            
            # Check if all three models for this algorithm are OOM (data = 0)
            is_oom = True
            for m in range(3):
                if float(data[i * 6 + k*3 + m][j + 3]) != 0:
                    is_oom = False
                    break
                    
            if is_oom:
                # Draw a cross for OOM instead of bars
                pos = exit_idx_x + j * width + width * 0.3  # Center position for cross
                marker_size = width * 150  # Size of the cross
                axes[i].scatter(pos, 0.02, marker='x', color=edgecolors[j], s=marker_size, 
                               linewidth=4, label=labels[j] if j < len(labels) else "", zorder=20)
            else:
                # Create three separate bar calls instead of trying to plot multiple bars at once
                for m in range(3):
                    num = float(data[i * 6 + k*3 +m][j + 3]) +llm_inference_time[USE_LLM_INDEX]
                    plot_label = [num]
                    pos = exit_idx_x + j * width + width * 0.3 * m
                    print(f"j: {j}, m: {m}, pos: {pos}")
                    # For log scale, we need to ensure values are positive
                    plot_value = max(0.01, num) if num < val_limit else val_limit
                    container = axes[i].bar(
                        pos,
                        plot_value,
                        width=width * 0.3,
                        color="white",
                        edgecolor=edgecolors[j],
                        # edgecolor="k",
                        hatch=local_hatches[m],  # Use different hatches for each of the 3 bars
                        linewidth=1.0,
                        label=labels[j] if m == 0 else "",  # Only add label for the first bar
                        zorder=10,
                    )
            # axes[i].bar_label(
            #     container,
            #     plot_label,
            #     fontsize=font_size - 2,
            #     zorder=200,
            #     fontweight="bold",
            # )

    if k == 0:
        axes[0].legend(
            bbox_to_anchor=(3.25, 1.02),
            ncol=7,
            loc="lower right",
            # fontsize=font_size,
            # markerscale=3,
            labelspacing=0.2,
            edgecolor="black",
            facecolor="white",
            framealpha=1,
            shadow=False,
            # fancybox=False,
            handlelength=2,
            handletextpad=0.5,
            columnspacing=0.5,
            prop={"weight": "bold", "size": font_size},
        ).set_zorder(100)

    axes[0].set_ylabel("Runtime (log scale)", fontsize=font_size, fontweight="bold")
    axes[0].set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$",r"$10^{4}$"], fontsize=font_size)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    axes[3].set_yticklabels([])

    plt.savefig(f"{SAVE_PTH }/speed_{model}_revised.pdf", bbox_inches="tight", dpi=300)
    ## print save
    print(f"{SAVE_PTH }/speed_{model}_revised.pdf")