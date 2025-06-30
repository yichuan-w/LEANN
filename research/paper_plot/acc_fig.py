import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set plot parameters
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1.5
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True

# Path settings
FIGURE_PATH = "./paper_plot/figures"

# Load accuracy data
acc_data = pd.read_csv("./paper_plot/data/acc.csv")

# Create figure with 4 subplots (one for each dataset)
fig, axs = plt.subplots(1, 4)
fig.set_size_inches(9, 2.5)

# Reduce the spacing between subplots
# plt.subplots_adjust(wspace=0.2)  # Reduced from 0.3 to 0.1

# Define datasets and their columns
datasets = ["NQ", "TriviaQA", "GPQA", "HotpotQA"]
metrics = ["Exact Match", "F1"]

# Define bar settings - make bars thicker
# total_width, n = 0.9, 3  # increased total width and n for three models
# width = total_width / n
# The 'width' variable below now defines the distance between the centers of adjacent bars within a group.
# It's also used as the base for calculating the actual plotted bar width.
# Original 2 bars had centers 1.0 apart. For 3 bars, we need a smaller distance.
# A value of 0.64 for distance between centers, with a scaling factor of 0.8 for bar width,
# results in an actual bar width of ~0.51, and a group span of ~1.79, similar to original's ~1.76.
n = 3 # Number of models
width = 0.64  # Distance between centers of adjacent bars in a group
bar_width_plotting_factor = 0.8 # Bar takes 80% of the space defined by 'width'

# Colors and hatches
edgecolors = ["dimgrey", "#63B8B6", "tomato"]  # Added color for PQ 5
hatches = ["/////", "xxxxx", "\\\\\\\\\\"]  # Added hatch for PQ 5
labels = ["BM25", "PQ Compressed", "Ours"] # Added PQ 5

# Create plots for each dataset
for i, dataset in enumerate(datasets):
    ax = axs[i]
    
    # Get data for this dataset and convert to percentages
    em_values = [
        acc_data.loc[0, f"{dataset} Exact Match"] * 100, 
        acc_data.loc[1, f"{dataset} Exact Match"] * 100,
        acc_data.loc[2, f"{dataset} Exact Match"] * 100  # Added PQ 5 EM data
    ]
    f1_values = [
        acc_data.loc[0, f"{dataset} F1"] * 100, 
        acc_data.loc[1, f"{dataset} F1"] * 100,
        acc_data.loc[2, f"{dataset} F1"] * 100  # Added PQ 5 F1 data
    ]
    
    # Define x positions for bars
    # For EM: center - width, center, center + width
    # For F1: center - width, center, center + width
    group_centers = [1.0, 3.0] # Centers for EM and F1 groups
    bar_offsets = [-width, 0, width]

    # Plot all bars on the same axis
    for metric_idx, metric_group_center in enumerate(group_centers):
        values_to_plot = em_values if metric_idx == 0 else f1_values
        for j, model_label in enumerate(labels):
            x_pos = metric_group_center + bar_offsets[j]
            bar_value = values_to_plot[j]
            
            ax.bar(
                x_pos,
                bar_value,
                width=width * bar_width_plotting_factor, # Use the new factor for bar width
                color="white",
                edgecolor=edgecolors[j],
                hatch=hatches[j],
                linewidth=1.5,
                label=model_label if i == 0 and metric_idx == 0 else None # Label only once
            )
            
            # Add value on top of bar
            ax.text(x_pos, bar_value + (0.1 if dataset == "GPQA" else 0.1), 
                    f"{bar_value:.1f}", ha='center', va='bottom', 
                    fontsize=9, fontweight='bold') # Reduced fontsize for text on bars
    
    # Set x-ticks and labels
    ax.set_xticks(group_centers)  # Position ticks at the center of each group
    xticklabels = ax.set_xticklabels(metrics, fontsize=12)

    # Now, shift these labels slightly to the right
    # Adjust this value to control the amount of shift (in data coordinates)
    # Given your group_centers are 1.0 and 3.0, a small value like 0.05 to 0.15 might be appropriate.
    # horizontal_shift = 0.7  # Try adjusting this value

    # for label in xticklabels:
    #     # Get the current x position (which is the tick location)
    #     current_x_pos = label.get_position()[0]
    #     # Set the new x position by adding the shift
    #     label.set_position((current_x_pos + horizontal_shift, label.get_position()[1]))
    #     # Ensure the label remains horizontally centered on this new x position
    #     # (set_xticklabels defaults to 'center', so this re-affirms it if needed)
    #     label.set_horizontalalignment('center')

    # Set title
    ax.set_title(dataset, fontsize=14)
    
    # Set y-label for all subplots
    if i == 0:
        ax.set_ylabel("Accuracy (\%)", fontsize=12, fontweight="bold")
    else:
        # Hide y-tick labels for non-first subplots to save space
        ax.tick_params(axis='y', labelsize=10)
    
    # Set y-limits based on data range
    all_values = em_values + f1_values
    max_val = max(all_values)
    min_val = min(all_values)
    
    # Special handling for GPQA which has very low values
    if dataset == "GPQA":
        ax.set_ylim(0, 10.0)  # Set a fixed range for GPQA
    else:
        # Reduce the extra space above the bars
        ax.set_ylim(min_val * 0.9, max_val * 1.1) # Adjusted upper limit for text
    
    # Format y-ticks as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: ' {:.0f}'.format(y)))
    
    # Set x-limits to properly space the bars with less blank space
    # ax.set_xlim(group_centers[0] - total_width, group_centers[1] + total_width)
    # Set xlim to be similar to original (0,4) for group_centers (1,3) => margin of 1.0
    ax.set_xlim(group_centers[0] - 1.0, group_centers[1] + 1.0)
    
    # Add a box around the subplot
    # for spine in ax.spines.values():
    #     spine.set_visible(True)
    #     spine.set_linewidth(1.0)
    
    # Add legend to first subplot
    if i == 0:
        ax.legend(
            bbox_to_anchor=(2.21, 1.35), # Adjusted anchor if needed
            ncol=3, # Changed to 3 columns for three labels
            loc="upper center",
            labelspacing=0.1,
            edgecolor="black",
            facecolor="white",
            framealpha=1,
            shadow=False,
            fancybox=False,
            handlelength=1.0,
            handletextpad=0.6,
            columnspacing=0.8,
            prop={"weight": "bold", "size": 12},
        )

# Save figure with tight layout but no additional padding
plt.savefig(FIGURE_PATH + "/accuracy_comparison.pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()