import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.ticker as mticker # Not actively used
import os

FIGURE_PATH = "paper_plot/figures"

try:
    os.makedirs(FIGURE_PATH, exist_ok=True)
    print(f"Images will be saved to: {os.path.abspath(FIGURE_PATH)}")
except OSError as e:
    print(f"Create {FIGURE_PATH} failed: {e}. Images will be saved in the current working directory.")
    FIGURE_PATH = "."

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True

method_labels = ["gte-small (33M)", "contriever-msmarco (110M)"]
dataset_names = ["NQ", "TriviaQA"]
metrics_plot1 = ["Exact Match", "F1"]

small_nq_f1 = 0.2621040899
small_tq_f1 = 0.4698198059
small_nq_em_score = 0.1845
small_tq_em_score = 0.4015
small_nq_time = 1.137
small_tq_time = 1.173

large_nq_f1 = 0.2841386117
large_tq_f1 = 0.4548340289
large_nq_em_score = 0.206
large_tq_em_score = 0.382
large_nq_time = 2.632
large_tq_time = 2.684

data_scores_plot1 = {
    "NQ": {"Exact Match": [small_nq_em_score, large_nq_em_score], "F1": [small_nq_f1, large_nq_f1]},
    "TriviaQA": {"Exact Match": [small_tq_em_score, large_tq_em_score], "F1": [small_tq_f1, large_tq_f1]}
}
latency_data_plot2 = {
    "NQ": [small_nq_time, large_nq_time],
    "TriviaQA": [small_tq_time, large_tq_time]
}

edgecolors = ["dimgrey", "tomato"]
hatches = ["/////", "\\\\\\\\\\"]

# Changed: bar_center_separation_in_group increased for larger gap
bar_center_separation_in_group = 0.42
# Changed: bar_visual_width decreased for narrower bars
bar_visual_width = 0.28

figsize_plot1 = (4, 2.5)
# Changed: figsize_plot2 width adjusted to match figsize_plot1 for legend/caption alignment
figsize_plot2 = (2.5, 2.5)

# Define plot1_xlim_per_subplot globally so it can be accessed by create_plot2_latency
plot1_xlim_per_subplot = (0.0, 2.0) # Explicit xlim for plot 1 subplots

common_subplots_adjust_params = dict(wspace=0.30, top=0.80, bottom=0.22, left=0.09, right=0.96)


def create_plot1_em_f1():
    fig, axs = plt.subplots(1, 2, figsize=figsize_plot1)
    fig.subplots_adjust(**common_subplots_adjust_params)

    num_methods = len(method_labels)
    metric_group_centers = np.array([0.5, 1.5])
    # plot1_xlim_per_subplot is now global

    for i, dataset_name in enumerate(dataset_names):
        ax = axs[i]
        for metric_idx, metric_name in enumerate(metrics_plot1):
            metric_center_pos = metric_group_centers[metric_idx]
            current_scores_raw = data_scores_plot1[dataset_name][metric_name]
            current_scores_percent = [val * 100 for val in current_scores_raw]

            for j, method_label in enumerate(method_labels):
                offset = (j - (num_methods - 1) / 2.0) * bar_center_separation_in_group
                bar_center_pos = metric_center_pos + offset
                
                ax.bar(
                    bar_center_pos, current_scores_percent[j], width=bar_visual_width, color="white",
                    edgecolor=edgecolors[j], hatch=hatches[j], linewidth=1.5,
                    label=method_label if i == 0 and metric_idx == 0 else None
                )
                ax.text(
                    bar_center_pos, current_scores_percent[j] + 0.8, f"{current_scores_percent[j]:.1f}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold'
                )

        ax.set_xticks(metric_group_centers)
        ax.set_xticklabels(metrics_plot1, fontsize=9, fontweight='bold')
        ax.set_title(dataset_name, fontsize=12, fontweight='bold')
        ax.set_xlim(plot1_xlim_per_subplot) # Apply consistent xlim
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: ' {:.0f}'.format(y)))

        if i == 0:
            ax.set_ylabel("Accuracy (\%)", fontsize=12, fontweight="bold")
        
        all_subplot_scores_percent = []
        for metric_name_iter in metrics_plot1:
            all_subplot_scores_percent.extend([val * 100 for val in data_scores_plot1[dataset_name][metric_name_iter]])
        
        max_val = max(all_subplot_scores_percent) if all_subplot_scores_percent else 0
        ax.set_ylim(0, max_val * 1.22 if max_val > 0 else 10)
        ax.tick_params(axis='y', labelsize=12)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=len(method_labels),
        edgecolor="black", facecolor="white", framealpha=1, shadow=False, fancybox=False,
        handlelength=1.5, handletextpad=0.4, columnspacing=0.8,
        prop={"weight": "bold", "size": 9}
    )
    
    # fig.text(0.5, 0.06, "(a) EM \& F1", ha='center', va='center', fontweight='bold', fontsize=11)


    save_path = os.path.join(FIGURE_PATH, "plot1_em_f1.pdf")
    # plt.tight_layout() # Adjusted call below
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88)) # Adjusted to make space for fig.text and fig.legend
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Figure 1 (Exact Match & F1) has been saved to: {save_path}")

def create_plot2_latency():
    fig, axs = plt.subplots(1, 2, figsize=figsize_plot2) # figsize_plot2 width is now 8.0
    fig.subplots_adjust(**common_subplots_adjust_params)

    num_methods = len(method_labels)
    method_group_center_in_subplot = 0.5 

# Calculate bar extents to determine focused xlim
    bar_positions_calc = []
    for j_idx in range(num_methods):
        offset_calc = (j_idx - (num_methods - 1) / 2.0) * bar_center_separation_in_group
        bar_center_pos_calc = method_group_center_in_subplot + offset_calc
        bar_positions_calc.append(bar_center_pos_calc)
    
    min_bar_actual_edge = min(bar_positions_calc) - bar_visual_width / 2.0
    max_bar_actual_edge = max(bar_positions_calc) + bar_visual_width / 2.0

    # Define padding around the bars
    # Option 1: Fixed padding (e.g., 0.15 as derived from plot 1 visual)
    # padding_val = 0.15 
    # plot2_xlim_calculated = (min_bar_actual_edge - padding_val, max_bar_actual_edge + padding_val)
    # This would be (0.15 - 0.15, 0.85 + 0.15) = (0.0, 1.0)

    # Option 2: Center the group (0.5) in a span of 1.0
    plot2_xlim_calculated = (method_group_center_in_subplot - 0.5, method_group_center_in_subplot + 0.5)
    # This is (0.5 - 0.5, 0.5 + 0.5) = (0.0, 1.0)
    # This is simpler and achieves the (0.0, 1.0) directly.

    for i, dataset_name in enumerate(dataset_names):
        ax = axs[i]
        current_latencies = latency_data_plot2[dataset_name]

        for j, method_label in enumerate(method_labels):
            offset = (j - (num_methods - 1) / 2.0) * bar_center_separation_in_group
            bar_center_pos = method_group_center_in_subplot + offset
            
            ax.bar(
                bar_center_pos, current_latencies[j], width=bar_visual_width, color="white",
                edgecolor=edgecolors[j], hatch=hatches[j], linewidth=1.5,
                label=method_label if i == 0 else None
            )
            ax.text(
                bar_center_pos, current_latencies[j] + 0.05, f"{current_latencies[j]:.2f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: ' {:.0f}'.format(y)))

        ax.set_xticks([0.5])
        ax.set_xticklabels(["Latency"], color="white", fontsize=12)
        # set tick hatches
        ax.tick_params(axis='x', colors="white")
        ax.set_title(dataset_name, fontsize=13, fontweight='bold')
        ax.set_xlim(plot2_xlim_calculated)

        if i == 0:
            ax.set_ylabel("Latency (s)", fontsize=12, fontweight="bold")
        
        max_latency_in_subplot = max(current_latencies) if current_latencies else 0
        ax.set_ylim(0, max_latency_in_subplot * 1.22 if max_latency_in_subplot > 0 else 1) 
        ax.tick_params(axis='y', labelsize=12)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=num_methods,
        edgecolor="black", facecolor="white", framealpha=1, shadow=False, fancybox=False,
        handlelength=1.5, handletextpad=0.4, columnspacing=0.8,
        prop={"weight": "bold", "size": 9}
    )

    # fig.text(0.5, 0.06, "(b) Latency", ha='center', va='center', fontweight='bold', fontsize=11)

    save_path = os.path.join(FIGURE_PATH, "plot2_latency.pdf")
    # plt.tight_layout() # Adjusted call below
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88)) # Adjusted to make space for fig.text and fig.legend
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Figure 2 (Latency) has been saved to: {save_path}")

if __name__ == "__main__":
    print("Start generating figures...")
    if plt.rcParams["text.usetex"]:
        print("Info: LaTeX rendering is enabled. Ensure LaTeX is installed and configured if issues arise, or set plt.rcParams['text.usetex'] to False.")
    
    create_plot1_em_f1()
    create_plot2_latency()
    print("All figures have been generated.")
