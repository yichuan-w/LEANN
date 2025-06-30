import argparse
import re
import matplotlib.pyplot as plt
import os


def parse_log(log_file_path):
    """
    Parses the log file to extract relevant data for accuracy-recall curve comparison.
    Args:
        log_file_path (str): Path to the log file to parse.
    Returns:
        dict: A dictionary containing extracted results.
    """
    data = {
        "recalls_with_skip": [],
        "f1_scores_with_skip": [],
        "exact_match_scores_with_skip": [],
        "recalls_without_skip": [],
        "f1_scores_without_skip": [],
        "exact_match_scores_without_skip": [],
        "nprobe_values": [],
    }

    with open(log_file_path, "r") as file:
        logs = file.readlines()

    # Variables to track the state during parsing
    is_skip_reorder_true = False
    is_skip_reorder_false = False
    current_nprobe = None

    for line in logs:
        # Debug: print the current line being processed
        # print(f"Processing line: {line.strip()}")

        # Check for skip_reorder flag
        if "skip_search_reorder=True" in line:
            is_skip_reorder_true = True
            is_skip_reorder_false = False
        elif "skip_search_reorder=False" in line:
            is_skip_reorder_true = False
            is_skip_reorder_false = True

        # Extract nprobe values (assuming they are given before the experiment)
        nprobe_match = re.search(r"nprobe=(\d+)", line)
        if nprobe_match:
            current_nprobe = int(nprobe_match.group(1))
            if current_nprobe not in data["nprobe_values"]:
                data["nprobe_values"].append(current_nprobe)
                print(f"Found nprobe value: {current_nprobe}")

        # Extract average recall rate
        avg_recall_match = re.search(
            r"Avg recall rate for (flat|diskann): ([0-9\.e\-]+)", line
        )
        if avg_recall_match:
            recall_value = float(avg_recall_match.group(2))
            print(
                f"Found avg recall rate: {recall_value} for {avg_recall_match.group(1)} in line {line!r}"
            )

            if "flat" in avg_recall_match.group(1):
                # data["recalls_without_skip"].append(recall_value)
                pass
            elif "diskann" in avg_recall_match.group(1):
                if is_skip_reorder_true:
                    data["recalls_with_skip"].append(recall_value)
                elif is_skip_reorder_false:
                    data["recalls_without_skip"].append(recall_value)

        # Extract exact_match, f1, and recall scores from evaluation results
        eval_match = re.search(
            r"\{'exact_match': ([0-9\.]+), 'exact_match_stderr': [0-9\.]+, 'f1': ([0-9\.]+), 'f1_stderr': [0-9\.]+",
            line,
        )
        if eval_match:
            exact_match = float(eval_match.group(1))
            f1 = float(eval_match.group(2))

            print(f"Found evaluation results -> Exact Match: {exact_match}, F1: {f1}")

            # Add to appropriate list based on skip_reorder flag
            if is_skip_reorder_true:
                data["exact_match_scores_with_skip"].append(exact_match)
                data["f1_scores_with_skip"].append(f1)
            elif is_skip_reorder_false:
                data["exact_match_scores_without_skip"].append(exact_match)
                data["f1_scores_without_skip"].append(f1)

    return data


def plot_skip_reorder_comparison(data, output_dir):
    """
    绘制带有和不带 skip_reorder 参数的准确率-召回率曲线。

    Args:
        data: The parsed data including recalls, f1 scores, and exact match scores.
        output_dir: Path where the plot will be saved.
    """
    recalls_with_skip = data["recalls_with_skip"]
    f1_scores_with_skip = data["f1_scores_with_skip"]
    exact_match_scores_with_skip = data["exact_match_scores_with_skip"]
    recalls_without_skip = data["recalls_without_skip"]
    f1_scores_without_skip = data["f1_scores_without_skip"]
    exact_match_scores_without_skip = data["exact_match_scores_without_skip"]
    nprobe_values = data["nprobe_values"]

    plt.figure(figsize=(10, 6))

    # Check if data lists are not empty and have the same length before plotting
    if (
        recalls_with_skip
        and len(recalls_with_skip) == len(f1_scores_with_skip)
        and len(recalls_with_skip) == len(exact_match_scores_with_skip)
    ):
        plt.plot(
            recalls_with_skip,
            f1_scores_with_skip,
            "bo-",
            label="F1 Score (with skip_reorder)",
            markersize=8,
            linewidth=2,
        )
        plt.plot(
            recalls_with_skip,
            exact_match_scores_with_skip,
            "rs-",
            label="Exact Match (with skip_reorder)",
            markersize=8,
            linewidth=2,
        )

    if (
        recalls_without_skip
        and len(recalls_without_skip) == len(f1_scores_without_skip)
        and len(recalls_without_skip) == len(exact_match_scores_without_skip)
    ):
        plt.plot(
            recalls_without_skip,
            f1_scores_without_skip,
            "go-",
            label="F1 Score (without skip_reorder)",
            markersize=8,
            linewidth=2,
        )
        plt.plot(
            recalls_without_skip,
            exact_match_scores_without_skip,
            "ms-",
            label="Exact Match (without skip_reorder)",
            markersize=8,
            linewidth=2,
        )

    plt.xlabel("Recall")
    plt.ylabel("Score")
    plt.title("Recall vs Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.xlim(0.0, 1.0)

    # Save the plot only if data is present
    if len(nprobe_values) > 0:
        plot_path = os.path.join(
            output_dir,
            f"recall_vs_acc_comparison.png",
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    else:
        print("No valid data to plot.")

    plt.close()


parser = argparse.ArgumentParser(description="Parse log file and plot results")
parser.add_argument(
    "log_file_path", type=str, help="Path to the log file"
)
parser.add_argument(
    "--output_dir", type=str, help="Path to the output directory", default="skip_reorder_comparison"
)
args = parser.parse_args()

# Parse the log
parsed_data = parse_log(args.log_file_path)

print(parsed_data)

# Plot the data
plot_skip_reorder_comparison(parsed_data, args.output_dir)
