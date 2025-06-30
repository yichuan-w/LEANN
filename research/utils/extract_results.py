import os
import math
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import pickle
import pdb


"""
Automatic result extraction for BM25.
"""


def extract_data_to_table(directory_path):
    # Regular expression pattern to match the data format in file content
    content_pattern = (
        r"# tokens: (\d+(\.\d+)?)\tLM PPL: (\d+(\.\d+)?)\tPPL: (\d+(\.\d+)?)"
    )
    # Regular expression pattern to extract info from file names
    file_name_pattern_M = r"(.+)-(\d+)M-seed_(\d+).txt"
    file_name_pattern = r"(.+)-(\d+)-seed_(\d+).txt"

    # Data storage
    data = []

    # Iterating through each file in the directory
    for file_name in os.listdir(directory_path):
        # Checking if the file name matches the pattern
        file_match_M = re.match(file_name_pattern_M, file_name)
        file_match = re.match(file_name_pattern, file_name)
        if file_match_M:
            domain, num_samples, seed = file_match_M.groups()

            # Reading the file and extracting data
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "r") as file:
                for line in file:
                    # Searching for the pattern in each line
                    content_match = re.search(content_pattern, line)
                    if content_match:
                        # Extracting values
                        tokens, lm_ppl, ppl = (
                            content_match.groups()[0],
                            content_match.groups()[2],
                            content_match.groups()[4],
                        )
                        # Adding the extracted data and extra info to the list
                        data.append(
                            {
                                "Domain": domain,
                                "Samples": int(num_samples) * 1e6,
                                "Seed": int(seed),
                                "#eval_tokens": float(tokens),
                                "LM_PPL": float(lm_ppl),
                                "PPL": float(ppl),
                            }
                        )
        elif file_match:
            domain, num_samples, seed = file_match.groups()

            # Reading the file and extracting data
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "r") as file:
                for line in file:
                    # Searching for the pattern in each line
                    content_match = re.search(content_pattern, line)
                    if content_match:
                        # Extracting values
                        tokens, lm_ppl, ppl = (
                            content_match.groups()[0],
                            content_match.groups()[2],
                            content_match.groups()[4],
                        )
                        # Adding the extracted data and extra info to the list
                        data.append(
                            {
                                "Domain": domain,
                                "Samples": int(num_samples),
                                "Seed": int(seed),
                                "#eval_tokens": float(tokens),
                                "LM_PPL": float(lm_ppl),
                                "PPL": float(ppl),
                            }
                        )

    df = pd.DataFrame(data)
    grouped_df = df.groupby(["Domain", "Samples", "#eval_tokens"]).mean()

    return df, grouped_df


"""
Automatic resutls extraction for dense retrieval. (new)
"""


def extract_dense_scaling_results(log_files, domain=None, plot=None):
    # Regular expression pattern to match the key-value pairs in the input string
    pattern = r"(\w[\w #]+) = ([\w.]+)"

    data_list = []
    for file in log_files:
        with open(file, "r") as file:
            for line in file:
                # Use re.findall to extract all matches of the pattern
                matches = re.findall(pattern, line)

                if matches:
                    data_dict = {
                        key.replace(" ", "_").lower(): (
                            None
                            if value == "None"
                            else float(value)
                            if value.replace(".", "", 1).isdigit()
                            else value
                        )
                        for key, value in matches
                    }
                    data_list.append(data_dict)

    df = pd.DataFrame(data_list)
    if "total_shards" in df.columns:
        df["subsample_ratio"] = df["sampled_shards"] / df["total_shards"]
    else:
        df["subsample_ratio"] = 1 / df["total_shards"]
    df = df.sort_values(by="subsample_ratio")
    print(df.head)

    if plot:
        # Setting the plot size for better visibility
        plt.figure(figsize=(10, 6))

        # Plotting
        for concate_k in df["concate_k"].unique():
            subset = df[df["concate_k"] == concate_k]
            if concate_k == 0:
                perplexity_when_concate_k_0 = subset["perplexity"].mean()
                plt.axhline(
                    y=perplexity_when_concate_k_0,
                    color="r",
                    linestyle="-",
                    label="Closed-book",
                )
            else:
                plt.plot(
                    subset["subsample_ratio"],
                    subset["perplexity"],
                    label=f"Concate_k = {concate_k}",
                )

        plt.title(f"Perplexity Change with Total Shards -- {domain}")
        plt.xlabel("Subsample Ratio")
        plt.ylabel("Perplexity")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot)
    return df


def plot_mmlu():
    # C4 results
    labels = [
        "LM-only",
        "top-1 w/ 1/32 C4 datastore",
        "top-1 w/ 2/32 C4 datastore",
        "top-1 w/ 3/32 C4 datastore",
        "top-1 w/ 4/32 C4 datastore",
        "top-1 w/ 5/32 C4 datastore",
        "top-1 w/ 6/32 C4 datastore",
    ]
    x = [0, 1, 2, 3, 4, 5, 6]
    few_shot_0_concat_1 = [30.69, 32.81, 32.05, 32.55, 32.57, 33.03, 32.88]
    few_shot_1_concat_1 = [39.67, 41.03, 41.74, 42.1, 42.62, 41.55, 42.09]
    few_shot_5_concat_1 = [42.47, 43.75, 44.37, 44.1, 44.84, 43.95, 44.49]

    # Plotting the data
    plt.figure(figsize=(14, 8))

    # Plot for few_shot_0_concat_1
    plt.plot(
        x,
        few_shot_0_concat_1,
        marker="o",
        linestyle="-",
        color="blue",
        label="Few-shot k=0, Concat k=1",
    )

    # Plot for few_shot_1_concat_1
    plt.plot(
        x,
        few_shot_1_concat_1,
        marker="s",
        linestyle="-",
        color="red",
        label="Few-shot k=1, Concat k=1",
    )

    # Plot for few_shot_5_concat_1
    plt.plot(
        x,
        few_shot_5_concat_1,
        marker="^",
        linestyle="-",
        color="green",
        label="Few-shot k=5, Concat k=1",
    )

    # Adding details
    plt.title("MMLU Performance")
    plt.xlabel("Retrieval-based LM Datastore Composition")
    plt.ylabel("Accuracy")
    plt.xticks(ticks=x, labels=labels, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("mmlu_c4_scaling.png")


def extract_lm_eval_results(
    result_dir, task_name, model_name, n_shot_list, n_doc_list, datastore_name_filter=""
):
    markers = ["o", "s", "^", "D", "*", "p", "H", "x"]
    colors = plt.cm.tab20.colors

    all_data = []
    for subdir, dirs, files in os.walk(result_dir):
        num_ints = len(os.path.basename(subdir).split("-"))
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        data["SubdirLevel"] = num_ints
                        data["n-shot"], data["n-doc"] = (
                            int(data["n-shot"]),
                            int(data["n-doc"]),
                        )
                        data["Value"] = float(data["Value"])
                        all_data.append(data)

    filtered_data = [
        d
        for d in all_data
        if datastore_name_filter in result_dir
        and d["n-shot"] in n_shot_list
        and d["n-doc"] in n_doc_list
        and d["SubdirLevel"] > 0
    ]

    plot_data = {}
    for d in filtered_data:
        key = (d["n-shot"], d["n-doc"])
        plot_data.setdefault(key, []).append((d["SubdirLevel"], d["Value"]))

    sorted_keys = sorted(plot_data.keys(), key=lambda x: (x[0], x[1]))

    closed_book_values = {}
    for i, key in enumerate(sorted_keys):
        n_shot, n_doc = key
        if n_doc == 0:
            value = plot_data[key][-1][-1]
            closed_book_values.update({n_shot: value})

    plt.figure(figsize=(15, 10))
    for i, key in enumerate(sorted_keys):
        n_shot, n_doc = key
        if n_doc == 0:
            continue
        values = plot_data[key]
        values.append(
            (0, closed_book_values[n_shot])
            if n_shot in closed_book_values.keys()
            else (0, None)
        )
        values.sort()  # Ensure the values are sorted by SubdirLevel
        x_values, y_values = zip(*values)  # Unzip the tuple pairs to separate lists
        marker = markers[n_shot] if n_doc else ""
        color = colors[i % len(colors)]  # Choose a color from the colormap
        label = f"n-shot={n_shot}, n-doc={n_doc}"
        plt.plot(
            x_values, y_values, marker=marker, color=color, linestyle="-", label=label
        )

    # plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', steps=[1, 2, 5, 10]))

    if subject_name == "mmlu":
        plot_dir = os.path.join("plots", "mmlu")
    else:
        plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    plt.xlabel("Number of Index Shards")
    plt.ylabel("Accuracy")
    plt.title(f"{task_name} scaling performance with {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/{task_name}_{model_name}.png")

    return all_data


def plot_mmlu_persub_figures(directory="plots"):
    files = [
        file
        for file in os.listdir(directory)
        if file.startswith("mmlu_") and file.endswith(".png")
    ]
    plots_per_figure = 16
    for i in range(0, len(files), plots_per_figure):
        # Create a new figure
        fig, axs = plt.subplots(4, 4, figsize=(20, 20))

        # Flatten the axis array for easy indexing
        axs = axs.flatten()

        # Iterate over each subplot in the current figure
        for ax, file in zip(axs, files[i : i + plots_per_figure]):
            # Read the image file
            img = plt.imread(os.path.join(directory, file))

            # Display the image in the subplot
            ax.imshow(img)
            ax.set_title(file)
            ax.axis("off")  # Hide axes

        # Adjust layout and display the figure
        plt.tight_layout()
        plt.savefig(f"mmlu_persub_{i}.png")


def plot_calibration_figures(domain, shard_id=8, show_ci=True, show_all_points=False):
    if show_all_points:
        show_ci = False

    data_path = f"out_calibration/{shard_id}_shard_{domain}/calibration_results_decon_rpj_{domain}_None_samples.pkl"

    with open(data_path, "rb") as file:
        all_results = pickle.load(file)

    all_lm_losses = [item[0] for item in all_results]
    all_retrieval_scores = [item[1] for item in all_results]
    print(f"Total {len(all_lm_losses)} examples.")

    # Compute PPL of top-1 doc v.s. golden doc from top-100
    losses_top1 = [losses[0] for losses in all_lm_losses]
    avg_losses_top1 = sum(losses_top1) / len(losses_top1)
    ppl_losses_top1 = math.exp(avg_losses_top1)

    lossed_top100_gold = [min(losses) for losses in all_lm_losses]
    avg_losses_top100_gold = sum(lossed_top100_gold) / len(lossed_top100_gold)
    ppl_lossed_top100_gold = math.exp(avg_losses_top100_gold)

    print(
        f"Top-1 doc PPL: {ppl_losses_top1:.4f}\nGold doc from top-100 PPL: {ppl_lossed_top100_gold:.4f}"
    )

    # Calibration plot
    lm_losses = np.array(all_lm_losses)
    retrieval_scores = np.array(all_retrieval_scores)

    from scipy.special import softmax
    import scipy.stats as stats

    softmax_lm_losses = softmax(lm_losses, axis=1)
    softmax_retrieval_scores = softmax(retrieval_scores, axis=1)

    if show_all_points:
        lm_losses = lm_losses.flatten()
        retrieval_scores = retrieval_scores.flatten()

        plt.figure(figsize=(8, 6))
        plt.plot(lm_losses, retrieval_scores, marker="o", linestyle="")
        plt.title(f"Calibration Curve with {shard_id} Shards")
        plt.xlabel("LM Losses")
        plt.ylabel("Retrieval Scores")
        plt.grid(True)
        plt.savefig(f"out_calibration/calibration_all_{shard_id}_shard_{domain}.png")

    elif show_ci:
        lm_losses_mean = np.mean(lm_losses, axis=0)
        retrieval_scores_mean = np.mean(retrieval_scores, axis=0)

        lm_losses_sem = stats.sem(lm_losses, axis=0)
        retrieval_scores_sem = stats.sem(retrieval_scores, axis=0)

        # Assuming a 95% confidence interval, z-score is approximately 1.96 for a normal distribution
        z_score = 1.96
        losses_ci = lm_losses_sem * z_score
        retrieval_ci = retrieval_scores_sem * z_score

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            lm_losses_mean,
            retrieval_scores_mean,
            xerr=losses_ci,
            yerr=retrieval_ci,
            fmt="o",
            ecolor="lightgray",
            alpha=0.5,
            capsize=5,
        )
        plt.xlabel("LM Losses")
        plt.ylabel("Retrieval Scores")
        plt.title(
            f"Calibration plot for {shard_id}-shard {domain} with Confidence Intervals"
        )
        plt.grid(True)
        plt.savefig(f"out_calibration/calibration_ci_{shard_id}_shard_{domain}.png")

    else:
        lm_losses = np.mean(lm_losses, axis=0)
        retrieval_scores = np.mean(retrieval_scores, axis=0)

        plt.figure(figsize=(8, 6))
        plt.plot(lm_losses, retrieval_scores, marker="o", linestyle="")
        plt.title(f"Calibration Curve with {shard_id} Shards")
        plt.xlabel("LM Losses")
        plt.ylabel("Retrieval Scores")
        plt.grid(True)
        plt.savefig(f"out_calibration/calibration_{shard_id}_shard_{domain}.png")

    return ppl_losses_top1, ppl_lossed_top100_gold, all_lm_losses, all_retrieval_scores


def plot_top1_vs_best_doc(domain, total_shards=8):
    lm_only_ppl = {
        "books": 21.5250,
        "stackexchange": 11.5948,
        "wiki": 14.0729,
    }

    top1_losses, best_losses = [], []
    for shard_id in range(1, total_shards + 1):
        top1_loss, best_loss, _, _ = plot_calibration_figures(domain, shard_id)
        top1_losses.append(top1_loss)
        best_losses.append(best_loss)

    x = [i for i in range(1, total_shards + 1)]
    plt.figure(figsize=(10, 6))

    # Plotting
    if lm_only_ppl[domain]:
        plt.axhline(
            y=lm_only_ppl[domain], color="r", linestyle="-", label="Closed-book"
        )

    plt.plot(x, top1_losses, label=f"Top-1 Doc")
    plt.plot(x, best_losses, label=f"Gold Doc")

    plt.title(f"Perplexity Change with Total Shards")
    plt.xlabel("Number of Shards")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"best_plot_{domain}.png")


def plot_top1_vs_best_doc_per_sample(domain, shard_id, show_top_k=10, special_mark_k=0):
    _, _, all_lm_losses, all_retrieval_scores = plot_calibration_figures(
        domain, shard_id
    )
    all_sorted_lm_losses, all_sorted_retrieval_scores = [], []
    for lm_losses, retrieval_scores in zip(all_retrieval_scores, all_lm_losses):
        sorted_scores, sorted_losses = zip(
            *sorted(zip(retrieval_scores, lm_losses), reverse=True)
        )
        all_sorted_lm_losses.append(sorted_losses)
        all_sorted_retrieval_scores.append(sorted_scores)

    num_samples = len(all_lm_losses)
    x = [i for i in range(num_samples)]
    plt.figure(figsize=(25, 6))

    # Plotting
    for i in range(show_top_k - 1, -1, -1):
        plt.plot(
            x,
            [losses[i] for losses in all_sorted_lm_losses],
            label=f"Top-{i + 1}th Doc",
            marker="x" if i == special_mark_k else "o",
            linestyle="",
        )

    plt.title(f"Per-sample Loss of {domain} with 1 retrieved doc")
    plt.xlabel("Index of the Evaluation Sample")
    plt.ylabel("Loss")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(f"per_sample_{domain}.png")


def compute_variance_across_hards(path, n_shot=5, n_doc=3):
    all_data = []
    for subdir, dirs, files in os.walk(path):
        num_ints = len(os.path.basename(subdir).split("-"))
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        data["SubdirLevel"] = num_ints
                        data["n-shot"], data["n-doc"] = (
                            int(data["n-shot"]),
                            int(data["n-doc"]),
                        )
                        data["Value"] = float(data["Value"])
                        all_data.append(data)

    plot_data = {}
    for d in all_data:
        key = (d["n-shot"], d["n-doc"])
        plot_data.setdefault(key, []).append((d["SubdirLevel"], d["Value"]))

    files_end = [d.split("/")[-1] for d, _, _ in os.walk(path)]
    shard_ids = [int(i) for i in files_end[1:]]
    key = n_shot, n_doc
    values = plot_data[key]
    _, y_values = zip(*values)

    plt.figure(figsize=(10, 6))
    try:
        plt.plot(shard_ids, y_values, marker="o", linestyle="")
    except:
        print(f"mismatched size for {key}: {len(shard_ids)}, {len(y_values)}")

    print(y_values)
    print(f"Saving to {f'per_sample_{files_end[0]}.png'}")

    plt.xlabel("Single-shard Index ID")
    plt.ylabel("PPL")
    plt.grid(True)
    plt.savefig(f"per_sample_{files_end[0]}.png")


if __name__ == "__main__":
    # # Replace with your directory path
    # directory_path = "out/2023_dec_25_single_domain"

    # # Extracting data to a table with additional information
    # df, grouped_df = extract_data_to_table(directory_path)
    # print(grouped_df)
    # print(grouped_df.index.get_level_values("Samples (M)").to_numpy())

    plot_info_list = [
        # {'logfile': 'rpj_c4.log', 'domain': 'rpj-c4', 'plot': 'scaling_c4_single_index_plot.png'},
        # {'logfile': 'rpj_arxiv.log', 'domain': 'rpj-arxiv', 'plot': 'scaling_arxiv_plot.png'},
        # {'logfile': 'rpj_book_scaling.log', 'domain': 'rpj-book', 'plot': 'scaling_book_plot.png'},
        # {'logfile': 'rpj_github_scaling.log', 'domain': 'rpj-github', 'plot': 'scaling_github_plot.png'},
        # {'logfile': 'rpj_stackexchange_scaling.log', 'domain': 'rpj-stackexchange', 'plot': 'scaling_stackexchange_plot.png'},
        # {'logfile': 'rpj_wiki.log', 'domain': 'rpj-wiki', 'plot': 'scaling_wiki_plot.png'},
        # {'logfile': 'out/2024_apr_decon/decon_rpj_wiki_contriever_ppl.log', 'domain': 'rpj-wiki-decon-contriever', 'plot': 'scaling_wiki_decon_plot_contriever.png'},
        # {'logfile': 'out/2024_apr_decon/decon_rpj_book_contriever_ppl.log', 'domain': 'rpj-book-decon-contriever', 'plot': 'scaling_book_decon_plot_contriever.png'},
        # {'logfile': 'out/2024_apr_decon/decon_rpj_arxiv_contriever_ppl.log', 'domain': 'rpj-arxiv-decon-contriever', 'plot': 'scaling_arxiv_decon_plot_contriever.png'},
        # {'logfile': 'out/2024_apr_decon/decon_rpj_stackexchange_contriever_ppl.log', 'domain': 'rpj-stackexchange-decon-contriever', 'plot': 'scaling_stackexchange_decon_plot_contriever.png'},
        # {'logfile': 'out/2024_apr_decon/decon_rpj_stackexchange_dragon_ppl.log', 'domain': 'rpj-stackexchange-decon-dragon', 'plot': 'scaling_stackexchange_decon_plot_dragon.png'},
        # {'logfile': 'out/2024_apr_decon/decon_rpj_wiki_dragon_ppl.log', 'domain': 'rpj-wiki-decon-dragon', 'plot': 'scaling_wiki_decon_plot_dragon.png'},
        # {'logfile': 'out/2024_apr_decon/decon_rpj_arxiv_dragon_ppl.log', 'domain': 'rpj-arxiv-decon-dragon', 'plot': 'scaling_arxiv_decon_plot_dragon.png'},
        # {'logfile': 'out/2024_apr_decon/decon_rpj_book_dragon_ppl.log', 'domain': 'rpj-book-decon-dragon', 'plot': 'scaling_book_decon_plot_dragon.png'},
    ]

    # for plot_info in plot_info_list:
    #     extract_dense_scaling_results([plot_info['logfile']], plot_info['domain'], plot_info['plot'])

    model_name = "lclm"
    subject_name = "gsm8k"
    datastore_name = "c4"
    result_dir = f"/gscratch/zlab/rulins/Scaling/lm_eval_results/{model_name}"

    all_subjects = [
        file
        for file in os.listdir(result_dir)
        if subject_name in file and datastore_name in file
    ]
    for subject in all_subjects:
        file_name = subject
        print(file_name)
        extract_lm_eval_results(
            os.path.join(result_dir, file_name),
            subject,
            model_name,
            [0, 5],  # few-shot
            [0, 3],  # n-doc
            file_name,
        )

    # plot_mmlu_persub_figures("plots/mmlu")

    # compute_variance_across_hards(f'/gscratch/zlab/rulins/Scaling/lm_eval_results/llama2-7b/subsample/nq_open-rpj_c4-32_shards')
    # compute_variance_across_hards(f'/gscratch/zlab/rulins/Scaling/lm_eval_results/llama2-7b/subsample/medqa_4options-rpj_c4-32_shards')

    # plot_calibration_figures(domain='wiki', shard_id=1, show_all_points=True)
    # plot_top1_vs_best_doc_per_sample(domain='stackexchange', shard_id=1, show_top_k=10, special_mark_k=0)
