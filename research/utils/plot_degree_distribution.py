import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_degree_distribution(degree_file_path: str, output_image_path: str):
    """
    Reads a file containing node degrees (one per line) and plots the
    degree distribution as a histogram.

    Args:
        degree_file_path: Path to the file containing degrees.
        output_image_path: Path to save the output plot image.
    """
    try:
        # Read degrees from the file
        degrees = np.loadtxt(degree_file_path, dtype=int)
        print(f"[LOG] Read {len(degrees)} degrees from {degree_file_path}")

        if len(degrees) == 0:
            print("[WARN] Degree file is empty. No plot generated.")
            return

        # Calculate basic statistics
        min_deg = np.min(degrees)
        max_deg = np.max(degrees)
        avg_deg = np.mean(degrees)
        median_deg = np.median(degrees)

        print(f"[LOG] Degree Stats: Min={min_deg}, Max={max_deg}, Avg={avg_deg:.2f}, Median={median_deg}")

        # Plotting the distribution
        plt.figure(figsize=(10, 6))
        # Determine appropriate number of bins, maybe max_deg+1 if not too large
        # Or use automatic binning like 'auto' or Sturges' rule etc.
        # Using max_deg - min_deg + 1 bins can be too many if the range is large
        # Let's try 'auto' binning first
        n_bins = 'auto'
        # If max_deg is reasonably small, we can use exact bins
        if max_deg <= 1000: # Heuristic threshold
             n_bins = max_deg - min_deg + 1

        counts, bin_edges, patches = plt.hist(degrees, bins=n_bins, edgecolor='black', alpha=0.7)
        plt.xlabel("Node Degree")
        plt.ylabel("Number of Nodes")
        plt.title(f"Degree Distribution (from {os.path.basename(degree_file_path)})")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add text for statistics on the plot
        stats_text = (
            f"Total Nodes: {len(degrees)}\n"
            f"Min Degree: {min_deg}\n"
            f"Max Degree: {max_deg}\n"
            f"Avg Degree: {avg_deg:.2f}\n"
            f"Median Degree: {median_deg}"
        )
        # Position the text box; adjust as needed
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


        plt.tight_layout()
        plt.savefig(output_image_path)
        print(f"[LOG] Degree distribution plot saved to {output_image_path}")
        # plt.show() # Uncomment if you want to display the plot interactively
        
        # Create weighted degree distribution plot
        plt.figure(figsize=(10, 6))
        # Calculate weighted distribution (degree * number of nodes)
        unique_degrees, degree_counts = np.unique(degrees, return_counts=True)
        weighted_counts = unique_degrees * degree_counts
        
        plt.bar(unique_degrees, weighted_counts, edgecolor='black', alpha=0.7)
        plt.xlabel("Node Degree")
        plt.ylabel("Degree × Number of Nodes")
        plt.title(f"Weighted Degree Distribution (from {os.path.basename(degree_file_path)})")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text for statistics on the plot
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        
        # Generate weighted output filename based on the original output path
        weighted_output_path = os.path.splitext(output_image_path)[0] + "_weighted" + os.path.splitext(output_image_path)[1]
        
        plt.tight_layout()
        plt.savefig(weighted_output_path)
        print(f"[LOG] Weighted degree distribution plot saved to {weighted_output_path}")

    except FileNotFoundError:
        print(f"[ERROR] Degree file not found: {degree_file_path}")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the degree distribution from a file containing node degrees."
    )
    parser.add_argument(
        "degree_file",
        type=str,
        help="Path to the input file containing node degrees (one degree per line)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="degree_distribution.png",
        help="Path to save the output plot image (default: degree_distribution.png)."
    )
    args = parser.parse_args()

    plot_degree_distribution(args.degree_file, args.output) 