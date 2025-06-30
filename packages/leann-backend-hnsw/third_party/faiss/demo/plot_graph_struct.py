import re
import matplotlib.pyplot as plt
import numpy as np

def extract_data_from_log(log_content):
    """Extract method names, recall lists, and recompute lists from the log file."""
    
    # Regular expressions to find the data - modified to match the actual format
    method_pattern = r"Building HNSW index with ([^\.]+)\.\.\.|Building HNSW index with ([^\n]+)..."
    recall_list_pattern = r"recall_list: (\[[\d\., ]+\])"
    recompute_list_pattern = r"recompute_list: (\[[\d\., ]+\])"
    avg_neighbors_pattern = r"neighbors per node: ([\d\.]+)"
    
    # Find all matches
    method_matches = re.findall(method_pattern, log_content)
    methods = []
    for match in method_matches:
        # Each match is a tuple with one empty string and one with the method name
        method = match[0] if match[0] else match[1]
        methods.append(method)
    
    recall_lists_str = re.findall(recall_list_pattern, log_content)
    recompute_lists_str = re.findall(recompute_list_pattern, log_content)
    avg_neighbors = re.findall(avg_neighbors_pattern, log_content)
    
    # Debug information
    print(f"Found {len(methods)} methods: {methods}")
    print(f"Found {len(recall_lists_str)} recall lists")
    print(f"Found {len(recompute_lists_str)} recompute lists")
    print(f"Found {len(avg_neighbors)} average neighbors values")
    
    # If the regex approach fails, try a more direct approach
    if len(methods) < 5:
        print("Regex approach failed, trying direct extraction...")
        sections = log_content.split("Building HNSW index with ")[1:]
        methods = []
        for section in sections:
            # Extract the method name (everything up to the first newline)
            method_name = section.split("\n")[0].strip()
            # Remove trailing dots if present
            method_name = method_name.rstrip('.')
            methods.append(method_name)
        print(f"Direct extraction found {len(methods)} methods: {methods}")
    
    # Convert string representations of lists to actual lists
    recall_lists = [eval(recall_list) for recall_list in recall_lists_str]
    recompute_lists = [eval(recompute_list) for recompute_list in recompute_lists_str]
    
    # Convert average neighbors to float
    avg_neighbors = [float(avg) for avg in avg_neighbors]
    
    # Make sure we have the same number of items in each list
    min_length = min(len(methods), len(recall_lists), len(recompute_lists), len(avg_neighbors))
    if min_length < 5:
        print(f"Warning: Expected 5 methods, but only found {min_length}")
    
    # Ensure all lists have the same length
    methods = methods[:min_length]
    recall_lists = recall_lists[:min_length]
    recompute_lists = recompute_lists[:min_length]
    avg_neighbors = avg_neighbors[:min_length]
    
    return methods, recall_lists, recompute_lists, avg_neighbors

def plot_performance(methods, recall_lists, recompute_lists, avg_neighbors):
    """Create a plot comparing the performance of different HNSW methods."""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    markers = ['o', 's', '^', 'x', 'd']
    
    for i, method in enumerate(methods):
        # Add average neighbors to the label
        label = f"{method} (avg. {avg_neighbors[i]} neighbors)"
        plt.plot(recompute_lists[i], recall_lists[i], label=label, 
                 color=colors[i], marker=markers[i], markersize=8, markevery=5)
    
    plt.xlabel('Distance Computations', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.title('HNSW Index Performance: Recall vs. Computation Cost', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.ylim(0, 1.0)
    
    # Add horizontal lines for different recall levels
    recall_levels = [0.90, 0.95, 0.96, 0.97, 0.98]
    line_styles = [':', '--', '-.', '-', '-']
    line_widths = [1, 1, 1, 1.5, 1.5]
    
    for i, level in enumerate(recall_levels):
        plt.axhline(y=level, color='gray', linestyle=line_styles[i], 
                   alpha=0.7, linewidth=line_widths[i])
        plt.text(130, level + 0.002, f'{level*100:.0f}% Recall', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('faiss/demo/H_hnsw_performance_comparison.png')
    plt.show()

def plot_recall_comparison(methods, recall_lists, recompute_lists, avg_neighbors):
    """Create a bar chart comparing computation costs at different recall levels."""
    
    recall_levels = [0.90, 0.95, 0.96, 0.97, 0.98]
    
    # Get computation costs for each method at each recall level
    computation_costs = []
    for i, method in enumerate(methods):
        method_costs = []
        for level in recall_levels:
            # Find the first index where recall exceeds the target level
            recall_idx = next((idx for idx, recall in enumerate(recall_lists[i]) if recall >= level), None)
            if recall_idx is not None:
                method_costs.append(recompute_lists[i][recall_idx])
            else:
                # If the method doesn't reach this recall level, use None
                method_costs.append(None)
        computation_costs.append(method_costs)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width of bars
    bar_width = 0.15
    
    # Set positions of the bars on X axis
    r = np.arange(len(recall_levels))
    
    # Colors for each method
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Create bars
    for i, method in enumerate(methods):
        # Filter out None values
        valid_costs = [cost if cost is not None else 0 for cost in computation_costs[i]]
        valid_positions = [pos for pos, cost in zip(r + i*bar_width, computation_costs[i]) if cost is not None]
        valid_costs = [cost for cost in computation_costs[i] if cost is not None]
        
        bars = ax.bar(valid_positions, valid_costs, width=bar_width, 
                     color=colors[i], label=f"{method} (avg. {avg_neighbors[i]} neighbors)")
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                   f'{height:.0f}', ha='center', va='bottom', rotation=0, fontsize=9)
    
    # Add labels and title
    ax.set_xlabel('Recall Level', fontsize=14)
    ax.set_ylabel('Distance Computations', fontsize=14)
    ax.set_title('Computation Cost Required to Achieve Different Recall Levels', fontsize=16)
    
    # Set x-ticks
    ax.set_xticks(r + bar_width * 2)
    ax.set_xticklabels([f'{level*100:.0f}%' for level in recall_levels])
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('faiss/demo/H_hnsw_recall_comparison.png')
    plt.show()

# Read the log file
with open('faiss/demo/output.log', 'r') as f:
    log_content = f.read()

# Extract data
methods, recall_lists, recompute_lists, avg_neighbors = extract_data_from_log(log_content)

# Plot the results
plot_performance(methods, recall_lists, recompute_lists, avg_neighbors)

# Plot the recall comparison
plot_recall_comparison(methods, recall_lists, recompute_lists, avg_neighbors)

# Print a summary of the methods and their characteristics
print("\nMethod Summary:")
for i, method in enumerate(methods):
    print(f"{method}:")
    print(f"  - Average neighbors per node: {avg_neighbors[i]:.2f}")
    
    # Find the recompute values needed for different recall levels
    recall_levels = [0.90, 0.95, 0.96, 0.97, 0.98]
    for level in recall_levels:
        recall_idx = next((idx for idx, recall in enumerate(recall_lists[i]) if recall >= level), None)
        if recall_idx is not None:
            print(f"  - Computations needed for {level*100:.0f}% recall: {recompute_lists[i][recall_idx]:.2f}")
        else:
            print(f"  - Does not reach {level*100:.0f}% recall in the test")
    print()