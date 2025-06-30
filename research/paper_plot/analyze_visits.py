#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /hnsw_degree_visit_plot_binned_academic.py
# \brief: Generates a binned bar plot of HNSW node average per-query visit probability
#         per degree bin, styled for academic publications, with caching.
# Author: raphael hao (Original script by user, styling and caching adapted by Gemini)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import os # For robust filepath manipulation
import math # For calculating scaling factor
import pickle # For caching data

# %%
# --- Matplotlib parameters for academic paper style (from reference) ---
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1.5
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True # Use LaTeX for text rendering (if available)

# --- Define styles from reference ---
edgecolors_ref = ["dimgrey", "#63B8B6", "tomato", "silver", "slategray"]

# %%
# --- File Paths ---
degree_file = '/opt/dlami/nvme/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/hnsw/degree_distribution.txt'
visit_log_file = './re.log'
output_image_file = './paper_plot/figures/hnsw_visit_count_per_degree_corrected.pdf'
# --- CACHE FILE PATH: Keep this consistent ---
CACHE_FILE_PATH = './binned_plot_data_cache.pkl'

# --- Configuration ---
# Set to True to bypass cache and force recomputation.
# Otherwise, delete CACHE_FILE_PATH manually to force recomputation.
FORCE_RECOMPUTE = False
NUMBER_OF_QUERIES = 1000.0 # Number of queries the visit_counts are based on

# Create directory for figures if it doesn't exist
output_dir = os.path.dirname(output_image_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# %%
# --- Attempt to load data from cache or compute ---
df_plot_data = None
bin_size_for_plot = None # Will hold the bin_size associated with df_plot_data

if not FORCE_RECOMPUTE and os.path.exists(CACHE_FILE_PATH):
    try:
        with open(CACHE_FILE_PATH, 'rb') as f:
            cache_content = pickle.load(f)
            df_plot_data = cache_content['data']
            bin_size_for_plot = cache_content['bin_size']
        # Basic validation of cached data
        # Expecting 'average_visit_count_per_node_in_bin' (raw average over NUMBER_OF_QUERIES)
        if not isinstance(df_plot_data, pd.DataFrame) or \
           'degree_bin_label' not in df_plot_data.columns or \
           'average_visit_count_per_node_in_bin' not in df_plot_data.columns or \
           not isinstance(bin_size_for_plot, int):
            print("Cached data is not in the expected format or missing 'average_visit_count_per_node_in_bin'. Recomputing.")
            df_plot_data = None # Invalidate to trigger recomputation
        else:
            print(f"Successfully loaded binned data from cache: {CACHE_FILE_PATH}")

        # --- Modify the label loaded from cache for display purpose ---
        # This modification only happens when data is loaded from cache and meets specific conditions.
        # Assumption: If the bin_size_for_plot in cache is 5,
        # then the original label "0-4" actually represents nodes with degree 1-4 (because you guarantee no 0-degree nodes).
        if df_plot_data is not None and 'degree_bin_label' in df_plot_data.columns and bin_size_for_plot == 5:
            # Check if "0-4" label exists
            if '0-4' in df_plot_data['degree_bin_label'].values:
                # Use .loc to ensure the modification is on the original DataFrame
                df_plot_data.loc[df_plot_data['degree_bin_label'] == '0-4', 'degree_bin_label'] = '1-4'
                print("Modified degree_bin_label from '0-4' to '1-4' for display purpose.")
    except Exception as e:
        print(f"Error loading from cache: {e}. Recomputing.")
        df_plot_data = None # Invalidate to trigger recomputation

if df_plot_data is None:
    print("Cache not found, invalid, or recompute forced. Computing data from scratch...")
    # --- 1. Read Degree Distribution File ---
    degrees_data = []
    try:
        with open(degree_file, 'r') as f:
            for i, line in enumerate(f):
                line_stripped = line.strip()
                if line_stripped:
                    degrees_data.append({'node_id': i, 'degree': int(line_stripped)})
    except FileNotFoundError:
        print(f"Error: Degree file '{degree_file}' not found. Using dummy data for degrees.")
        degrees_data = [{'node_id': i, 'degree': (i % 20) + 1 } for i in range(200)]
        degrees_data.extend([{'node_id': 200+i, 'degree': i} for i in range(58, 67)]) # For 60-64 bin
        degrees_data.extend([{'node_id': 300+i, 'degree': (i % 5)+1} for i in range(10)]) # Low degrees
        degrees_data.extend([{'node_id': 400+i, 'degree': 80 + (i%5)} for i in range(10)]) # High degrees


    if not degrees_data:
        print(f"Critical Error: No data loaded or generated for degrees. Exiting.")
        exit()
    df_degrees = pd.DataFrame(degrees_data)
    print(f"Successfully loaded/generated {len(df_degrees)} degree entries.")

    # --- 2. Read Visit Log File and Count Frequencies ---
    visit_counts = Counter()
    node_id_pattern = re.compile(r"Vis(i)?ted node: (\d+)")
    try:
        with open(visit_log_file, 'r') as f_log:
            for line_num, line in enumerate(f_log, 1):
                match = node_id_pattern.search(line)
                if match:
                    try:
                        node_id = int(match.group(2))
                        visit_counts[node_id] += 1 # Increment visit count for the node
                    except ValueError:
                        print(f"Warning: Non-integer node_id in log '{visit_log_file}' line {line_num}: {line.strip()}")
    except FileNotFoundError:
        print(f"Warning: Visit log file '{visit_log_file}' not found. Using dummy visit counts.")
        if not df_degrees.empty:
            for node_id_val in df_degrees['node_id'].sample(frac=0.9, random_state=1234): # Seed for reproducibility
                degree_val = df_degrees[df_degrees['node_id'] == node_id_val]['degree'].iloc[0]
                # Generate visit counts to test different probability magnitudes
                if node_id_val % 23 == 0: # Very low probability
                     lambda_val = 0.0005 * (100 / (max(1,degree_val) + 1)) # avg visits over 1k queries
                elif node_id_val % 11 == 0: # Low probability
                     lambda_val = 0.05 * (100 / (max(1,degree_val) + 1))
                elif node_id_val % 5 == 0: # Moderate probability
                     lambda_val = 2.5 * (100 / (max(1,degree_val) + 1))
                else: # Higher probability (but still < 1000 visits for a single node usually)
                     lambda_val = 50 * (100 / (max(1,degree_val) + 1))
                visit_counts[node_id_val] = np.random.poisson(lambda_val)
                if visit_counts[node_id_val] < 0: visit_counts[node_id_val] = 0

    if not visit_counts:
        print(f"Warning: No visit data parsed/generated. Plot may show zero visits.")
        df_visits = pd.DataFrame(columns=['node_id', 'visit_count'])
    else:
        df_visits_list = [{'node_id': nid, 'visit_count': count} for nid, count in visit_counts.items()]
        df_visits = pd.DataFrame(df_visits_list)
    print(f"Parsed/generated {len(df_visits)} unique visited nodes, totaling {sum(visit_counts.values())} visits (simulated over {NUMBER_OF_QUERIES} queries).")

    # --- 3. Merge Degree Data with Visit Data ---
    df_merged = pd.merge(df_degrees, df_visits, on='node_id', how='left')
    df_merged['visit_count'] = df_merged['visit_count'].fillna(0).astype(float) # visit_count is total over NUMBER_OF_QUERIES
    print(f"Merged data contains {len(df_merged)} entries.")

    # --- 5. Binning Degrees and Calculating Average Visit Count per Node in Bin (over NUMBER_OF_QUERIES) ---
    current_bin_size = 5
    bin_size_for_plot = current_bin_size

    if not df_degrees.empty:
        print(f"\nBinning degrees into groups of {current_bin_size} for average visit count calculation...")

        df_merged_with_bins = df_merged.copy()
        df_merged_with_bins['degree_bin_start'] = (df_merged_with_bins['degree'] // current_bin_size) * current_bin_size
        
        df_binned_analysis = df_merged_with_bins.groupby('degree_bin_start').agg(
            total_visit_count_in_bin=('visit_count', 'sum'),
            node_count_in_bin=('node_id', 'nunique')
        ).reset_index()

        # This is the average number of times a node in this bin was visited over NUMBER_OF_QUERIES queries.
        # This value is what gets cached.
        df_binned_analysis['average_visit_count_per_node_in_bin'] = 0.0
        df_binned_analysis.loc[df_binned_analysis['node_count_in_bin'] > 0, 'average_visit_count_per_node_in_bin'] = \
            df_binned_analysis['total_visit_count_in_bin'] / df_binned_analysis['node_count_in_bin']
        
        df_binned_analysis['degree_bin_label'] = df_binned_analysis['degree_bin_start'].astype(str) + '-' + \
                                                 (df_binned_analysis['degree_bin_start'] + current_bin_size - 1).astype(str)
        
        bin_to_drop_label = '60-64'
        original_length = len(df_binned_analysis)
        df_plot_data_intermediate = df_binned_analysis[df_binned_analysis['degree_bin_label'] != bin_to_drop_label].copy()
        if len(df_plot_data_intermediate) < original_length:
            print(f"\nManually dropped the bin: '{bin_to_drop_label}'")
        else:
            print(f"\nNote: Bin '{bin_to_drop_label}' not found for dropping or already removed.")
        
        df_plot_data = df_plot_data_intermediate
        
        print(f"\nBinned data (average visit count per node in bin over {NUMBER_OF_QUERIES} queries) for plotting prepared:")
        print(df_plot_data[['degree_bin_label', 'average_visit_count_per_node_in_bin']].head())

        if df_plot_data is not None and not df_plot_data.empty:
            try:
                with open(CACHE_FILE_PATH, 'wb') as f:
                    pickle.dump({'data': df_plot_data, 'bin_size': bin_size_for_plot}, f)
                print(f"Saved computed binned data to cache: {CACHE_FILE_PATH}")
            except Exception as e:
                print(f"Error saving data to cache: {e}")
        elif df_plot_data is None or df_plot_data.empty:
             print("Computed data for binned plot is empty, not saving to cache.")
    else:
        print("Degree data (df_degrees) is empty. Cannot perform binning.")
        df_plot_data = pd.DataFrame()
        bin_size_for_plot = current_bin_size

# %%
# --- 6. Plotting (Binned Bar Chart - Academic Style) ---

if df_plot_data is not None and not df_plot_data.empty and 'average_visit_count_per_node_in_bin' in df_plot_data.columns:
    base_name, ext = os.path.splitext(output_image_file)
    # --- OUTPUT PDF FILE NAME: Keep this consistent ---
    binned_output_image_file = base_name + ext

    fig, ax = plt.subplots(figsize=(6, 2.5)) # Adjusted figure size

    df_plot_data_plotting = df_plot_data.copy()
    # Calculate per-query probability: (avg visits over N queries) / N
    df_plot_data_plotting['per_query_visit_probability'] = \
        df_plot_data_plotting['average_visit_count_per_node_in_bin'] / NUMBER_OF_QUERIES
    
    max_probability = df_plot_data_plotting['per_query_visit_probability'].max()
    
    y_axis_values_to_plot = df_plot_data_plotting['per_query_visit_probability']
    y_axis_label = r"Per-Query Node Visit Probability in Bin" # Base label

    apply_scaling_to_label_and_values = False # Initialize flag
    exponent_for_label_display = 0 # Initialize exponent

    if pd.notna(max_probability) and max_probability > 0:
        potential_exponent = math.floor(math.log10(max_probability))
        
        if potential_exponent <= -4 or potential_exponent >= 0: 
            apply_scaling_to_label_and_values = True
            exponent_for_label_display = potential_exponent
            # No specific adjustment for potential_exponent >=0 here, it's handled by the general logic.

        if apply_scaling_to_label_and_values:
            y_axis_label = rf"Visit Probability ($\times 10^{{{exponent_for_label_display}}}$)"
            y_axis_values_to_plot = df_plot_data_plotting['per_query_visit_probability'] / (10**exponent_for_label_display)
            print(f"Plotting with Max per-query probability: {max_probability:.2e}, Exponent for label: {exponent_for_label_display}. Y-axis values scaled for plot.")
        else:
            print(f"Plotting with Max per-query probability: {max_probability:.2e}. Plotting direct probabilities without label scaling (exponent {potential_exponent} is within no-scale range [-3, -1]).")

    elif pd.notna(max_probability) and max_probability == 0:
        print("Max per-query probability is 0. Plotting direct probabilities (all zeros).")
    else:
        print(f"Max per-query probability is NaN or invalid ({max_probability}). Plotting direct probabilities without scaling if possible.")
    
    ax.bar(
        df_plot_data_plotting['degree_bin_label'],
        y_axis_values_to_plot,
        color='white',
        edgecolor=edgecolors_ref[0],
        linewidth=1.5,
        width=0.8
    )
    
    ax.set_xlabel('Node Degree', fontsize=10.5, labelpad=6)
    # MODIFIED LINE: Added labelpad to move the y-axis label to the left
    ax.set_ylabel(y_axis_label, fontsize=10.5, labelpad=10) 

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0f}%"))

    num_bins = len(df_plot_data_plotting)
    if num_bins > 12: 
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    elif num_bins > 8:
         ax.tick_params(axis='x', labelsize=9)
    else:
        ax.tick_params(axis='x', labelsize=10)
    
    ax.tick_params(axis='y', labelsize=10)

    padding_factor = 0.05
    current_max_y_on_axis = y_axis_values_to_plot.max()
    
    upper_y_limit = 0.1 # Default small upper limit
    if pd.notna(current_max_y_on_axis):
        if current_max_y_on_axis > 0:
             # Adjust minimum visible range based on whether scaling was applied and the exponent
            min_meaningful_limit = 0.01
            if apply_scaling_to_label_and_values and exponent_for_label_display >= 0 : # Numbers on axis are smaller due to positive exponent scaling
                 min_meaningful_limit = 0.1 # If original numbers were e.g. 2500 (2.5 x 10^3), scaled axis is 2.5, 0.1 is fine
            elif not apply_scaling_to_label_and_values and pd.notna(max_probability) and max_probability >=1: # Direct large probabilities
                 min_meaningful_limit = 1 # If max prob is 2.5 (250%), axis value 2.5, needs larger base limit
            
            upper_y_limit = max(min_meaningful_limit, current_max_y_on_axis * (1 + padding_factor))

        else: # current_max_y_on_axis is 0
            upper_y_limit = 0.1 
        ax.set_ylim(0, upper_y_limit)
    else:
        ax.set_ylim(0, 1.0) # Default for empty or NaN data

    plt.tight_layout()
    plt.savefig(binned_output_image_file, bbox_inches="tight", dpi=300)
    print(f"Binned bar chart saved to {binned_output_image_file}")
    plt.show()
    plt.close(fig)
else:
    if df_plot_data is None:
        print("Data for plotting (df_plot_data) is None. Skipping plot generation.")
    elif df_plot_data.empty:
        print("Data for plotting (df_plot_data) is empty. Skipping plot generation.")
    elif 'average_visit_count_per_node_in_bin' not in df_plot_data.columns: 
        print("Essential column 'average_visit_count_per_node_in_bin' is missing in df_plot_data. Skipping plot generation.")

# %%
print("Script finished.")