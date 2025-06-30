import subprocess
import os
import re
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np # Added for sorting unique nprobes

# --- Configuration ---
PYTHON_EXE = "python"  # Or specify the full path if needed
SCRIPT_TO_RUN = "run_server.py"
LOG_DIR = "exp_logs"  # Directory where run_server.py saves logs/summaries
RESULTS_DIR = "experiment_results" # Directory to save plot and data
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Argument Parsing for Flexibility ---
parser = argparse.ArgumentParser(description="Run recall experiments or plot results from existing CSV.")
parser.add_argument('--nprobes', type=str, default="2,4,8,16,32,64", help='Comma-separated list of nprobe values (used when running experiments)')
parser.add_argument('--degrees', type=str, default="None,30,240", help='Comma-separated list of degree values (use None for default, used when running experiments)')
parser.add_argument('--task', type=str, default="nq", help='Task argument for run_server.py (used when running experiments, or inferred from --input-csv)')
parser.add_argument('--input-csv', type=str, default=None, help='Path to an existing CSV file to plot directly, skipping experiments.') # New argument
args = parser.parse_args()

# --- Initialize Variables ---
results_df = None
task_name = args.task # Default task name
NPROBE_VALUES = []
DEGREE_VALUES = []

# --- Mode Selection: Run Experiments or Plot from CSV ---

if args.input_csv:
    # --- Plot from CSV Mode ---
    print(f"--- Plotting from existing CSV: {args.input_csv} ---")
    try:
        results_df = pd.read_csv(args.input_csv)
        print(f"Loaded data with {len(results_df)} rows.")

        # ---- NEW: Replace 'default' string with 60 if present ----
        if 'degree' in results_df.columns and results_df['degree'].dtype == 'object': # Check if column exists and might contain strings
            results_df['degree'] = results_df['degree'].replace('default', 60)
            # Attempt to convert the column to numeric after replacement
            results_df['degree'] = pd.to_numeric(results_df['degree'], errors='coerce')
            print("Replaced 'default' degree values with 60 and converted column to numeric.")
        # ---- END NEW ----

        # Infer task name from filename if possible
        match = re.search(r'results_([^_]+)_[\d_]+\.csv', os.path.basename(args.input_csv))
        if match:
            task_name = match.group(1)
            print(f"Inferred task name: {task_name}")
        else:
            print(f"Could not infer task name from filename, using default: {task_name}")

        # Get NPROBE_VALUES from loaded data for plotting ticks
        if 'nprobe' in results_df.columns:
            NPROBE_VALUES = sorted(results_df['nprobe'].unique())
            print(f"Nprobe values from data: {NPROBE_VALUES}")
        else:
            print("Warning: 'nprobe' column not found in CSV. Plot ticks might be incorrect.")


    except FileNotFoundError:
        print(f"Error: Input CSV file not found: {args.input_csv}")
        exit(1)
    except Exception as e:
        print(f"Error reading CSV file {args.input_csv}: {e}")
        exit(1)

else:
    # --- Run Experiments Mode ---
    print("--- Running New Experiments ---")
    # Parse nprobe values
    try:
        NPROBE_VALUES = [int(p.strip()) for p in args.nprobes.split(',')]
    except ValueError:
        print("Error: Invalid nprobe values. Please provide comma-separated integers.")
        exit(1)

    # Parse degree values
    DEGREE_VALUES = []
    for d_str in args.degrees.split(','):
        d_str = d_str.strip()
        if d_str.lower() == 'none':
            DEGREE_VALUES.append(None)
        else:
            try:
                DEGREE_VALUES.append(int(d_str))
            except ValueError:
                print(f"Error: Invalid degree value '{d_str}'. Use 'None' or integers.")
                exit(1)

    print(f"Nprobe values to test: {NPROBE_VALUES}")
    print(f"Degree values to test: {DEGREE_VALUES}")
    print(f"Task: {task_name}") # Use task_name


    # --- Helper Functions (Only needed for experiment mode) ---
    def parse_recall_from_summary(summary_file_path):
        """Parses the recall rate from the summary file."""
        try:
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Regex to find the recall rate line
                match = re.search(r"Average Recall Rate:\s*([\d.]+)", content)
                if match:
                    return float(match.group(1))
                else:
                    print(f"Warning: Could not find recall rate in {summary_file_path}")
                    return None
        except FileNotFoundError:
            print(f"Error: Summary file not found at {summary_file_path}")
            return None
        except Exception as e:
            print(f"Error reading or parsing summary file {summary_file_path}: {e}")
            return None

    def find_summary_file(output_text):
        """Finds the summary file path from the script's output."""
        # Regex to find the summary file path line
        match = re.search(r"Summary written to:\s*(.*\.txt)", output_text)
        if match:
            return match.group(1).strip()
        else:
            # Fallback: Search for any summary file pattern in the log directory if not found in stdout
            print("Warning: Could not find summary file path in script output. Searching log directory...")
            try:
                # Look for the most recent summary file
                log_files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.startswith("summary_") and f.endswith(".txt")]
                if log_files:
                    latest_summary = max(log_files, key=os.path.getmtime)
                    print(f"Found potential summary file by search: {latest_summary}")
                    return latest_summary
            except FileNotFoundError:
                print(f"Warning: Log directory '{LOG_DIR}' not found during fallback search.")
            except Exception as e:
                print(f"Error during fallback summary file search: {e}")
            print("Fallback search failed.")
            return None


    # --- Main Experiment Loop ---
    results = []
    start_experiment_time = time.time()

    for degree in DEGREE_VALUES:
        for nprobe in NPROBE_VALUES:
            run_start_time = time.time()
            degree_str = str(degree) if degree is not None else "Default"
            print(f"\n--- Running Experiment: degree={degree_str}, nprobe={nprobe}, task={task_name} ---") # Use task_name

            # Base command
            cmd = [
                PYTHON_EXE, "-u", SCRIPT_TO_RUN,
                "--nprobe", str(nprobe),
                "--task", task_name # Use task_name
            ]
            # Add degree if specified
            if degree is not None:
                cmd.extend(["--degree", str(degree)])

            print(f"Executing command: {' '.join(cmd)}")

            recall_rate = None
            summary_file = None
            process_returncode = -1 # Default to error

            try:
                # Run the script and capture output
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False, # Check returncode manually
                    encoding='utf-8',
                    errors='replace'
                )
                process_returncode = process.returncode

                print(f"Command finished with return code: {process.returncode}")
                # Uncomment below to see full output for debugging
                # print("--- stdout ---")
                # print(process.stdout)
                # print("--- stderr ---")
                # print(process.stderr)
                # print("--------------")

                # Attempt to find summary file path from stdout
                summary_file = find_summary_file(process.stdout)

                if summary_file:
                    # Give the filesystem a moment before reading
                    time.sleep(1)
                    recall_rate = parse_recall_from_summary(summary_file)
                else:
                     print("ERROR: Could not locate summary file for this run.")

                if process.returncode != 0:
                     print(f"Warning: Script execution failed (return code {process.returncode}) for degree={degree_str}, nprobe={nprobe}.")
                     # Recall might still be None or potentially parsed if summary existed

            except FileNotFoundError:
                print(f"CRITICAL ERROR: Could not find script '{SCRIPT_TO_RUN}' or Python executable '{PYTHON_EXE}'.")
                # No result to append for this specific run in this case
                continue # Skip to next iteration
            except Exception as e:
                print(f"CRITICAL ERROR running experiment for degree={degree_str}, nprobe={nprobe}: {e}")
                # Append error result
                results.append({
                    "degree": 60 if degree is None else degree, # <-- MODIFIED: Use 60 for None
                    "nprobe": nprobe,
                    "recall": None,
                    "duration_s": time.time() - run_start_time,
                    "return_code": process_returncode, # Use captured or default error code
                    "summary_file": summary_file,
                    "error": str(e)
                })
                continue # Skip to next iteration

            run_duration = time.time() - run_start_time
            print(f"Result: degree={degree_str}, nprobe={nprobe}, recall={recall_rate}, duration={run_duration:.2f}s")
            results.append({
                "degree": 60 if degree is None else degree, # <-- MODIFIED: Use 60 for None
                "nprobe": nprobe,
                "recall": recall_rate,
                "duration_s": run_duration,
                "return_code": process_returncode,
                "summary_file": summary_file,
                "error": None if process_returncode == 0 and recall_rate is not None else "Run failed or recall not found"
            })

            # Optional: add a small delay between runs if needed
            # time.sleep(5)

    # --- Post-Experiment Processing ---
    print("\n--- Experiment Complete. Processing Results ---")
    total_duration = time.time() - start_experiment_time
    print(f"Total experiment duration: {total_duration:.2f}s")

    if not results:
        print("No results collected. Exiting.")
        exit()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv_path = os.path.join(RESULTS_DIR, f"experiment_results_{task_name}_{timestamp}.csv") # Use task_name
    try:
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to: {results_csv_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


# --- Data Processing and Plotting (Common to both modes) ---

if results_df is None or results_df.empty:
     print("No data available to plot. Exiting.")
     exit()

print("\n--- Generating Plot ---")

# Filter out runs where recall could not be parsed or is missing
plot_df = results_df.dropna(subset=['recall'])

# Filter out rows where degree is 'default' as we need numeric degree for calculation
# Also ensure nprobe is numeric
plot_df_numeric = plot_df[pd.to_numeric(plot_df['degree'], errors='coerce').notna()].copy()
plot_df_numeric['degree'] = pd.to_numeric(plot_df_numeric['degree'])
plot_df_numeric['nprobe'] = pd.to_numeric(plot_df_numeric['nprobe']) # Ensure nprobe is numeric

if plot_df_numeric.empty:
    print("No successful runs with numeric degree and recall values found. Cannot generate plot.")
else:
    # Calculate the new x-axis value
    plot_df_numeric['degree_times_nprobe'] = plot_df_numeric['degree'] * plot_df_numeric['nprobe']

    # Convert 'degree' column back to string for legend grouping
    plot_df_numeric['degree_label'] = plot_df_numeric['degree'].astype(int).astype(str)

    # Plotting
    plt.figure(figsize=(12, 7))

    # Group by degree and plot recall vs degree * nprobe
    # Sort group by the new x-axis value for correct line plotting
    for degree_label, group in plot_df_numeric.groupby('degree_label'):
        group = group.sort_values('degree_times_nprobe')
        plt.plot(group['degree_times_nprobe'], group['recall'], marker='o', linestyle='-', label=f'Degree={degree_label}')

    plt.xlabel("Degree * Nprobe") # Updated X-axis label
    plt.ylabel("Average Recall Rate")
    plt.title(f"Recall Rate vs. Degree * Nprobe (Task: {task_name})") # Updated title
    plt.legend(title="Graph Degree")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    # plt.xscale('log', base=2) # Removed log scale, let matplotlib decide or adjust later if needed
    # plt.xticks(...) # Removed custom ticks, let matplotlib decide

    # Save plot
    plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Use a new timestamp for the plot
    # Updated filename
    plot_path = os.path.join(RESULTS_DIR, f"recall_vs_degree_nprobe_{task_name}_{plot_timestamp}.png")
    try:
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Uncomment to display the plot interactively

print("\nDone.") 