#!/usr/bin/env python3
import subprocess
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import os
import time
import sys
import argparse
import concurrent.futures
import signal
import psutil

parser = argparse.ArgumentParser()
parser.add_argument("--path-suffix", type=str, default="", help="Path suffix for the index")
parser.add_argument("--pq-compressed", type=int, default=None)
parser.add_argument("--beam-width", type=int, default=2, help="DiskANN beam width for search (controls number of IO requests per iteration)")
parser.add_argument("--index-type", type=str, default="diskann", help="Index type to test (default: diskann)")
parser.add_argument("--task", type=str, default="nq", help="Task to run (default: nq)")
parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of concurrent processes")
parser.add_argument("--timeout", type=int, default=1800, help="Timeout for each process in seconds")
parser.add_argument("--retry-count", type=int, default=2, help="Number of retries for failed runs")
parser.add_argument(
    "--target-recalls",
    type=float,
    nargs='+',
    default=[0.85, 0.90, 0.95],
    help="Target recalls to achieve (e.g., --target-recalls 0.85 0.90 0.95)"
)
args = parser.parse_args()
path_suffix = args.path_suffix

pq_compressed = args.pq_compressed
beam_width = args.beam_width
max_workers = args.max_workers
timeout = args.timeout
retry_count = args.retry_count

TARGET_RECALLS = args.target_recalls

task = args.task

# Process management
running_processes = {}  # PID -> Process object

# Based on previous data, search around these values
if args.index_type == "diskann":
    if task == "nq":
        if pq_compressed is None:
            NPROBE_RANGES = {
                0.85: range(10, 50),
                0.90: range(62, 67),    # Narrow range around 64 (63, 64, 65, 66)
                0.95: range(190, 195)   # Narrow range around 192 (190, 191, 192, 193, 194)
            }
        elif pq_compressed == 10:
            NPROBE_RANGES = {
                0.85: range(10, 70),
                0.90: range(90, 127),    # Narrow range around 64 (63, 64, 65, 66)
                0.95: range(200, 384)   # Narrow range around 192 (190, 191, 192, 193, 194)
            }
        elif pq_compressed == 20:
            NPROBE_RANGES = {
                0.85: range(10, 50),
                0.90: range(64, 128),    # Narrow range around 64 (63, 64, 65, 66)
                0.95: range(188, 192)   # Narrow range around 192 (190, 191, 192, 193, 194)
            }
        elif pq_compressed == 5:
            NPROBE_RANGES = {
                0.85: range(10, 500),
                0.90: range(768, 2000),    # Narrow range around 64 (63, 64, 65, 66)
                0.95: range(3000, 4096)   # Narrow range around 192 (190, 191, 192, 193, 194)
            }
    elif task == "trivia":
        if pq_compressed is None:
            NPROBE_RANGES = {
                0.85: range(90, 150),
                0.90: range(150, 200),    # Narrow range around 64 (63, 64, 65, 66)
                0.95: range(200, 300)   # Narrow range around 192 (190, 191, 192, 193, 194)
            }
    elif task == "gpqa":
        if pq_compressed is None:
            NPROBE_RANGES = {
                0.85: range(1, 30),
                0.90: range(1, 30),    # Narrow range around 64 (63, 64, 65, 66)
                0.95: range(1, 30)   # Narrow range around 192 (190, 191, 192, 193, 194)
            }
    elif task == "hotpot":
        if pq_compressed is None:
            NPROBE_RANGES = {
                0.85: range(19, 160),
                0.90: range(120, 210),    # Narrow range around 64 (63, 64, 65, 66)
                0.95: range(1000, 1200)   # Narrow range around 192 (190, 191, 192, 193, 194)
            }
elif args.index_type == "ivf_disk":
    if task == "nq":
        assert pq_compressed is None
        NPROBE_RANGES = {
            0.85: range(13, 16),
            0.90: range(30,40),
            0.95: range(191, 194)
        }
    elif task == "trivia":
        assert pq_compressed is None
        NPROBE_RANGES = {
            0.85: range(13, 50),
            0.90: range(30, 100),
            0.95: range(100, 400)
        }
    elif task == "gpqa":
        assert pq_compressed is None
        NPROBE_RANGES = {
            0.85: range(1, 30),
            0.90: range(1, 30),    # Narrow range around 64 (63, 64, 65, 66)
            0.95: range(1, 30)   # Narrow range around 192 (190, 191, 192, 193, 194)
        }
    elif task == "hotpot":
        NPROBE_RANGES = {
            0.85: range(13, 100),
            0.90: range(30, 200),
            0.95: range(191, 700)
        }
elif args.index_type == "hnsw":
    if task == "nq":
        NPROBE_RANGES = {
            0.85: range(130, 140),
            0.90: range(550, 666),
            0.95: range(499, 1199),
        }
    if task == "gpqa":
        NPROBE_RANGES = {
            0.85: range(40, 70),
            0.90: range(60, 100),
            0.95: range(200, 500),
        }
    elif task == "hotpot":
        NPROBE_RANGES = {
            0.85: range(450, 480),
            0.90: range(1000, 1300),
            0.95: range(2000, 4000),
        }
    elif task == "trivia":
        NPROBE_RANGES = {
            0.85: range(100, 400),
            0.90: range(700, 1800),
            0.95: range(506, 1432)
        }

# Create a directory for logs if it doesn't exist
os.makedirs("nprobe_logs", exist_ok=True)

# Set up signal handling for clean termination
def signal_handler(sig, frame):
    print("Received termination signal. Cleaning up running processes...")
    for pid, process in running_processes.items():
        try:
            if process.poll() is None:  # Process is still running
                print(f"Terminating process {pid}...")
                process.terminate()
                time.sleep(0.5)
                if process.poll() is None:  # If still running after terminate
                    print(f"Killing process {pid}...")
                    process.kill()
                    
                # Kill any child processes
                try:
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        print(f"Killing child process {child.pid}...")
                        child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except:
            pass
    
    print("All processes terminated. Exiting.")
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_batch_demo(nprobe: int, retry: int = 0) -> Optional[float]:
    """Run main.py in batch mode with a specific nprobe value and extract the recall."""
    command = f"python -u ./demo/main.py --search-only --load-indices {args.index_type} --domain rpj_wiki --lazy-load-passages --nprobe {nprobe} --task {task} --skip-passages"
    if pq_compressed is not None:
        command += f" --diskann-search-memory-maximum {pq_compressed}"
    if beam_width is not None:
        command += f" --diskann-beam-width {beam_width}"
    if args.index_type == "hnsw":
        command += f" --hnsw-old"
    # command += " --embedder intfloat/multilingual-e5-small"

    cmd = [
        "fish", "-c",
        # f"set -gx LD_PRELOAD \"/lib/x86_64-linux-gnu/libmkl_core.so /lib/x86_64-linux-gnu/libmkl_intel_lp64.so /lib/x86_64-linux-gnu/libmkl_intel_thread.so /lib/x86_64-linux-gnu/libiomp5.so\" && "
        "source ./.venv/bin/activate.fish &&"
        + command
    ]
    
    print(f"Running with nprobe={nprobe}, beam_width={beam_width}, retry={retry}/{retry_count}")
    log_file = f"nprobe_logs/nprobe_{nprobe}_beam{beam_width}_{path_suffix}_retry{retry}.log"
    
    try:
        # Also save the command to the log file
        with open(log_file, "w") as f:
            f.write(f"Command: {cmd[1]}\n\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=== OUTPUT BEGINS ===\n")
        
        # Run the command and tee the output to both stdout and the log file
        with open(log_file, "a") as f:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Register the process for cleanup
            pid = process.pid
            running_processes[pid] = process
            
            # Process output line by line for real-time logging
            if process.stdout:  # Check if stdout is not None
                # Set a timeout
                start_time = time.time()
                current_output = ""
                
                while process.poll() is None:
                    # Check for timeout
                    if time.time() - start_time > timeout:
                        print(f"Process timeout for nprobe={nprobe}, killing...")
                        f.write("\n\nProcess timed out, killing...\n")
                        process.terminate()
                        time.sleep(0.5)
                        if process.poll() is None:
                            process.kill()
                            
                        # Clean up child processes
                        try:
                            parent = psutil.Process(pid)
                            children = parent.children(recursive=True)
                            for child in children:
                                child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
                        if pid in running_processes:
                            del running_processes[pid]
                            
                        # Retry if we have attempts left
                        if retry < retry_count:
                            print(f"Retrying nprobe={nprobe}...")
                            return run_batch_demo(nprobe, retry + 1)
                        return None
                
                    # Read output with a small timeout to allow for process checking
                    try:
                        line = process.stdout.readline()
                        if not line:
                            time.sleep(0.1)  # Small pause to avoid busy waiting
                            continue
                            
                        print(line, end='')  # Print to console
                        f.write(line)  # Write to log file
                        f.flush()  # Make sure it's written immediately
                    except:
                        time.sleep(0.1)
            
            exit_code = process.wait()
            
            # Process complete, remove from running list
            if pid in running_processes:
                del running_processes[pid]
                
            f.write(f"\nExit code: {exit_code}\n")
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        # Re-read the log file to extract recall rate
        with open(log_file, "r") as f:
            log_content = f.read()
            
        # Try multiple patterns to find recall rate
        recall = None
        patterns = [
            fr"Avg recall rate for {args.index_type}: ([0-9.]+)",
            r"recall: ([0-9.]+)",
            fr"{args.index_type}.*?recall.*?([0-9.]+)",
            fr"recall.*?{args.index_type}.*?([0-9.]+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            if matches:
                try:
                    recall = float(matches[-1])  # Take the last one if multiple matches
                    print(f"Found recall rate using pattern: {pattern}")
                    break
                except ValueError:
                    continue
                    
        if recall is None:
            print(f"Warning: Could not extract recall rate from output log {log_file}")
            # Try to find any number that looks like a recall rate (between 0 and 1)
            possible_recalls = re.findall(r"recall.*?([0-9]+\.[0-9]+)", log_content, re.IGNORECASE)
            if possible_recalls:
                try:
                    recall_candidates = [float(r) for r in possible_recalls if 0 <= float(r) <= 1]
                    if recall_candidates:
                        recall = recall_candidates[-1]  # Take the last one
                        print(f"Guessed recall rate: {recall} (based on pattern matching)")
                except ValueError:
                    pass
                    
        if recall is None:
            # Log this failure with more context
            with open("nprobe_logs/failed_recalls.log", "a") as f:
                f.write(f"Failed to extract recall for nprobe={nprobe} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Retry if we have attempts left
            if retry < retry_count:
                print(f"Retrying nprobe={nprobe} due to failed recall extraction...")
                return run_batch_demo(nprobe, retry + 1)
                
            return None
        
        print(f"nprobe={nprobe}, recall={recall:.4f}")
        return recall
        
    except subprocess.TimeoutExpired:
        print(f"Command timed out for nprobe={nprobe}")
        with open(log_file, "a") as f:
            f.write("\n\nCommand timed out after 1800 seconds\n")
        
        # Retry if we have attempts left
        if retry < retry_count:
            print(f"Retrying nprobe={nprobe}...")
            return run_batch_demo(nprobe, retry + 1)
            
        return None
        
    except Exception as e:
        print(f"Error running command for nprobe={nprobe}: {e}")
        with open(log_file, "a") as f:
            f.write(f"\n\nError: {e}\n")
            
        # Retry if we have attempts left
        if retry < retry_count:
            print(f"Retrying nprobe={nprobe} due to error: {e}...")
            return run_batch_demo(nprobe, retry + 1)
            
        return None

def batch_run_nprobe_values(nprobe_values):
    """Run multiple nprobe values in parallel with a thread pool."""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_nprobe = {executor.submit(run_batch_demo, nprobe): nprobe for nprobe in nprobe_values}
        
        for future in concurrent.futures.as_completed(future_to_nprobe):
            nprobe = future_to_nprobe[future]
            try:
                recall = future.result()
                if recall is not None:
                    results[nprobe] = recall
                    print(f"Completed nprobe={nprobe} with recall={recall:.4f}")
            except Exception as e:
                print(f"Error processing nprobe={nprobe}: {e}")
                
    return results

def adaptive_search_nprobe(target_recall: float, min_nprobe: int, max_nprobe: int, tolerance: float = 0.001) -> Dict:
    """
    Use an adaptive search strategy to find the optimal nprobe value for a target recall.
    Combines binary search with exploration to handle non-linear relationships.
    
    Args:
        target_recall: The target recall to achieve
        min_nprobe: Minimum nprobe value to start search
        max_nprobe: Maximum nprobe value for search
        tolerance: How close we need to get to the target_recall
        
    Returns:
        Dictionary with the best nprobe, achieved recall, and other metadata
    """
    print(f"\nAdaptive searching for nprobe that achieves {target_recall*100:.1f}% recall...")
    print(f"Search range: {min_nprobe} - {max_nprobe}")
    
    with open(f"nprobe_logs/summary_{path_suffix}.log", "a") as f:
        f.write(f"\nAdaptive searching for nprobe that achieves {target_recall*100:.1f}% recall...\n")
        f.write(f"Search range: {min_nprobe} - {max_nprobe}\n")
    
    best_result = {"nprobe": None, "recall": None, "difference": float('inf')}
    all_results = {"nprobe": [], "recall": []}
    
    # Save initial file for this search
    search_results_file = f"nprobe_logs/search_results_{path_suffix}_{target_recall:.2f}.json"
    search_data = {
        "target": target_recall,
        "current_best": best_result,
        "all_results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "search_range": {"min": min_nprobe, "max": max_nprobe}
    }
    
    with open(search_results_file, "w") as f:
        json.dump(search_data, f, indent=2)
    
    # Start with a strategic sampling to understand the recall curve
    # Choose more points if the range is large
    range_size = max_nprobe - min_nprobe
    if range_size > 500:
        num_initial_samples = 5
    elif range_size > 100:
        num_initial_samples = 4
    else:
        num_initial_samples = 3
        
    sample_points = [min_nprobe]
    step = range_size // (num_initial_samples - 1)
    for i in range(1, num_initial_samples - 1):
        sample_points.append(min_nprobe + i * step)
    sample_points.append(max_nprobe)
    
    # Run initial sample points in parallel
    initial_results = batch_run_nprobe_values(sample_points)
    
    # Update all_results and best_result based on initial_results
    for nprobe, recall in initial_results.items():
        all_results["nprobe"].append(nprobe)
        all_results["recall"].append(recall)
        
        diff = abs(recall - target_recall)
        if diff < best_result["difference"]:
            best_result = {"nprobe": nprobe, "recall": recall, "difference": diff}
    
    # Update search results file
    search_data = {
        "target": target_recall,
        "current_best": best_result,
        "all_results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "search_range": {"min": min_nprobe, "max": max_nprobe}
    }
    with open(search_results_file, "w") as f:
        json.dump(search_data, f, indent=2)
    
    # Check if we've already reached target within tolerance
    if best_result["difference"] <= tolerance:
        print(f"Found good enough nprobe value: {best_result['nprobe']} with recall {best_result['recall']:.4f}")
        return best_result
    
    # Analyze initial results to decide on next strategy
    # Sort results by nprobe
    sorted_results = sorted([(n, r) for n, r in zip(all_results["nprobe"], all_results["recall"])])
    nprobes, recalls = zip(*sorted_results)
    
    # Check if the relationship is monotonic
    is_monotonic = all(recalls[i] <= recalls[i+1] for i in range(len(recalls)-1)) or \
                  all(recalls[i] >= recalls[i+1] for i in range(len(recalls)-1))
    
    if is_monotonic:
        print("Relationship appears monotonic, proceeding with binary search.")
        # Find the two closest points that bracket the target
        bracket_low, bracket_high = None, None
        for i in range(len(recalls)-1):
            if (recalls[i] <= target_recall <= recalls[i+1]) or (recalls[i] >= target_recall >= recalls[i+1]):
                bracket_low, bracket_high = nprobes[i], nprobes[i+1]
                break
                
        if bracket_low is None:
            # Target is outside our current range, adjust range
            if all(r < target_recall for r in recalls):
                # All recalls are too low, need to increase nprobe
                bracket_low = nprobes[-1]
                bracket_high = min(max_nprobe, nprobes[-1] * 2)
            else:
                # All recalls are too high, need to decrease nprobe
                bracket_low = max(min_nprobe, nprobes[0] // 2)
                bracket_high = nprobes[0]
                
        # Binary search between bracket_low and bracket_high
        while abs(bracket_high - bracket_low) > 3:
            mid_nprobe = (bracket_low + bracket_high) // 2
            if mid_nprobe in initial_results:
                mid_recall = initial_results[mid_nprobe]
            else:
                mid_recall = run_batch_demo(mid_nprobe)
                if mid_recall is not None:
                    all_results["nprobe"].append(mid_nprobe)
                    all_results["recall"].append(mid_recall)
                    
                    diff = abs(mid_recall - target_recall)
                    if diff < best_result["difference"]:
                        best_result = {"nprobe": mid_nprobe, "recall": mid_recall, "difference": diff}
                    
                    # Update search results file
                    search_data["current_best"] = best_result
                    search_data["all_results"] = all_results
                    with open(search_results_file, "w") as f:
                        json.dump(search_data, f, indent=2)
                
            # Check if we're close enough
            if mid_recall is not None:
                if abs(mid_recall - target_recall) <= tolerance:
                    break
                
                # Adjust brackets
                if mid_recall < target_recall:
                    bracket_low = mid_nprobe
                else:
                    bracket_high = mid_nprobe
            else:
                # If we failed to get a result, try a different point
                bracket_high = mid_nprobe - 1
    else:
        print("Relationship appears non-monotonic, using adaptive sampling.")
        # For non-monotonic relationships, we'll use adaptive sampling
        # First, find the best current point
        best_idx = recalls.index(min(recalls, key=lambda r: abs(r - target_recall)))
        best_nprobe = nprobes[best_idx]
        
        # Try points around the best point with decreasing radius
        radius = max(50, (max_nprobe - min_nprobe) // 10)
        min_radius = 3
        
        while radius >= min_radius:
            # Try points at current radius around best_nprobe
            test_points = []
            lower_bound = max(min_nprobe, best_nprobe - radius)
            upper_bound = min(max_nprobe, best_nprobe + radius)
            
            if lower_bound not in initial_results and lower_bound != best_nprobe:
                test_points.append(lower_bound)
            if upper_bound not in initial_results and upper_bound != best_nprobe:
                test_points.append(upper_bound)
                
            # Add a point in the middle if range is large enough
            if upper_bound - lower_bound > 2*radius/3 and len(test_points) < max_workers:
                mid_point = (lower_bound + upper_bound) // 2
                if mid_point not in initial_results and mid_point != best_nprobe:
                    test_points.append(mid_point)
            
            # Run tests
            if test_points:
                new_results = batch_run_nprobe_values(test_points)
                initial_results.update(new_results)
                
                # Update all_results and best_result
                for nprobe, recall in new_results.items():
                    all_results["nprobe"].append(nprobe)
                    all_results["recall"].append(recall)
                    
                    diff = abs(recall - target_recall)
                    if diff < best_result["difference"]:
                        best_result = {"nprobe": nprobe, "recall": recall, "difference": diff}
                        best_nprobe = nprobe  # Update the center for next iteration
                
                # Update search results file
                search_data["current_best"] = best_result
                search_data["all_results"] = all_results
                with open(search_results_file, "w") as f:
                    json.dump(search_data, f, indent=2)
                    
                # Check if we're close enough
                if best_result["difference"] <= tolerance:
                    break
            
            # Reduce radius for next iteration
            radius = max(min_radius, radius // 2)
    
    # After search, do a final fine-tuning around the best result
    if best_result["nprobe"] is not None:
        fine_tune_range = range(max(min_nprobe, best_result["nprobe"] - 2), 
                               min(max_nprobe, best_result["nprobe"] + 3))
        
        fine_tune_points = [n for n in fine_tune_range if n not in all_results["nprobe"]]
        if fine_tune_points:
            fine_tune_results = batch_run_nprobe_values(fine_tune_points)
            
            for nprobe, recall in fine_tune_results.items():
                all_results["nprobe"].append(nprobe)
                all_results["recall"].append(recall)
                
                diff = abs(recall - target_recall)
                if diff < best_result["difference"]:
                    best_result = {"nprobe": nprobe, "recall": recall, "difference": diff}
            
            # Final update to search results file
            search_data["current_best"] = best_result
            search_data["all_results"] = all_results
            search_data["search_range"] = {"min": min_nprobe, "max": max_nprobe, "phase": "fine_tune"}
            with open(search_results_file, "w") as f:
                json.dump(search_data, f, indent=2)
    
    return best_result

def find_optimal_nprobe_values():
    """Find the optimal nprobe values for target recall rates using adaptive search."""
    # Dictionary to store results for each target recall
    results = {}
    # Dictionary to store all nprobe-recall pairs for plotting
    all_data = {target: {"nprobe": [], "recall": []} for target in TARGET_RECALLS}
    
    # Create a summary file for all runs
    with open(f"nprobe_logs/summary_{path_suffix}.log", "w") as f:
        f.write(f"Find optimal nprobe values - started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Target recalls: {TARGET_RECALLS}\n")
        f.write(f"nprobe ranges: {NPROBE_RANGES}\n\n")
        f.write(f"Max workers: {max_workers}\n")
        f.write(f"Timeout per process: {timeout}s\n")
        f.write(f"Retry count: {retry_count}\n\n")
    
    for target in TARGET_RECALLS:
        # Use the existing NPROBE_RANGES to determine min and max values
        min_nprobe = min(NPROBE_RANGES[target])
        max_nprobe = max(NPROBE_RANGES[target])
        
        print(f"\nUsing NPROBE_RANGES for target {target*100:.1f}%: {min_nprobe} to {max_nprobe}")
        
        # Run adaptive search instead of binary search
        best_result = adaptive_search_nprobe(
            target_recall=target,
            min_nprobe=min_nprobe, 
            max_nprobe=max_nprobe
        )
        
        results[target] = best_result
        
        # Save all tested points to all_data for plotting
        search_results_file = f"nprobe_logs/search_results_{path_suffix}_{target:.2f}.json"
        try:
            with open(search_results_file, "r") as f:
                search_data = json.load(f)
                if "all_results" in search_data:
                    all_data[target]["nprobe"] = search_data["all_results"]["nprobe"]
                    all_data[target]["recall"] = search_data["all_results"]["recall"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load search results for {target}: {e}")
        
        print(f"For target recall {target*100:.1f}%:")
        print(f"  Best nprobe value: {best_result['nprobe']}")
        print(f"  Achieved recall: {best_result['recall']:.4f}")
        print(f"  Difference: {best_result['difference']:.4f}")
        
        with open(f"nprobe_logs/summary_{path_suffix}.log", "a") as f:
            f.write(f"For target recall {target*100:.1f}%:\n")
            f.write(f"  Best nprobe value: {best_result['nprobe']}\n")
            f.write(f"  Achieved recall: {best_result['recall']:.4f}\n")
            f.write(f"  Difference: {best_result['difference']:.4f}\n")
    
    # Plot the results if we have data
    if all_data and any(data["nprobe"] for data in all_data.values()):
        plt.figure(figsize=(10, 6))
        
        # Plot each target's data
        for target in TARGET_RECALLS:
            if not all_data[target]["nprobe"]:
                continue
                
            nprobe_values = all_data[target]["nprobe"]
            recall_values = all_data[target]["recall"]
            
            # Sort data points for better visualization
            sorted_points = sorted(zip(nprobe_values, recall_values))
            sorted_nprobe, sorted_recall = zip(*sorted_points) if sorted_points else ([], [])
            
            plt.plot(sorted_nprobe, sorted_recall, 'o-', 
                     label=f"Target {target*100:.1f}%, Best={results[target]['nprobe']}")
            
            # Mark the optimal point
            opt_nprobe = results[target]["nprobe"]
            opt_recall = results[target]["recall"]
            plt.plot(opt_nprobe, opt_recall, 'r*', markersize=15)
            
            # Add a horizontal line at the target recall
            plt.axhline(y=target, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('nprobe value')
        plt.ylabel('Recall rate')
        plt.title(f'Recall Rate vs nprobe Value (Max Workers: {max_workers})')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'nprobe_logs/nprobe_vs_recall_{path_suffix}.png')
        print(f"Plot saved to nprobe_logs/nprobe_vs_recall_{path_suffix}.png")
    else:
        print("No data to plot.")
        with open(f"nprobe_logs/summary_{path_suffix}.log", "a") as f:
            f.write("No data to plot.\n")
    
    # Save final results
    with open(f"nprobe_logs/optimal_nprobe_values_{path_suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(f"nprobe_logs/summary_{path_suffix}.log", "a") as f:
        f.write(f"\nFind optimal nprobe values - finished at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if results:
            f.write("\nOptimal nprobe values for target recall rates:\n")
            for target, data in results.items():
                f.write(f"{target*100:.1f}% recall: nprobe={data['nprobe']} (actual recall: {data['recall']:.4f})\n")
        else:
            f.write("No optimal nprobe values found.\n")
    
    return results

if __name__ == "__main__":
    try:
        results = find_optimal_nprobe_values()
        
        if not results:
            print("No optimal nprobe values found.")
            sys.exit(1)
            
        print("\nOptimal nprobe values for target recall rates:")
        for target, data in results.items():
            print(f"{target*100:.1f}% recall: nprobe={data['nprobe']} (actual recall: {data['recall']:.4f})")
        
        # Generate the command for running the latency test with the optimal nprobe values
        optimal_values = [data["nprobe"] for target, data in sorted(results.items())]
        test_cmd = f"source ./.venv/bin/activate.fish && cd ~ && python ./Power-RAG/demo/test_serve.py --nprobe_values {' '.join(map(str, optimal_values))}"
        
        print("\nRun this command to test latency with the optimal nprobe values:")
        print(test_cmd)
        
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Cleaning up running processes...")
        signal_handler(signal.SIGINT, None)
        sys.exit(1)
    except Exception as e:
        # Clean up any running processes before re-raising
        import traceback
        traceback.print_exc()
        signal_handler(signal.SIGINT, None)
        raise e