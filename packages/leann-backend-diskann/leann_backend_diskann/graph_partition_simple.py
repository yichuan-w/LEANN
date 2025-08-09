#!/usr/bin/env python3
"""
Simplified Graph Partition Module for LEANN DiskANN Backend

This module provides a simple Python interface for graph partitioning
that directly calls the existing executables.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def partition_graph_simple(
    index_prefix_path: str, output_dir: Optional[str] = None, **kwargs
) -> tuple[str, str]:
    """
    Simple function to partition a graph index.

    Args:
        index_prefix_path: Path to the index prefix (e.g., "/path/to/index")
        output_dir: Output directory (defaults to parent of index_prefix_path)
        **kwargs: Additional parameters for graph partitioning

    Returns:
        Tuple of (disk_graph_index_path, partition_bin_path)
    """
    # Set default parameters
    params = {
        "gp_times": 10,
        "lock_nums": 10,
        "cut": 100,
        "scale_factor": 1,
        "data_type": "float",
        "thread_nums": 10,
        **kwargs,
    }

    # Determine output directory
    if output_dir is None:
        output_dir = str(Path(index_prefix_path).parent)

    # Find the graph_partition directory
    current_file = Path(__file__)
    graph_partition_dir = current_file.parent.parent / "third_party" / "DiskANN" / "graph_partition"

    if not graph_partition_dir.exists():
        raise RuntimeError(f"Graph partition directory not found: {graph_partition_dir}")

    # Find input index file
    old_index_file = f"{index_prefix_path}_disk_beam_search.index"
    if not os.path.exists(old_index_file):
        old_index_file = f"{index_prefix_path}_disk.index"

    if not os.path.exists(old_index_file):
        raise RuntimeError(f"Index file not found: {old_index_file}")

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_data_dir = Path(temp_dir) / "data"
        temp_data_dir.mkdir(parents=True, exist_ok=True)

        # Set up paths for temporary files
        graph_path = temp_data_dir / "starling" / "_M_R_L_B" / "GRAPH"
        graph_gp_path = (
            graph_path
            / f"GP_TIMES_{params['gp_times']}_LOCK_{params['lock_nums']}_GP_USE_FREQ0_CUT{params['cut']}_SCALE{params['scale_factor']}"
        )
        graph_gp_path.mkdir(parents=True, exist_ok=True)

        # Run the build script with our parameters
        cmd = [str(graph_partition_dir / "build.sh"), "release", "split_graph", index_prefix_path]

        # Set environment variables for parameters
        env = os.environ.copy()
        env.update(
            {
                "GP_TIMES": str(params["gp_times"]),
                "GP_LOCK_NUMS": str(params["lock_nums"]),
                "GP_CUT": str(params["cut"]),
                "GP_SCALE_F": str(params["scale_factor"]),
                "DATA_TYPE": params["data_type"],
                "GP_T": str(params["thread_nums"]),
            }
        )

        print(f"Running graph partition with command: {' '.join(cmd)}")
        print(f"Working directory: {graph_partition_dir}")

        # Run the command
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, cwd=graph_partition_dir
        )

        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(
                f"Graph partitioning failed with return code {result.returncode}.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Check if output files were created
        disk_graph_path = Path(output_dir) / "_disk_graph.index"
        partition_bin_path = Path(output_dir) / "_partition.bin"

        if not disk_graph_path.exists():
            raise RuntimeError(f"Expected output file not found: {disk_graph_path}")

        if not partition_bin_path.exists():
            raise RuntimeError(f"Expected output file not found: {partition_bin_path}")

        print("✅ Partitioning completed successfully!")
        print(f"   Disk graph index: {disk_graph_path}")
        print(f"   Partition binary: {partition_bin_path}")

        return str(disk_graph_path), str(partition_bin_path)


# Example usage
if __name__ == "__main__":
    try:
        disk_graph_path, partition_bin_path = partition_graph_simple(
            "/Users/yichuan/Desktop/release2/leann/diskannbuild/test_doc_files",
            gp_times=5,
            lock_nums=5,
            cut=50,
        )
        print("Success! Output files:")
        print(f"  - {disk_graph_path}")
        print(f"  - {partition_bin_path}")
    except Exception as e:
        print(f"Error: {e}")
