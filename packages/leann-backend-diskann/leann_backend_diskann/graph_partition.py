#!/usr/bin/env python3
"""
Graph Partition Module for LEANN DiskANN Backend

This module provides Python bindings for the graph partition functionality
of DiskANN, allowing users to partition disk-based indices for better
performance.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class GraphPartitioner:
    """
    A Python interface for DiskANN's graph partition functionality.

    This class provides methods to partition disk-based indices for improved
    search performance and memory efficiency.
    """

    def __init__(self, build_type: str = "release"):
        """
        Initialize the GraphPartitioner.

        Args:
            build_type: Build type for the executables ("debug" or "release")
        """
        self.build_type = build_type
        self._ensure_executables()

    def _get_executable_path(self, name: str) -> str:
        """Get the path to a graph partition executable."""
        # Get the directory where this Python module is located
        module_dir = Path(__file__).parent
        # Navigate to the graph_partition directory
        graph_partition_dir = module_dir.parent / "third_party" / "DiskANN" / "graph_partition"
        executable_path = graph_partition_dir / "build" / self.build_type / "graph_partition" / name

        if not executable_path.exists():
            raise FileNotFoundError(f"Executable {name} not found at {executable_path}")

        return str(executable_path)

    def _ensure_executables(self):
        """Ensure that the required executables are built."""
        try:
            self._get_executable_path("partitioner")
            self._get_executable_path("index_relayout")
        except FileNotFoundError:
            # Try to build the executables automatically
            print("Executables not found, attempting to build them...")
            self._build_executables()

    def _build_executables(self):
        """Build the required executables."""
        graph_partition_dir = (
            Path(__file__).parent.parent / "third_party" / "DiskANN" / "graph_partition"
        )
        original_dir = os.getcwd()

        try:
            os.chdir(graph_partition_dir)

            # Clean any existing build
            if (graph_partition_dir / "build").exists():
                shutil.rmtree(graph_partition_dir / "build")

            # Run the build script
            cmd = ["./build.sh", self.build_type, "split_graph", "/tmp/dummy"]
            subprocess.run(cmd, capture_output=True, text=True, cwd=graph_partition_dir)

            # Check if executables were created
            partitioner_path = self._get_executable_path("partitioner")
            relayout_path = self._get_executable_path("index_relayout")

            print(f"✅ Built partitioner: {partitioner_path}")
            print(f"✅ Built index_relayout: {relayout_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to build executables: {e}")
        finally:
            os.chdir(original_dir)

    def partition_graph(
        self,
        index_prefix_path: str,
        output_dir: Optional[str] = None,
        partition_prefix: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Partition a disk-based index for improved performance.

        Args:
            index_prefix_path: Path to the index prefix (e.g., "/path/to/index")
            output_dir: Output directory for results (defaults to parent of index_prefix_path)
            partition_prefix: Prefix for output files (defaults to basename of index_prefix_path)
            **kwargs: Additional parameters for graph partitioning:
                - gp_times: Number of LDG partition iterations (default: 10)
                - lock_nums: Number of lock nodes (default: 10)
                - cut: Cut adjacency list degree (default: 100)
                - scale_factor: Scale factor (default: 1)
                - data_type: Data type (default: "float")
                - thread_nums: Number of threads (default: 10)

        Returns:
            Tuple of (disk_graph_index_path, partition_bin_path)

        Raises:
            RuntimeError: If the partitioning process fails
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

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Determine partition prefix
        if partition_prefix is None:
            partition_prefix = Path(index_prefix_path).name

        # Get executable paths
        partitioner_path = self._get_executable_path("partitioner")
        relayout_path = self._get_executable_path("index_relayout")

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to the graph_partition directory for temporary files
            graph_partition_dir = (
                Path(__file__).parent.parent / "third_party" / "DiskANN" / "graph_partition"
            )
            original_dir = os.getcwd()

            try:
                os.chdir(graph_partition_dir)

                # Create temporary data directory
                temp_data_dir = Path(temp_dir) / "data"
                temp_data_dir.mkdir(parents=True, exist_ok=True)

                # Set up paths for temporary files
                graph_path = temp_data_dir / "starling" / "_M_R_L_B" / "GRAPH"
                graph_gp_path = (
                    graph_path
                    / f"GP_TIMES_{params['gp_times']}_LOCK_{params['lock_nums']}_GP_USE_FREQ0_CUT{params['cut']}_SCALE{params['scale_factor']}"
                )
                graph_gp_path.mkdir(parents=True, exist_ok=True)

                # Find input index file
                old_index_file = f"{index_prefix_path}_disk_beam_search.index"
                if not os.path.exists(old_index_file):
                    old_index_file = f"{index_prefix_path}_disk.index"

                if not os.path.exists(old_index_file):
                    raise RuntimeError(f"Index file not found: {old_index_file}")

                # Run partitioner
                gp_file_path = graph_gp_path / "_part.bin"
                partitioner_cmd = [
                    partitioner_path,
                    "--index_file",
                    old_index_file,
                    "--data_type",
                    params["data_type"],
                    "--gp_file",
                    str(gp_file_path),
                    "-T",
                    str(params["thread_nums"]),
                    "--ldg_times",
                    str(params["gp_times"]),
                    "--scale",
                    str(params["scale_factor"]),
                    "--mode",
                    "1",
                ]

                print(f"Running partitioner: {' '.join(partitioner_cmd)}")
                result = subprocess.run(
                    partitioner_cmd, capture_output=True, text=True, cwd=graph_partition_dir
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Partitioner failed with return code {result.returncode}.\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}"
                    )

                # Run relayout
                part_tmp_index = graph_gp_path / "_part_tmp.index"
                relayout_cmd = [
                    relayout_path,
                    old_index_file,
                    str(gp_file_path),
                    params["data_type"],
                    "1",
                ]

                print(f"Running relayout: {' '.join(relayout_cmd)}")
                result = subprocess.run(
                    relayout_cmd, capture_output=True, text=True, cwd=graph_partition_dir
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Relayout failed with return code {result.returncode}.\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}"
                    )

                # Copy results to output directory
                disk_graph_path = Path(output_dir) / f"{partition_prefix}_disk_graph.index"
                partition_bin_path = Path(output_dir) / f"{partition_prefix}_partition.bin"

                shutil.copy2(part_tmp_index, disk_graph_path)
                shutil.copy2(gp_file_path, partition_bin_path)

                print(f"Results copied to: {output_dir}")
                return str(disk_graph_path), str(partition_bin_path)

            finally:
                os.chdir(original_dir)

    def get_partition_info(self, partition_bin_path: str) -> dict:
        """
        Get information about a partition file.

        Args:
            partition_bin_path: Path to the partition binary file

        Returns:
            Dictionary containing partition information
        """
        if not os.path.exists(partition_bin_path):
            raise FileNotFoundError(f"Partition file not found: {partition_bin_path}")

        # For now, return basic file information
        # In the future, this could parse the binary file for detailed info
        stat = os.stat(partition_bin_path)
        return {
            "file_size": stat.st_size,
            "file_path": partition_bin_path,
            "modified_time": stat.st_mtime,
        }


def partition_graph(
    index_prefix_path: str,
    output_dir: Optional[str] = None,
    partition_prefix: Optional[str] = None,
    build_type: str = "release",
    **kwargs,
) -> tuple[str, str]:
    """
    Convenience function to partition a graph index.

    Args:
        index_prefix_path: Path to the index prefix
        output_dir: Output directory (defaults to parent of index_prefix_path)
        partition_prefix: Prefix for output files (defaults to basename of index_prefix_path)
        build_type: Build type for executables ("debug" or "release")
        **kwargs: Additional parameters for graph partitioning

    Returns:
        Tuple of (disk_graph_index_path, partition_bin_path)
    """
    partitioner = GraphPartitioner(build_type=build_type)
    return partitioner.partition_graph(index_prefix_path, output_dir, partition_prefix, **kwargs)


# Example usage:
if __name__ == "__main__":
    # Example: partition an index
    try:
        disk_graph_path, partition_bin_path = partition_graph(
            "/path/to/your/index_prefix", gp_times=10, lock_nums=10, cut=100
        )
        print("Partitioning completed successfully!")
        print(f"Disk graph index: {disk_graph_path}")
        print(f"Partition binary: {partition_bin_path}")
    except Exception as e:
        print(f"Partitioning failed: {e}")
