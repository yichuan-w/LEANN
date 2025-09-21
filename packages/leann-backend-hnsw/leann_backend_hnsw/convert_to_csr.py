import argparse
import gc  # Import garbage collector interface
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Set up logging to avoid print buffer issues
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logger.setLevel(log_level)

# --- FourCCs (add more if needed) ---
INDEX_HNSW_FLAT_FOURCC = int.from_bytes(b"IHNf", "little")
# Add other HNSW fourccs if you expect different storage types inside HNSW
# INDEX_HNSW_PQ_FOURCC = int.from_bytes(b'IHNp', 'little')
# INDEX_HNSW_SQ_FOURCC = int.from_bytes(b'IHNs', 'little')
# INDEX_HNSW_CAGRA_FOURCC = int.from_bytes(b'IHNc', 'little') # Example

EXPECTED_HNSW_FOURCCS = {INDEX_HNSW_FLAT_FOURCC}  # Modify if needed
NULL_INDEX_FOURCC = int.from_bytes(b"null", "little")

# --- Helper functions for reading/writing binary data ---


def read_struct(f, fmt):
    """Reads data according to the struct format."""
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError(
            f"File ended unexpectedly reading struct fmt '{fmt}'. Expected {size} bytes, got {len(data)}."
        )
    return struct.unpack(fmt, data)[0]


def read_vector_raw(f, element_fmt_char):
    """Reads a vector (size followed by data), returns count and raw bytes."""
    count = -1  # Initialize count
    total_bytes = -1  # Initialize total_bytes
    try:
        count = read_struct(f, "<Q")  # size_t usually 64-bit unsigned
        element_size = struct.calcsize(element_fmt_char)
        # --- FIX for MemoryError: Check for unreasonably large count ---
        max_reasonable_count = 10 * (10**9)  # ~10 billion elements limit
        if count > max_reasonable_count or count < 0:
            raise MemoryError(
                f"Vector count {count} seems unreasonably large, possibly due to file corruption or incorrect format read."
            )

        total_bytes = count * element_size
        # --- FIX for MemoryError: Check for huge byte size before allocation ---
        max_reasonable_bytes = 50 * (1024**3)  # ~50 GB limit
        if total_bytes > max_reasonable_bytes or total_bytes < 0:  # Check for overflow
            raise MemoryError(
                f"Attempting to read {total_bytes} bytes ({count} elements * {element_size} bytes/element), which exceeds the safety limit. File might be corrupted or format mismatch."
            )

        data_bytes = f.read(total_bytes)

        if len(data_bytes) != total_bytes:
            raise EOFError(
                f"File ended unexpectedly reading vector data. Expected {total_bytes} bytes, got {len(data_bytes)}."
            )
        return count, data_bytes
    except (MemoryError, OverflowError) as e:
        # Add context to the error message
        print(
            f"\nError during raw vector read (element_fmt='{element_fmt_char}', count={count}, total_bytes={total_bytes}): {e}",
            file=sys.stderr,
        )
        raise e  # Re-raise the original error type


def read_numpy_vector(f, np_dtype, struct_fmt_char):
    """Reads a vector into a NumPy array."""
    count = -1  # Initialize count for robust error handling
    print(
        f"  Reading vector (dtype={np_dtype}, fmt='{struct_fmt_char}')... ",
        end="",
        flush=True,
    )
    try:
        count, data_bytes = read_vector_raw(f, struct_fmt_char)
        print(f"Count={count}, Bytes={len(data_bytes)}")
        if count > 0 and len(data_bytes) > 0:
            arr = np.frombuffer(data_bytes, dtype=np_dtype)
            if arr.size != count:
                raise ValueError(
                    f"Inconsistent array size after reading. Expected {count}, got {arr.size}"
                )
            return arr
        elif count == 0:
            return np.array([], dtype=np_dtype)
        else:
            raise ValueError("Read zero bytes but count > 0.")
    except MemoryError as e:
        # Now count should be defined (or -1 if error was in read_struct)
        print(
            f"\nMemoryError creating NumPy array (dtype={np_dtype}, count={count}). {e}",
            file=sys.stderr,
        )
        raise e
    except Exception as e:  # Catch other potential errors like ValueError
        print(
            f"\nError reading numpy vector (dtype={np_dtype}, fmt='{struct_fmt_char}', count={count}): {e}",
            file=sys.stderr,
        )
        raise e


def write_numpy_vector(f, arr, struct_fmt_char):
    """Writes a NumPy array as a vector (size followed by data)."""
    count = arr.size
    f.write(struct.pack("<Q", count))
    try:
        expected_dtype = np.dtype(struct_fmt_char)
        if arr.dtype != expected_dtype:
            data_to_write = arr.astype(expected_dtype).tobytes()
        else:
            data_to_write = arr.tobytes()
        f.write(data_to_write)
        del data_to_write  # Hint GC
    except MemoryError as e:
        print(
            f"\nMemoryError converting NumPy array to bytes for writing (size={count}, dtype={arr.dtype}). {e}",
            file=sys.stderr,
        )
        raise e


def write_list_vector(f, lst, struct_fmt_char):
    """Writes a Python list as a vector iteratively."""
    count = len(lst)
    f.write(struct.pack("<Q", count))
    fmt = "<" + struct_fmt_char
    chunk_size = 1024 * 1024
    element_size = struct.calcsize(fmt)
    # Allocate buffer outside the loop if possible, or handle MemoryError during allocation
    try:
        buffer = bytearray(chunk_size * element_size)
    except MemoryError:
        print(
            f"MemoryError: Cannot allocate buffer for writing list vector chunk (size {chunk_size * element_size} bytes).",
            file=sys.stderr,
        )
        raise
    buffer_count = 0

    for i, item in enumerate(lst):
        try:
            offset = buffer_count * element_size
            struct.pack_into(fmt, buffer, offset, item)
            buffer_count += 1

            if buffer_count == chunk_size or i == count - 1:
                f.write(buffer[: buffer_count * element_size])
                buffer_count = 0

        except struct.error as e:
            print(
                f"\nStruct packing error for item {item} at index {i} with format '{fmt}'. {e}",
                file=sys.stderr,
            )
            raise e


def get_cum_neighbors(cum_nneighbor_per_level_np, level):
    """Helper to get cumulative neighbors count, matching C++ logic."""
    if level < 0:
        return 0
    if level < len(cum_nneighbor_per_level_np):
        return cum_nneighbor_per_level_np[level]
    else:
        return cum_nneighbor_per_level_np[-1] if len(cum_nneighbor_per_level_np) > 0 else 0


def write_compact_format(
    f_out,
    original_hnsw_data,
    assign_probas_np,
    cum_nneighbor_per_level_np,
    levels_np,
    compact_level_ptr,
    compact_node_offsets_np,
    compact_neighbors_data,
    storage_fourcc,
    storage_data,
):
    """Write HNSW data in compact format following C++ read order exactly."""
    # Write IndexHNSW Header
    f_out.write(struct.pack("<I", original_hnsw_data["index_fourcc"]))
    f_out.write(struct.pack("<i", original_hnsw_data["d"]))
    f_out.write(struct.pack("<q", original_hnsw_data["ntotal"]))
    f_out.write(struct.pack("<q", original_hnsw_data["dummy1"]))
    f_out.write(struct.pack("<q", original_hnsw_data["dummy2"]))
    f_out.write(struct.pack("<?", original_hnsw_data["is_trained"]))
    f_out.write(struct.pack("<i", original_hnsw_data["metric_type"]))
    if original_hnsw_data["metric_type"] > 1:
        f_out.write(struct.pack("<f", original_hnsw_data["metric_arg"]))

    # Write HNSW struct parts (standard order)
    write_numpy_vector(f_out, assign_probas_np, "d")
    write_numpy_vector(f_out, cum_nneighbor_per_level_np, "i")
    write_numpy_vector(f_out, levels_np, "i")

    # Write compact format flag
    f_out.write(struct.pack("<?", True))  # storage_is_compact = True

    # Write compact data in CORRECT C++ read order: level_ptr, node_offsets FIRST
    if isinstance(compact_level_ptr, np.ndarray):
        write_numpy_vector(f_out, compact_level_ptr, "Q")
    else:
        write_list_vector(f_out, compact_level_ptr, "Q")

    write_numpy_vector(f_out, compact_node_offsets_np, "Q")

    # Write HNSW scalar parameters
    f_out.write(struct.pack("<i", original_hnsw_data["entry_point"]))
    f_out.write(struct.pack("<i", original_hnsw_data["max_level"]))
    f_out.write(struct.pack("<i", original_hnsw_data["efConstruction"]))
    f_out.write(struct.pack("<i", original_hnsw_data["efSearch"]))
    f_out.write(struct.pack("<i", original_hnsw_data["dummy_upper_beam"]))

    # Write storage fourcc (this determines how to read what follows)
    f_out.write(struct.pack("<I", storage_fourcc))

    # Write compact neighbors data AFTER storage fourcc
    write_list_vector(f_out, compact_neighbors_data, "i")

    # Write storage data if not NULL (only after neighbors)
    if storage_fourcc != NULL_INDEX_FOURCC and storage_data:
        f_out.write(storage_data)


@dataclass
class HNSWComponents:
    original_hnsw_data: dict[str, Any]
    assign_probas_np: np.ndarray
    cum_nneighbor_per_level_np: np.ndarray
    levels_np: np.ndarray
    is_compact: bool
    compact_level_ptr: Optional[np.ndarray] = None
    compact_node_offsets_np: Optional[np.ndarray] = None
    compact_neighbors_data: Optional[list[int]] = None
    offsets_np: Optional[np.ndarray] = None
    neighbors_np: Optional[np.ndarray] = None
    storage_fourcc: int = NULL_INDEX_FOURCC
    storage_data: bytes = b""


def _read_hnsw_structure(f) -> HNSWComponents:
    original_hnsw_data: dict[str, Any] = {}

    hnsw_index_fourcc = read_struct(f, "<I")
    if hnsw_index_fourcc not in EXPECTED_HNSW_FOURCCS:
        raise ValueError(
            f"Unexpected HNSW FourCC: {hnsw_index_fourcc:08x}. Expected one of {EXPECTED_HNSW_FOURCCS}."
        )

    original_hnsw_data["index_fourcc"] = hnsw_index_fourcc
    original_hnsw_data["d"] = read_struct(f, "<i")
    original_hnsw_data["ntotal"] = read_struct(f, "<q")
    original_hnsw_data["dummy1"] = read_struct(f, "<q")
    original_hnsw_data["dummy2"] = read_struct(f, "<q")
    original_hnsw_data["is_trained"] = read_struct(f, "?")
    original_hnsw_data["metric_type"] = read_struct(f, "<i")
    original_hnsw_data["metric_arg"] = 0.0
    if original_hnsw_data["metric_type"] > 1:
        original_hnsw_data["metric_arg"] = read_struct(f, "<f")

    assign_probas_np = read_numpy_vector(f, np.float64, "d")
    cum_nneighbor_per_level_np = read_numpy_vector(f, np.int32, "i")
    levels_np = read_numpy_vector(f, np.int32, "i")

    ntotal = len(levels_np)
    if ntotal != original_hnsw_data["ntotal"]:
        original_hnsw_data["ntotal"] = ntotal

    pos_before_compact = f.tell()
    is_compact_flag = None
    try:
        is_compact_flag = read_struct(f, "<?")
    except EOFError:
        is_compact_flag = None

    if is_compact_flag:
        compact_level_ptr = read_numpy_vector(f, np.uint64, "Q")
        compact_node_offsets_np = read_numpy_vector(f, np.uint64, "Q")

        original_hnsw_data["entry_point"] = read_struct(f, "<i")
        original_hnsw_data["max_level"] = read_struct(f, "<i")
        original_hnsw_data["efConstruction"] = read_struct(f, "<i")
        original_hnsw_data["efSearch"] = read_struct(f, "<i")
        original_hnsw_data["dummy_upper_beam"] = read_struct(f, "<i")

        storage_fourcc = read_struct(f, "<I")
        compact_neighbors_data_np = read_numpy_vector(f, np.int32, "i")
        compact_neighbors_data = compact_neighbors_data_np.tolist()
        storage_data = f.read()

        return HNSWComponents(
            original_hnsw_data=original_hnsw_data,
            assign_probas_np=assign_probas_np,
            cum_nneighbor_per_level_np=cum_nneighbor_per_level_np,
            levels_np=levels_np,
            is_compact=True,
            compact_level_ptr=compact_level_ptr,
            compact_node_offsets_np=compact_node_offsets_np,
            compact_neighbors_data=compact_neighbors_data,
            storage_fourcc=storage_fourcc,
            storage_data=storage_data,
        )

    # Non-compact case
    f.seek(pos_before_compact)

    pos_before_probe = f.tell()
    try:
        suspected_flag = read_struct(f, "<B")
        if suspected_flag != 0x00:
            f.seek(pos_before_probe)
    except EOFError:
        f.seek(pos_before_probe)

    offsets_np = read_numpy_vector(f, np.uint64, "Q")
    neighbors_np = read_numpy_vector(f, np.int32, "i")

    original_hnsw_data["entry_point"] = read_struct(f, "<i")
    original_hnsw_data["max_level"] = read_struct(f, "<i")
    original_hnsw_data["efConstruction"] = read_struct(f, "<i")
    original_hnsw_data["efSearch"] = read_struct(f, "<i")
    original_hnsw_data["dummy_upper_beam"] = read_struct(f, "<i")

    storage_fourcc = NULL_INDEX_FOURCC
    storage_data = b""
    try:
        storage_fourcc = read_struct(f, "<I")
        storage_data = f.read()
    except EOFError:
        storage_fourcc = NULL_INDEX_FOURCC

    return HNSWComponents(
        original_hnsw_data=original_hnsw_data,
        assign_probas_np=assign_probas_np,
        cum_nneighbor_per_level_np=cum_nneighbor_per_level_np,
        levels_np=levels_np,
        is_compact=False,
        offsets_np=offsets_np,
        neighbors_np=neighbors_np,
        storage_fourcc=storage_fourcc,
        storage_data=storage_data,
    )


def _read_hnsw_structure_from_file(path: str) -> HNSWComponents:
    with open(path, "rb") as f:
        return _read_hnsw_structure(f)


def write_original_format(
    f_out,
    original_hnsw_data,
    assign_probas_np,
    cum_nneighbor_per_level_np,
    levels_np,
    offsets_np,
    neighbors_np,
    storage_fourcc,
    storage_data,
):
    """Write non-compact HNSW data in original FAISS order."""

    f_out.write(struct.pack("<I", original_hnsw_data["index_fourcc"]))
    f_out.write(struct.pack("<i", original_hnsw_data["d"]))
    f_out.write(struct.pack("<q", original_hnsw_data["ntotal"]))
    f_out.write(struct.pack("<q", original_hnsw_data["dummy1"]))
    f_out.write(struct.pack("<q", original_hnsw_data["dummy2"]))
    f_out.write(struct.pack("<?", original_hnsw_data["is_trained"]))
    f_out.write(struct.pack("<i", original_hnsw_data["metric_type"]))
    if original_hnsw_data["metric_type"] > 1:
        f_out.write(struct.pack("<f", original_hnsw_data["metric_arg"]))

    write_numpy_vector(f_out, assign_probas_np, "d")
    write_numpy_vector(f_out, cum_nneighbor_per_level_np, "i")
    write_numpy_vector(f_out, levels_np, "i")

    write_numpy_vector(f_out, offsets_np, "Q")
    write_numpy_vector(f_out, neighbors_np, "i")

    f_out.write(struct.pack("<i", original_hnsw_data["entry_point"]))
    f_out.write(struct.pack("<i", original_hnsw_data["max_level"]))
    f_out.write(struct.pack("<i", original_hnsw_data["efConstruction"]))
    f_out.write(struct.pack("<i", original_hnsw_data["efSearch"]))
    f_out.write(struct.pack("<i", original_hnsw_data["dummy_upper_beam"]))

    f_out.write(struct.pack("<I", storage_fourcc))
    if storage_fourcc != NULL_INDEX_FOURCC and storage_data:
        f_out.write(storage_data)


def prune_hnsw_embeddings(input_filename: str, output_filename: str) -> bool:
    """Rewrite an HNSW index while dropping the embedded storage section."""

    start_time = time.time()
    try:
        with open(input_filename, "rb") as f_in, open(output_filename, "wb") as f_out:
            original_hnsw_data: dict[str, Any] = {}

            hnsw_index_fourcc = read_struct(f_in, "<I")
            if hnsw_index_fourcc not in EXPECTED_HNSW_FOURCCS:
                print(
                    f"Error: Expected HNSW Index FourCC ({list(EXPECTED_HNSW_FOURCCS)}), got {hnsw_index_fourcc:08x}.",
                    file=sys.stderr,
                )
                return False

            original_hnsw_data["index_fourcc"] = hnsw_index_fourcc
            original_hnsw_data["d"] = read_struct(f_in, "<i")
            original_hnsw_data["ntotal"] = read_struct(f_in, "<q")
            original_hnsw_data["dummy1"] = read_struct(f_in, "<q")
            original_hnsw_data["dummy2"] = read_struct(f_in, "<q")
            original_hnsw_data["is_trained"] = read_struct(f_in, "?")
            original_hnsw_data["metric_type"] = read_struct(f_in, "<i")
            original_hnsw_data["metric_arg"] = 0.0
            if original_hnsw_data["metric_type"] > 1:
                original_hnsw_data["metric_arg"] = read_struct(f_in, "<f")

            assign_probas_np = read_numpy_vector(f_in, np.float64, "d")
            cum_nneighbor_per_level_np = read_numpy_vector(f_in, np.int32, "i")
            levels_np = read_numpy_vector(f_in, np.int32, "i")

            ntotal = len(levels_np)
            if ntotal != original_hnsw_data["ntotal"]:
                original_hnsw_data["ntotal"] = ntotal

            pos_before_compact = f_in.tell()
            is_compact_flag = None
            try:
                is_compact_flag = read_struct(f_in, "<?")
            except EOFError:
                is_compact_flag = None

            if is_compact_flag:
                compact_level_ptr = read_numpy_vector(f_in, np.uint64, "Q")
                compact_node_offsets_np = read_numpy_vector(f_in, np.uint64, "Q")

                original_hnsw_data["entry_point"] = read_struct(f_in, "<i")
                original_hnsw_data["max_level"] = read_struct(f_in, "<i")
                original_hnsw_data["efConstruction"] = read_struct(f_in, "<i")
                original_hnsw_data["efSearch"] = read_struct(f_in, "<i")
                original_hnsw_data["dummy_upper_beam"] = read_struct(f_in, "<i")

                _storage_fourcc = read_struct(f_in, "<I")
                compact_neighbors_data_np = read_numpy_vector(f_in, np.int32, "i")
                compact_neighbors_data = compact_neighbors_data_np.tolist()
                _storage_data = f_in.read()

                write_compact_format(
                    f_out,
                    original_hnsw_data,
                    assign_probas_np,
                    cum_nneighbor_per_level_np,
                    levels_np,
                    compact_level_ptr,
                    compact_node_offsets_np,
                    compact_neighbors_data,
                    NULL_INDEX_FOURCC,
                    b"",
                )
            else:
                f_in.seek(pos_before_compact)

                pos_before_probe = f_in.tell()
                try:
                    suspected_flag = read_struct(f_in, "<B")
                    if suspected_flag != 0x00:
                        f_in.seek(pos_before_probe)
                except EOFError:
                    f_in.seek(pos_before_probe)

                offsets_np = read_numpy_vector(f_in, np.uint64, "Q")
                neighbors_np = read_numpy_vector(f_in, np.int32, "i")

                original_hnsw_data["entry_point"] = read_struct(f_in, "<i")
                original_hnsw_data["max_level"] = read_struct(f_in, "<i")
                original_hnsw_data["efConstruction"] = read_struct(f_in, "<i")
                original_hnsw_data["efSearch"] = read_struct(f_in, "<i")
                original_hnsw_data["dummy_upper_beam"] = read_struct(f_in, "<i")

                _storage_fourcc = None
                _storage_data = b""
                try:
                    _storage_fourcc = read_struct(f_in, "<I")
                    _storage_data = f_in.read()
                except EOFError:
                    _storage_fourcc = NULL_INDEX_FOURCC

                write_original_format(
                    f_out,
                    original_hnsw_data,
                    assign_probas_np,
                    cum_nneighbor_per_level_np,
                    levels_np,
                    offsets_np,
                    neighbors_np,
                    NULL_INDEX_FOURCC,
                    b"",
                )

        print(f"[{time.time() - start_time:.2f}s] Pruned embeddings from {input_filename}")
        return True
    except Exception as exc:
        print(f"Failed to prune embeddings: {exc}", file=sys.stderr)
        return False


# --- Main Conversion Logic ---


def convert_hnsw_graph_to_csr(input_filename, output_filename, prune_embeddings=True):
    """
    Converts an HNSW graph file to the CSR format.
    Supports both original and already-compact formats (backward compatibility).

    Args:
        input_filename: Input HNSW index file
        output_filename: Output CSR index file
        prune_embeddings: Whether to prune embedding storage (write NULL storage marker)
    """
    # Keep prints simple; rely on CI runner to flush output as needed

    print(f"Starting conversion: {input_filename} -> {output_filename}")
    start_time = time.time()
    original_hnsw_data = {}
    neighbors_np = None  # Initialize to allow check in finally block
    try:
        with open(input_filename, "rb") as f_in, open(output_filename, "wb") as f_out:
            # --- Read IndexHNSW FourCC and Header ---
            print(f"[{time.time() - start_time:.2f}s] Reading Index HNSW header...")
            # ... (Keep the header reading logic as before) ...
            hnsw_index_fourcc = read_struct(f_in, "<I")
            if hnsw_index_fourcc not in EXPECTED_HNSW_FOURCCS:
                print(
                    f"Error: Expected HNSW Index FourCC ({list(EXPECTED_HNSW_FOURCCS)}), got {hnsw_index_fourcc:08x}.",
                    file=sys.stderr,
                )
                return False
            original_hnsw_data["index_fourcc"] = hnsw_index_fourcc
            original_hnsw_data["d"] = read_struct(f_in, "<i")
            original_hnsw_data["ntotal"] = read_struct(f_in, "<q")
            original_hnsw_data["dummy1"] = read_struct(f_in, "<q")
            original_hnsw_data["dummy2"] = read_struct(f_in, "<q")
            original_hnsw_data["is_trained"] = read_struct(f_in, "?")
            original_hnsw_data["metric_type"] = read_struct(f_in, "<i")
            original_hnsw_data["metric_arg"] = 0.0
            if original_hnsw_data["metric_type"] > 1:
                original_hnsw_data["metric_arg"] = read_struct(f_in, "<f")
            print(
                f"[{time.time() - start_time:.2f}s]   Header read: d={original_hnsw_data['d']}, ntotal={original_hnsw_data['ntotal']}"
            )

            # --- Read original HNSW struct data ---
            print(f"[{time.time() - start_time:.2f}s] Reading HNSW struct vectors...")
            assign_probas_np = read_numpy_vector(f_in, np.float64, "d")
            print(
                f"[{time.time() - start_time:.2f}s]   Read assign_probas ({assign_probas_np.size})"
            )
            gc.collect()

            cum_nneighbor_per_level_np = read_numpy_vector(f_in, np.int32, "i")
            print(
                f"[{time.time() - start_time:.2f}s]   Read cum_nneighbor_per_level ({cum_nneighbor_per_level_np.size})"
            )
            gc.collect()

            levels_np = read_numpy_vector(f_in, np.int32, "i")
            print(f"[{time.time() - start_time:.2f}s]   Read levels ({levels_np.size})")
            gc.collect()

            ntotal = len(levels_np)
            if ntotal != original_hnsw_data["ntotal"]:
                print(
                    f"Warning: ntotal mismatch! Header says {original_hnsw_data['ntotal']}, levels vector size is {ntotal}. Using levels vector size.",
                    file=sys.stderr,
                )
                original_hnsw_data["ntotal"] = ntotal

            # --- Check for compact format flag ---
            print(f"[{time.time() - start_time:.2f}s]   Probing for compact storage flag...")
            pos_before_compact = f_in.tell()
            try:
                is_compact_flag = read_struct(f_in, "<?")
                print(f"[{time.time() - start_time:.2f}s]   Found compact flag: {is_compact_flag}")

                if is_compact_flag:
                    # Input is already in compact format - read compact data
                    print(
                        f"[{time.time() - start_time:.2f}s]   Input is already in compact format, reading compact data..."
                    )

                    compact_level_ptr = read_numpy_vector(f_in, np.uint64, "Q")
                    print(
                        f"[{time.time() - start_time:.2f}s]   Read compact_level_ptr ({compact_level_ptr.size})"
                    )

                    compact_node_offsets_np = read_numpy_vector(f_in, np.uint64, "Q")
                    print(
                        f"[{time.time() - start_time:.2f}s]   Read compact_node_offsets ({compact_node_offsets_np.size})"
                    )

                    # Read scalar parameters
                    original_hnsw_data["entry_point"] = read_struct(f_in, "<i")
                    original_hnsw_data["max_level"] = read_struct(f_in, "<i")
                    original_hnsw_data["efConstruction"] = read_struct(f_in, "<i")
                    original_hnsw_data["efSearch"] = read_struct(f_in, "<i")
                    original_hnsw_data["dummy_upper_beam"] = read_struct(f_in, "<i")
                    print(
                        f"[{time.time() - start_time:.2f}s]   Read scalar params (ep={original_hnsw_data['entry_point']}, max_lvl={original_hnsw_data['max_level']})"
                    )

                    # Read storage fourcc
                    storage_fourcc = read_struct(f_in, "<I")
                    print(
                        f"[{time.time() - start_time:.2f}s]   Found storage fourcc: {storage_fourcc:08x}"
                    )

                    if prune_embeddings and storage_fourcc != NULL_INDEX_FOURCC:
                        # Read compact neighbors data
                        compact_neighbors_data_np = read_numpy_vector(f_in, np.int32, "i")
                        print(
                            f"[{time.time() - start_time:.2f}s]   Read compact neighbors data ({compact_neighbors_data_np.size})"
                        )
                        compact_neighbors_data = compact_neighbors_data_np.tolist()
                        del compact_neighbors_data_np

                        # Skip storage data and write with NULL marker
                        print(
                            f"[{time.time() - start_time:.2f}s]   Pruning embeddings: Writing NULL storage marker."
                        )
                        storage_fourcc = NULL_INDEX_FOURCC
                    elif not prune_embeddings:
                        # Read and preserve compact neighbors and storage
                        compact_neighbors_data_np = read_numpy_vector(f_in, np.int32, "i")
                        compact_neighbors_data = compact_neighbors_data_np.tolist()
                        del compact_neighbors_data_np

                        # Read remaining storage data
                        storage_data = f_in.read()
                    else:
                        # Already pruned (NULL storage)
                        compact_neighbors_data_np = read_numpy_vector(f_in, np.int32, "i")
                        compact_neighbors_data = compact_neighbors_data_np.tolist()
                        del compact_neighbors_data_np
                        storage_data = b""

                    # Write the updated compact format
                    print(f"[{time.time() - start_time:.2f}s] Writing updated compact format...")
                    write_compact_format(
                        f_out,
                        original_hnsw_data,
                        assign_probas_np,
                        cum_nneighbor_per_level_np,
                        levels_np,
                        compact_level_ptr,
                        compact_node_offsets_np,
                        compact_neighbors_data,
                        storage_fourcc,
                        storage_data if not prune_embeddings else b"",
                    )

                    print(f"[{time.time() - start_time:.2f}s] Conversion complete.")
                    return True

                else:
                    # is_compact=False, rewind and read original format
                    f_in.seek(pos_before_compact)
                    print(
                        f"[{time.time() - start_time:.2f}s]   Compact flag is False, reading original format..."
                    )

            except EOFError:
                # No compact flag found, assume original format
                f_in.seek(pos_before_compact)
                print(
                    f"[{time.time() - start_time:.2f}s]   No compact flag found, assuming original format..."
                )

            # --- Handle potential extra byte in original format (like C++ code) ---
            print(
                f"[{time.time() - start_time:.2f}s]   Probing for potential extra byte before non-compact offsets..."
            )
            pos_before_probe = f_in.tell()
            try:
                suspected_flag = read_struct(f_in, "<B")  # Read 1 byte
                if suspected_flag == 0x00:
                    print(
                        f"[{time.time() - start_time:.2f}s]   Found and consumed an unexpected 0x00 byte."
                    )
                elif suspected_flag == 0x01:
                    print(
                        f"[{time.time() - start_time:.2f}s]   ERROR: Found 0x01 but is_compact should be False"
                    )
                    raise ValueError("Inconsistent compact flag state")
                else:
                    # Rewind - this byte is part of offsets data
                    f_in.seek(pos_before_probe)
                    print(
                        f"[{time.time() - start_time:.2f}s]   Rewound to original position (byte was 0x{suspected_flag:02x})"
                    )
            except EOFError:
                f_in.seek(pos_before_probe)
                print(
                    f"[{time.time() - start_time:.2f}s]   No extra byte found (EOF), proceeding with offsets read"
                )

            # --- Read original format data ---
            offsets_np = read_numpy_vector(f_in, np.uint64, "Q")
            print(f"[{time.time() - start_time:.2f}s]   Read offsets ({offsets_np.size})")
            if len(offsets_np) != ntotal + 1:
                raise ValueError(
                    f"Inconsistent offsets size: len(levels)={ntotal} but len(offsets)={len(offsets_np)}"
                )
            gc.collect()

            print(f"[{time.time() - start_time:.2f}s]   Attempting to read neighbors vector...")
            neighbors_np = read_numpy_vector(f_in, np.int32, "i")
            print(f"[{time.time() - start_time:.2f}s]   Read neighbors ({neighbors_np.size})")
            expected_neighbors_size = offsets_np[-1] if ntotal > 0 else 0
            if neighbors_np.size != expected_neighbors_size:
                print(
                    f"Warning: neighbors vector size mismatch. Expected {expected_neighbors_size} based on offsets, got {neighbors_np.size}."
                )
            gc.collect()

            original_hnsw_data["entry_point"] = read_struct(f_in, "<i")
            original_hnsw_data["max_level"] = read_struct(f_in, "<i")
            original_hnsw_data["efConstruction"] = read_struct(f_in, "<i")
            original_hnsw_data["efSearch"] = read_struct(f_in, "<i")
            original_hnsw_data["dummy_upper_beam"] = read_struct(f_in, "<i")
            print(
                f"[{time.time() - start_time:.2f}s]   Read scalar params (ep={original_hnsw_data['entry_point']}, max_lvl={original_hnsw_data['max_level']})"
            )

            print(f"[{time.time() - start_time:.2f}s] Checking for storage data...")
            storage_fourcc = None
            try:
                storage_fourcc = read_struct(f_in, "<I")
                print(
                    f"[{time.time() - start_time:.2f}s]   Found storage fourcc: {storage_fourcc:08x}."
                )
            except EOFError:
                print(f"[{time.time() - start_time:.2f}s]   No storage data found (EOF).")
            except Exception as e:
                print(
                    f"[{time.time() - start_time:.2f}s]   Error reading potential storage data: {e}"
                )

            # --- Perform Conversion ---
            print(f"[{time.time() - start_time:.2f}s] Converting to CSR format...")

            # Use lists for potentially huge data, np for offsets
            compact_neighbors_data = []
            compact_level_ptr = []
            compact_node_offsets_np = np.zeros(ntotal + 1, dtype=np.uint64)

            current_level_ptr_idx = 0
            current_data_idx = 0
            total_valid_neighbors_counted = 0  # For validation

            # Optimize calculation by getting slices once per node if possible
            for i in range(ntotal):
                if i > 0 and i % (ntotal // 100 or 1) == 0:  # Log progress roughly every 1%
                    progress = (i / ntotal) * 100
                    elapsed = time.time() - start_time
                    print(
                        f"\r[{elapsed:.2f}s]   Converting node {i}/{ntotal} ({progress:.1f}%)...",
                        end="",
                    )

                node_max_level = levels_np[i] - 1
                if node_max_level < -1:
                    node_max_level = -1

                node_ptr_start_index = current_level_ptr_idx
                compact_node_offsets_np[i] = node_ptr_start_index

                original_offset_start = offsets_np[i]
                num_pointers_expected = (node_max_level + 1) + 1

                for level in range(node_max_level + 1):
                    compact_level_ptr.append(current_data_idx)

                    begin_orig_np = original_offset_start + get_cum_neighbors(
                        cum_nneighbor_per_level_np, level
                    )
                    end_orig_np = original_offset_start + get_cum_neighbors(
                        cum_nneighbor_per_level_np, level + 1
                    )

                    begin_orig = int(begin_orig_np)
                    end_orig = int(end_orig_np)

                    neighbors_len = len(neighbors_np)  # Cache length
                    begin_orig = min(max(0, begin_orig), neighbors_len)
                    end_orig = min(max(begin_orig, end_orig), neighbors_len)

                    if begin_orig < end_orig:
                        # Slicing creates a copy, could be memory intensive for large M
                        # Consider iterating if memory becomes an issue here
                        level_neighbors_slice = neighbors_np[begin_orig:end_orig]
                        valid_neighbors_mask = level_neighbors_slice >= 0
                        num_valid = np.count_nonzero(valid_neighbors_mask)

                        if num_valid > 0:
                            # Append valid neighbors
                            compact_neighbors_data.extend(
                                level_neighbors_slice[valid_neighbors_mask]
                            )
                            current_data_idx += num_valid
                            total_valid_neighbors_counted += num_valid

                compact_level_ptr.append(current_data_idx)
                current_level_ptr_idx += num_pointers_expected

            compact_node_offsets_np[ntotal] = current_level_ptr_idx
            print(
                f"\r[{time.time() - start_time:.2f}s]   Conversion loop finished.                        "
            )  # Clear progress line

            # --- Validation Checks ---
            print(f"[{time.time() - start_time:.2f}s] Running validation checks...")
            valid_check_passed = True
            # Check 1: Total valid neighbors count
            print("    Checking total valid neighbor count...")
            expected_valid_count = np.sum(neighbors_np >= 0)
            if total_valid_neighbors_counted != len(compact_neighbors_data):
                print(
                    f"Error: Mismatch between counted valid neighbors ({total_valid_neighbors_counted}) and final compact_data size ({len(compact_neighbors_data)})!",
                    file=sys.stderr,
                )
                valid_check_passed = False
            if expected_valid_count != len(compact_neighbors_data):
                print(
                    f"Error: Mismatch between NumPy count of valid neighbors ({expected_valid_count}) and final compact_data size ({len(compact_neighbors_data)})!",
                    file=sys.stderr,
                )
                valid_check_passed = False
            else:
                print(f"    OK: Total valid neighbors = {len(compact_neighbors_data)}")

            # Check 2: Final pointer indices consistency
            print("    Checking final pointer indices...")
            if compact_node_offsets_np[ntotal] != len(compact_level_ptr):
                print(
                    f"Error: Final node offset ({compact_node_offsets_np[ntotal]}) doesn't match level_ptr size ({len(compact_level_ptr)})!",
                    file=sys.stderr,
                )
                valid_check_passed = False
            if (
                len(compact_level_ptr) > 0 and compact_level_ptr[-1] != len(compact_neighbors_data)
            ) or (len(compact_level_ptr) == 0 and len(compact_neighbors_data) != 0):
                last_ptr = compact_level_ptr[-1] if len(compact_level_ptr) > 0 else -1
                print(
                    f"Error: Last level pointer ({last_ptr}) doesn't match compact_data size ({len(compact_neighbors_data)})!",
                    file=sys.stderr,
                )
                valid_check_passed = False
            else:
                print("    OK: Final pointers match data size.")

            if not valid_check_passed:
                print(
                    "Error: Validation checks failed. Output file might be incorrect.",
                    file=sys.stderr,
                )
                # Optional: Exit here if validation fails
                # return False

            # --- Explicitly delete large intermediate arrays ---
            print(
                f"[{time.time() - start_time:.2f}s] Deleting original neighbors and offsets arrays..."
            )
            del neighbors_np
            del offsets_np
            gc.collect()

            print(
                f"    CSR Stats: |data|={len(compact_neighbors_data)}, |level_ptr|={len(compact_level_ptr)}"
            )

            # --- Write CSR HNSW graph data using unified function ---
            print(
                f"[{time.time() - start_time:.2f}s] Writing CSR HNSW graph data in FAISS-compatible order..."
            )

            # Determine storage fourcc and data based on prune_embeddings
            if prune_embeddings:
                print("   Pruning embeddings: Writing NULL storage marker.")
                output_storage_fourcc = NULL_INDEX_FOURCC
                storage_data = b""
            else:
                # Keep embeddings - read and preserve original storage data
                if storage_fourcc and storage_fourcc != NULL_INDEX_FOURCC:
                    print("   Preserving embeddings: Reading original storage data...")
                    storage_data = f_in.read()  # Read remaining storage data
                    output_storage_fourcc = storage_fourcc
                    print(f"   Read {len(storage_data)} bytes of storage data")
                else:
                    print("   No embeddings found in original file (NULL storage)")
                    output_storage_fourcc = NULL_INDEX_FOURCC
                    storage_data = b""

            # Use the unified write function
            write_compact_format(
                f_out,
                original_hnsw_data,
                assign_probas_np,
                cum_nneighbor_per_level_np,
                levels_np,
                compact_level_ptr,
                compact_node_offsets_np,
                compact_neighbors_data,
                output_storage_fourcc,
                storage_data,
            )

            # Clean up memory
            del assign_probas_np, cum_nneighbor_per_level_np, levels_np
            del compact_neighbors_data, compact_level_ptr, compact_node_offsets_np
            gc.collect()

            end_time = time.time()
            print(f"[{end_time - start_time:.2f}s] Conversion complete.")
            return True

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_filename}", file=sys.stderr)
        return False
    except MemoryError as e:
        print(
            f"\nFatal MemoryError during conversion: {e}. Insufficient RAM.",
            file=sys.stderr,
        )
        # Clean up potentially partially written output file?
        try:
            os.remove(output_filename)
        except OSError:
            pass
        return False
    except EOFError as e:
        print(
            f"Error: Reached end of file unexpectedly reading {input_filename}. {e}",
            file=sys.stderr,
        )
        try:
            os.remove(output_filename)
        except OSError:
            pass
        return False
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        try:
            os.remove(output_filename)
        except OSError:
            pass
        return False
    # Ensure neighbors_np is deleted even if an error occurs after its allocation
    finally:
        try:
            if "neighbors_np" in locals() and neighbors_np is not None:
                del neighbors_np
                gc.collect()
        except NameError:
            pass


def prune_hnsw_embeddings_inplace(index_filename: str) -> bool:
    """Convenience wrapper to prune embeddings in-place."""

    temp_path = f"{index_filename}.prune.tmp"
    success = prune_hnsw_embeddings(index_filename, temp_path)
    if success:
        try:
            os.replace(temp_path, index_filename)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to replace original index with pruned version: {exc}")
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return False
    else:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    return success


# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Faiss IndexHNSWFlat file to a CSR-based HNSW graph file."
    )
    parser.add_argument("input_index_file", help="Path to the input IndexHNSWFlat file")
    parser.add_argument(
        "output_csr_graph_file", help="Path to write the output CSR HNSW graph file"
    )
    parser.add_argument(
        "--prune-embeddings",
        action="store_true",
        default=True,
        help="Prune embedding storage (write NULL storage marker)",
    )
    parser.add_argument(
        "--keep-embeddings",
        action="store_true",
        help="Keep embedding storage (overrides --prune-embeddings)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_index_file):
        print(f"Error: Input file not found: {args.input_index_file}", file=sys.stderr)
        sys.exit(1)

    if os.path.abspath(args.input_index_file) == os.path.abspath(args.output_csr_graph_file):
        print("Error: Input and output filenames cannot be the same.", file=sys.stderr)
        sys.exit(1)

    prune_embeddings = args.prune_embeddings and not args.keep_embeddings
    success = convert_hnsw_graph_to_csr(
        args.input_index_file, args.output_csr_graph_file, prune_embeddings
    )
    if not success:
        sys.exit(1)
