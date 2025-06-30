import struct
import sys
import numpy as np
import os

INDEX_FLAT_L2_FOURCC = int.from_bytes(b'IxF2', 'little')
INDEX_FLAT_IP_FOURCC = int.from_bytes(b'IxFI', 'little')
INDEX_HNSW_FLAT_FOURCC = int.from_bytes(b'IHNf', 'little')
INDEX_HNSW_PQ_FOURCC = int.from_bytes(b'IHNp', 'little')
INDEX_HNSW_SQ_FOURCC = int.from_bytes(b'IHNs', 'little')
INDEX_HNSW_2L_FOURCC = int.from_bytes(b'IHN2', 'little')
INDEX_HNSW_CAGRA_FOURCC = int.from_bytes(b'IHNc', 'little')
NULL_INDEX_FOURCC = int.from_bytes(b'null', 'little')

HNSW_FOURCCS = {
    INDEX_HNSW_FLAT_FOURCC,
    INDEX_HNSW_PQ_FOURCC,
    INDEX_HNSW_SQ_FOURCC,
    INDEX_HNSW_2L_FOURCC,
    INDEX_HNSW_CAGRA_FOURCC,
}
FLAT_FOURCCS = {INDEX_FLAT_L2_FOURCC, INDEX_FLAT_IP_FOURCC}


# --- Helper functions for reading binary data ---

def read_struct(f, fmt):
    """Reads data according to the struct format."""
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("File ended unexpectedly.")
    return struct.unpack(fmt, data)[0]

def read_vector(f, element_fmt):
    """Reads a vector (size followed by data)."""
    count = read_struct(f, '<Q') # size_t usually 64-bit unsigned
    element_size = struct.calcsize(element_fmt)
    data_bytes = f.read(count * element_size)
    if len(data_bytes) != count * element_size:
         raise EOFError("File ended unexpectedly when reading vector data.")
    # Unpack the elements individually if needed, or return raw bytes
    # For simplicity here, we'll return the raw bytes and size
    return count, data_bytes

def read_vector_data(f, element_fmt, count):
    """Reads vector data when the count is known."""
    element_size = struct.calcsize(element_fmt)
    data_bytes = f.read(count * element_size)
    if len(data_bytes) != count * element_size:
         raise EOFError("File ended unexpectedly when reading vector data.")
    return data_bytes

# --- Main Separation Logic ---

def separate_hnsw_flat(input_filename, graph_output_filename, storage_output_filename):
    """
    Separates an IndexHNSWFlat file into graph and storage components.
    """
    print(f"Processing: {input_filename}")
    try:
        with open(input_filename, 'rb') as f_in, \
             open(graph_output_filename, 'wb') as f_graph_out, \
             open(storage_output_filename, 'wb') as f_storage_out:

            # 1. Read and write HNSW FourCC
            hnsw_fourcc = read_struct(f_in, '<I')
            if hnsw_fourcc != INDEX_HNSW_FLAT_FOURCC:
                print(f"Error: Expected IndexHNSWFlat FourCC ({INDEX_HNSW_FLAT_FOURCC:08x}), "
                      f"but got {hnsw_fourcc:08x}. Is this an IndexHNSWFlat file?", file=sys.stderr)
                return False
            f_graph_out.write(struct.pack('<I', hnsw_fourcc))
            print(f"  Index type: HNSWFlat ({hnsw_fourcc:08x})")

            # 2. Read and write Index Header
            # d, ntotal, dummy1, dummy2, is_trained, metric_type, [metric_arg]
            d = read_struct(f_in, '<i')
            ntotal = read_struct(f_in, '<q') # idx_t is int64 in Faiss default
            dummy1 = read_struct(f_in, '<q')
            dummy2 = read_struct(f_in, '<q')
            is_trained = read_struct(f_in, '?') # bool -> 1 byte
            metric_type = read_struct(f_in, '<i')
            metric_arg = 0.0
            header_data = struct.pack('<iq?i', d, ntotal, # omit dummies here
                                      is_trained, metric_type)
            if metric_type > 1:
                metric_arg = read_struct(f_in, '<f')
                header_data += struct.pack('<f', metric_arg)

            # Write header *without* dummies to graph file
            # We'll reconstruct the full header later if needed, but for now
            # just keep the essential parts. Alternatively, write the exact bytes read.
            # Let's write exact bytes for simplicity of reassembly
            f_graph_out.write(struct.pack('<i', d))
            f_graph_out.write(struct.pack('<q', ntotal))
            f_graph_out.write(struct.pack('<q', dummy1))
            f_graph_out.write(struct.pack('<q', dummy2))
            f_graph_out.write(struct.pack('?', is_trained))
            f_graph_out.write(struct.pack('<i', metric_type))
            if metric_type > 1:
                 f_graph_out.write(struct.pack('<f', metric_arg))

            print(f"  Dimensions (d): {d}")
            print(f"  Num vectors (ntotal): {ntotal}")
            print(f"  Is trained: {is_trained}")
            print(f"  Metric type: {metric_type}")
            if metric_type > 1:
                print(f"  Metric arg: {metric_arg}")

            # 3. Read and write HNSW struct data
            print("  Reading HNSW graph data...")
            # assign_probas (vector<double>)
            count, data = read_vector(f_in, '<d')
            f_graph_out.write(struct.pack('<Q', count))
            f_graph_out.write(data)
            print(f"    assign_probas size: {count}")

            # cum_nneighbor_per_level (vector<int>)
            count, data = read_vector(f_in, '<i')
            f_graph_out.write(struct.pack('<Q', count))
            f_graph_out.write(data)
            print(f"    cum_nneighbor_per_level size: {count}")

            # levels (vector<int>) - Store node levels
            count, data = read_vector(f_in, '<i')
            f_graph_out.write(struct.pack('<Q', count))
            f_graph_out.write(data)
            print(f"    levels size: {count}")

             # offsets (vector<size_t>) - Store offsets for neighbors
            count, data = read_vector(f_in, '<Q')
            f_graph_out.write(struct.pack('<Q', count))
            f_graph_out.write(data)
            print(f"    offsets size: {count}")

            # neighbors (vector<storage_idx_t> -> int32_t typically)
            count, data = read_vector(f_in, '<i') # Assuming storage_idx_t is int32
            f_graph_out.write(struct.pack('<Q', count))
            f_graph_out.write(data)
            print(f"    neighbors size: {count}")

            # entry_point, max_level, efConstruction, efSearch
            entry_point = read_struct(f_in, '<i')
            max_level = read_struct(f_in, '<i')
            efConstruction = read_struct(f_in, '<i')
            efSearch = read_struct(f_in, '<i')
            # Read and discard the dummy upper_beam
            _ = read_struct(f_in, '<i')

            f_graph_out.write(struct.pack('<i', entry_point))
            f_graph_out.write(struct.pack('<i', max_level))
            f_graph_out.write(struct.pack('<i', efConstruction))
            f_graph_out.write(struct.pack('<i', efSearch))
            f_graph_out.write(struct.pack('<i', 1)) # Write dummy upper_beam back
            print(f"    entry_point: {entry_point}")
            print(f"    max_level: {max_level}")
            print(f"    efConstruction: {efConstruction}")
            print(f"    efSearch: {efSearch}")


            # --- Storage Part ---
            print("  Reading storage (IndexFlat) data...")
            storage_start_pos = f_in.tell()

            # 4. Check: Read the storage FourCC (should be IndexFlat)
            storage_fourcc = read_struct(f_in, '<I')
            if storage_fourcc not in FLAT_FOURCCS:
                 print(f"Error: Expected IndexFlat FourCC ({list(FLAT_FOURCCS)}), "
                       f"but got {storage_fourcc:08x} after HNSW data.", file=sys.stderr)
                 return False
            print(f"    Storage type: IndexFlat ({storage_fourcc:08x})")

            # 5. Read the rest of the file as storage data
            f_in.seek(storage_start_pos) # Go back to start of storage
            storage_data = f_in.read() # Read everything remaining
            f_storage_out.write(storage_data)
            print(f"  Wrote {len(storage_data)} bytes to storage file.")

            # 6. Final Check: Did we reach the end of the input file?
            if f_in.read(1):
                 print("Warning: Unexpected data found after storage part in input file.", file=sys.stderr)

            print(f"Separation complete:")
            print(f"  Graph structure: {graph_output_filename}")
            print(f"  Vector storage: {storage_output_filename}")
            return True

    except EOFError as e:
        print(f"Error: Reached end of file unexpectedly. The input file might be incomplete or corrupted. {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return False

# --- Example Usage ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python separate_hnsw_flat.py <input_index_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_name = os.path.splitext(input_file)[0]
    graph_file = base_name + ".hnsw_graph"
    storage_file = base_name + ".flat_storage"

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    success = separate_hnsw_flat(input_file, graph_file, storage_file)
    if not success:
        sys.exit(1)