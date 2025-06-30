#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static const size_t DISKANN_SECTOR_LEN = 4096; // Typical sector size

// ! Use float as CoordT
using CoordT = float;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <diskann_index_file> <output_degree_file>" << std::endl;
    return -1;
  }

  std::string disk_index_path = argv[1];
  std::string output_degree_path = argv[2];
  std::ifstream in(disk_index_path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Failed to open file: " << disk_index_path << std::endl;
    return -1;
  }

  // =========== 1) Read meta information (corresponds to
  // save_bin<uint64_t>(...,...,...,1,0)) ============== Read bin header:
  // (npts_i32, dim_i32)
  int32_t meta_count_i32 = 0, meta_dim_i32 = 0;
  in.read(reinterpret_cast<char *>(&meta_count_i32), sizeof(int32_t));
  in.read(reinterpret_cast<char *>(&meta_dim_i32), sizeof(int32_t));
  size_t meta_count = static_cast<size_t>(meta_count_i32);
  size_t meta_dim = static_cast<size_t>(meta_dim_i32);

  // According to the diskann::save_bin writing method, here meta_dim is usually
  // 1
  std::cout << "[LOG] meta_count=" << meta_count << ", meta_dim=" << meta_dim
            << std::endl;
  if (meta_dim != 1) {
    std::cerr << "[ERROR] meta_dim != 1,不符合 create_disk_layout 的写盘约定。"
              << std::endl;
    return -1;
  }

  // Read meta array
  std::vector<uint64_t> meta(meta_count);
  in.read(reinterpret_cast<char *>(meta.data()), meta_count * sizeof(uint64_t));
  if (!in.good()) {
    std::cerr << "[ERROR] Failed to read meta array, file is incomplete."
              << std::endl;
    return -1;
  }

  // meta[0..] Metadata
  // 0: npts_64, 1: ndims_64, 2: medoid, 3: max_node_len, 4: nnodes_per_sector,
  // 5: vamana_frozen_num, 6: vamana_frozen_loc, 7: append_reorder_data, ...
  const uint64_t npts_64 = meta[0];
  const uint64_t ndims_64 = meta[1];
  const uint64_t medoid = meta[2];
  const uint64_t max_node_len = meta[3];
  const uint64_t nnodes_per_sector = meta[4];
  const uint64_t vamana_frozen_num = meta[5];
  const uint64_t vamana_frozen_loc = meta[6];
  const uint64_t append_reorder_data = meta[7];

  std::cout << "[LOG] npts_64=" << npts_64 << " ndims_64=" << ndims_64
            << " max_node_len=" << max_node_len
            << " nnodes_per_sector=" << nnodes_per_sector << std::endl;
  // If append_reorder_data==1, it means that reorder_data is appended at the
  // end of the index, but it does not affect the degree statistics, we can
  // ignore that part of the vector.

  // =========== 2) Skip the first sector (all empty/placeholder information)
  // ==============
  in.seekg(DISKANN_SECTOR_LEN, std::ios::beg);
  if (!in.good()) {
    std::cerr << "[ERROR] Failed to seek to the first sector." << std::endl;
    return -1;
  }

  // =========== 3) Calculate the total number of sectors ==============
  // In create_disk_layout:
  //    If nnodes_per_sector > 0, then n_sectors = ceil(npts_64 /
  //   nnodes_per_sector)  Otherwise nsectors_per_node = ceil(max_node_len /
  //   4096), n_sectors = nsectors_per_node * npts_64
  uint64_t n_sectors = 0;
  if (nnodes_per_sector > 0) {
    // Equivalent to Roundup(npts_64, nnodes_per_sector) / nnodes_per_sector
    n_sectors = (npts_64 + nnodes_per_sector - 1) / nnodes_per_sector;
  } else {
    // multi-sector per node
    uint64_t nsectors_per_node =
        (max_node_len + DISKANN_SECTOR_LEN - 1) / DISKANN_SECTOR_LEN;
    n_sectors = nsectors_per_node * npts_64;
  }
  std::cout << "[LOG] estimated #sectors storing adjacency = " << n_sectors
            << std::endl;

  // =========== 4) Read the degree of all nodes in order ==============
  // The memory layout of adjacency_count in each node: offset = ndims_64 *
  // sizeof(CoordT) This is followed by 4 bytes for the number of neighbors
  // uint32_t If you want to read the complete neighbor list, it is
  // adjacency_count * sizeof(uint32_t) But we only count the count
  std::vector<uint32_t> degrees(npts_64, 0); // Store the degree of each node
  size_t node_id = 0;                        // Current node number
  // Buffer for reading one sector at a time
  std::vector<char> sector_buf(DISKANN_SECTOR_LEN, 0);
  // If nnodes_per_sector>0, it means that one sector holds multiple nodes
  // Otherwise, one node occupies nsectors_per_node sectors
  if (nnodes_per_sector > 0) {
    // Read one sector at a time
    for (uint64_t s = 0; s < n_sectors; s++) {
      in.read((char *)sector_buf.data(), DISKANN_SECTOR_LEN);
      if (!in.good()) {
        if (node_id < npts_64) {
          std::cerr << "[ERROR] Failed to read sector " << s
                    << ", nodes not finished, file error or incomplete."
                    << std::endl;
          return -1;
        }
        break; // If all nodes are read, you can exit
      }
      // Parse each node in sector_buf
      for (uint64_t i = 0; i < nnodes_per_sector; i++) {
        if (node_id >= npts_64)
          break; // All node degrees have been obtained
        // The starting offset of the node in sector_buf
        size_t node_offset = i * max_node_len;
        // offset first skips ndims_64 * sizeof(CoordT)
        size_t degree_offset = node_offset + ndims_64 * sizeof(CoordT);
        // Ensure not out of bounds
        if (degree_offset + sizeof(uint32_t) > sector_buf.size()) {
          std::cerr << "[ERROR] 不应该发生: 读取degree越过了扇区边界."
                    << std::endl;
          return -1;
        }
        uint32_t deg = 0;
        memcpy(&deg, sector_buf.data() + degree_offset, sizeof(uint32_t));
        degrees[node_id] = deg;
        node_id++;
      }
    }
  } else {
    // Each node occupies nsectors_per_node sectors
    uint64_t nsectors_per_node =
        (max_node_len + DISKANN_SECTOR_LEN - 1) / DISKANN_SECTOR_LEN;
    // Read each node
    for (uint64_t n = 0; n < npts_64; n++) {
      // Read multiple sectors into a multi-sector buffer
      std::vector<char> node_buf(nsectors_per_node * DISKANN_SECTOR_LEN, 0);
      in.read((char *)node_buf.data(), node_buf.size());
      if (!in.good()) {
        std::cerr << "[ERROR] Failed to read sector corresponding to node " << n
                  << ", file error or incomplete." << std::endl;
        return -1;
      }
      // Parse the degree in node_buf
      size_t degree_offset = ndims_64 * sizeof(CoordT);
      if (degree_offset + sizeof(uint32_t) > node_buf.size()) {
        std::cerr << "[ERROR] Should not happen: reading degree beyond node "
                     "region."
                  << std::endl;
        return -1;
      }
      uint32_t deg = 0;
      memcpy(&deg, node_buf.data() + degree_offset, sizeof(uint32_t));
      degrees[n] = deg;
    }
  }

  // We assert here: node_id should equal npts_64 (in multi-node mode)
  if (nnodes_per_sector > 0) {
    if (node_id != npts_64) {
      std::cerr << "[ERROR] Actually read " << node_id
                << " nodes, but meta npts_64=" << npts_64
                << ", file may be incorrect or parsing method is wrong."
                << std::endl;
      return -1;
    }
  }

  // =========== 5) Calculate min / max / average degree ==============
  uint64_t sum_deg = 0;
  uint32_t min_deg = std::numeric_limits<uint32_t>::max();
  uint32_t max_deg = 0;

  for (uint64_t n = 0; n < npts_64; n++) {
    uint32_t d = degrees[n];
    sum_deg += d;
    if (d < min_deg)
      min_deg = d;
    if (d > max_deg)
      max_deg = d;
  }
  double avg_deg = (npts_64 == 0) ? 0.0 : double(sum_deg) / double(npts_64);

  // =========== 6) Output results ==============
  std::cout << "DiskANN index file: " << disk_index_path << std::endl;
  std::cout << "Total points: " << npts_64 << std::endl;
  std::cout << "Min degree : " << min_deg << std::endl;
  std::cout << "Max degree : " << max_deg << std::endl;
  std::cout << "Avg degree : " << avg_deg << std::endl;

  // =========== 7) Write degrees to output file ==============
  std::ofstream out_deg(output_degree_path);
  if (!out_deg.is_open()) {
    std::cerr << "[ERROR] Failed to open output file: " << output_degree_path
              << std::endl;
    // Don't necessarily exit, maybe just warn? Depends on desired behavior.
    // For now, we continue closing the input file.
  } else {
    std::cout << "[LOG] Writing degrees to " << output_degree_path << "..."
              << std::endl;
    for (uint64_t n = 0; n < npts_64; n++) {
      out_deg << degrees[n] << std::endl;
    }
    out_deg.close();
    std::cout << "[LOG] Finished writing degrees." << std::endl;
  }

  in.close();
  return 0;
}
