/*
Run with
g++ ./demo_reader.cpp -o ./demo_reader && ./demo_reader --stats \
/powerrag/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/diskann/_partition.bin
\
/powerrag/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/diskann/_disk_graph.index
*/

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits> // Include for std::numeric_limits
#include <string> // Include for std::string comparison
#include <vector>

#define READ_U64(f, val)                                                       \
  f.read(reinterpret_cast<char *>(&val), sizeof(uint64_t))
#define READ_U32(f, val)                                                       \
  f.read(reinterpret_cast<char *>(&val), sizeof(uint32_t))
#define SECTOR_SIZE 4096

// Helper: Get file size
static size_t get_file_size(const std::string &fname) {
  std::ifstream ifs(fname, std::ios::binary | std::ios::ate);
  if (ifs.fail() || !ifs.is_open()) {
    return 0;
  }
  return static_cast<size_t>(ifs.tellg());
}

// Print first few hex of sector for debug
static void print_hex(const char *buf, size_t len, size_t max_len = 64) {
  size_t show_len = (len < max_len) ? len : max_len;
  for (size_t i = 0; i < show_len; i++) {
    unsigned char c = (unsigned char)buf[i];
    std::cout << std::hex << std::setw(2) << std::setfill('0') << (unsigned)c
              << " ";
    if ((i + 1) % 16 == 0)
      std::cout << "\n   ";
  }
  std::cout << std::dec << "\n";
}

/*
  Corrected demo_reader:
  1) Read from partition.bin:
      - C, partition_nums, nd
      - graph_partitions[i]: all nodeIDs in partition i
      - id2partition[nodeID]: nodeID => partition i
  2) Read from _disk_graph.index:
      a) sector0 first has 2 ints: meta_n, meta_dim
      b) then meta_n uint64_t
         e.g.: [0]=nd, [1]=dim, [2]=??, [3]=max_node_len, [4]=C, [5]..??,
  [8]=file_size... specific positions need to be combined with relayout writing c) graph_node_len =
  max_node_len - dim_in_meta*sizeof(float) 3) User given target_node_id =>
      partition_id= id2partition[node_id]
      find node index j in graph_partitions[partition_id]
      offset = (partition_id+1)*4096 => sector
      adjacency_offset= j*graph_node_len => neighbor_count => neighbors
*/
int main(int argc, char **argv) {
  bool calculate_stats = false;
  // int arg_offset = 0; // Offset for positional arguments
  std::string partition_bin;
  std::string graph_index;
  uint64_t target_node_id = 0; // Initialize

  if (argc != 4) {
    std::cerr << "Usage:\n"
              << "  " << argv[0]
              << " <partition.bin> <disk_graph.index> <target_node_id>  (Reads "
                 "adjacency for a node)\n"
              << "  " << argv[0]
              << " --stats <partition.bin> <disk_graph.index>           "
                 "(Calculates degree statistics)\n";
    return 1;
  }

  // Check if the first argument is the stats flag
  if (std::string(argv[1]) == "--stats") {
    calculate_stats = true;
    partition_bin = argv[2];
    graph_index = argv[3];
    std::cout << "Mode: Calculating Degree Statistics\n";
  } else {
    // Assume default mode (single node lookup)
    calculate_stats = false;
    partition_bin = argv[1];
    graph_index = argv[2];
    try { // Add error handling for stoull
      target_node_id = std::stoull(argv[3]);
    } catch (const std::invalid_argument &ia) {
      std::cerr << "Error: Invalid target_node_id: " << argv[3] << std::endl;
      return 1;
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: target_node_id out of range: " << argv[3]
                << std::endl;
      return 1;
    }
    std::cout << "Mode: Single Node Lookup for Node ID " << target_node_id
              << "\n";
  }

  // 1) Read partition.bin
  std::ifstream pf(partition_bin, std::ios::binary);
  if (!pf.is_open()) {
    std::cerr << "Cannot open partition.bin: " << partition_bin << std::endl;
    return 1;
  }
  uint64_t C, partition_nums, nd;
  READ_U64(pf, C);
  READ_U64(pf, partition_nums);
  READ_U64(pf, nd);
  std::cout << "[partition.bin header] C=" << C
            << ", partition_nums=" << partition_nums << ", nd=" << nd
            << std::endl;

  // Read partition node lists
  std::vector<std::vector<uint32_t> > graph_partitions(partition_nums);
  for (uint64_t i = 0; i < partition_nums; i++) {
    uint32_t psize;
    READ_U32(pf, psize);
    graph_partitions[i].resize(psize);
    pf.read(reinterpret_cast<char *>(graph_partitions[i].data()),
            psize * sizeof(uint32_t));
  }
  // Read _id2partition[node], size= nd
  std::vector<uint32_t> id2partition(nd);
  pf.read(reinterpret_cast<char *>(id2partition.data()), nd * sizeof(uint32_t));
  pf.close();
  std::cout << "Done loading partition info.\n";

  if (target_node_id >= nd) {
    std::cerr << "target_node_id=" << target_node_id
              << " out of range nd=" << nd << std::endl;
    return 1;
  }

  // 2) Parse _disk_graph.index
  std::ifstream gf(graph_index, std::ios::binary);
  if (!gf.is_open()) {
    std::cerr << "Cannot open disk_graph.index: " << graph_index << std::endl;
    return 1;
  }
  // (a) sector0 => first read 2 ints
  int meta_n, meta_dim;
  gf.read((char *)&meta_n, sizeof(int));
  gf.read((char *)&meta_dim, sizeof(int));
  std::cout << "[debug] meta_n=" << meta_n << ", meta_dim=" << meta_dim << "\n";

  // (b) Read meta_n uint64_t
  std::vector<uint64_t> meta_info(meta_n);
  gf.read(reinterpret_cast<char *>(meta_info.data()),
          meta_n * sizeof(uint64_t));
  // Print
  for (int i = 0; i < meta_n; i++) {
    std::cout << " meta_info[" << i << "]= " << meta_info[i] << "\n";
  }

  size_t file_size = get_file_size(graph_index);
  std::cout << "[disk_graph.index size] " << file_size << " bytes\n";

  // **According to relayout log** you said: meta_info[0]=nd=60450220, meta_info[1]=dim=769,
  //    meta_info[2]=??(16495248?), meta_info[3]=max_node_len=3320,
  //    meta_info[4]=16 (C),
  //    meta_info[8]= 15475261440(file size)
  // We manually parse here first:
  uint64_t nd_in_meta = meta_info[0];
  uint64_t dim_in_meta = meta_info[1];
  uint64_t max_node_len = meta_info[3];
  uint64_t c_in_meta = meta_info[4];
  uint64_t entire_file_sz = meta_info[8];

  std::cout << "Based on meta_info:\n"
            << "  nd_in_meta= " << nd_in_meta
            << ", dim_in_meta= " << dim_in_meta
            << ", max_node_len= " << max_node_len
            << ", c_in_meta= " << c_in_meta
            << ", entire_file_size= " << entire_file_sz << "\n";

  // Calculate graph_node_len
  uint64_t dim_size = dim_in_meta * sizeof(float);
  uint64_t graph_node_len = max_node_len - dim_size;
  std::cout << " => graph_node_len= " << graph_node_len << "\n\n";

  if (calculate_stats) {
    // --- Degree Statistics Calculation Mode ---
    std::cout << " Calculated graph_node_len = " << graph_node_len << "\n\n";

    if (nd == 0) {
      std::cerr << "Graph has 0 nodes (nd=0). Cannot calculate stats."
                << std::endl;
      gf.close();
      return 1;
    }

    uint32_t min_degree = std::numeric_limits<uint32_t>::max();
    uint32_t max_degree = 0;
    uint64_t total_degree = 0;
    uint64_t nodes_processed = 0;
    std::vector<char> sectorBuf(SECTOR_SIZE);

    std::cout << "Calculating degrees for " << nd << " nodes across "
              << partition_nums << " partitions..." << std::endl;

    for (uint32_t p = 0; p < partition_nums; ++p) {
      uint64_t sector_offset = uint64_t(p + 1) * SECTOR_SIZE;
      gf.seekg(sector_offset, std::ios::beg);
      if (gf.fail()) {
        std::cerr << "Error seeking to sector offset for partition " << p
                  << std::endl;
        gf.close();
        return 1;
      }
      gf.read(sectorBuf.data(), SECTOR_SIZE);
      if (gf.fail() && !gf.eof()) {
        std::cerr << "Error reading sector data for partition " << p
                  << std::endl;
        gf.close();
        return 1;
      }
      gf.clear(); // Reset fail bits

      const auto &part_list = graph_partitions[p];
      for (size_t j = 0; j < part_list.size(); ++j) {
        uint64_t node_offset = j * graph_node_len;
        if (node_offset + sizeof(uint32_t) > SECTOR_SIZE) {
          std::cerr << "Error: Node offset out of sector bounds.\n"
                    << " Partition=" << p << ", node_subIndex=" << j
                    << ", node_offset=" << node_offset
                    << ", graph_node_len=" << graph_node_len << std::endl;
          gf.close();
          return 1;
        }
        char *adjacency_ptr = sectorBuf.data() + node_offset;
        uint32_t neighbor_count = *reinterpret_cast<uint32_t *>(adjacency_ptr);
        min_degree = std::min(min_degree, neighbor_count);
        max_degree = std::max(max_degree, neighbor_count);
        total_degree += neighbor_count;
        nodes_processed++;
      }
      if (p % 10 == 0 || p == partition_nums - 1) {
        std::cout << "  Processed partition " << p + 1 << " / "
                  << partition_nums << "...\r" << std::flush;
      }
    }
    std::cout << "\nFinished processing partitions." << std::endl;

    if (nodes_processed != nd) {
      std::cerr << "Warning: Processed " << nodes_processed
                << " nodes, but expected " << nd << std::endl;
    }

    double avg_degree = (nd > 0) ? static_cast<double>(total_degree) / nd : 0.0;
    std::cout << "\n--- Degree Statistics ---\n";
    std::cout << "Min Degree: "
              << (min_degree == std::numeric_limits<uint32_t>::max()
                      ? 0
                      : min_degree)
              << std::endl; // Handle case of 0 nodes
    std::cout << "Max Degree: " << max_degree << std::endl;
    std::cout << "Avg Degree: " << std::fixed << std::setprecision(2)
              << avg_degree << std::endl;
    std::cout << "Total Degree (Sum): " << total_degree << std::endl;
    std::cout << "Nodes Processed: " << nodes_processed << std::endl;

  } else {
    uint64_t nd_in_meta = meta_info[0];
    uint64_t c_in_meta = meta_info[4];
    uint64_t entire_file_sz = meta_info[8];
    std::cout << "Based on meta_info:\n"
              << "  nd_in_meta= " << nd_in_meta
              << ", dim_in_meta= " << dim_in_meta
              << ", max_node_len= " << max_node_len
              << ", c_in_meta= " << c_in_meta
              << ", entire_file_size= " << entire_file_sz << "\n";
    std::cout << " => graph_node_len= " << graph_node_len << "\n\n";

    if (target_node_id >= nd) {
      std::cerr << "target_node_id=" << target_node_id
                << " out of range nd=" << nd << std::endl;
      gf.close();
      return 1;
    }

    // We need id2partition only for single-node lookup
    std::vector<uint32_t> id2partition(nd);
    { // Read id2partition again as it was skipped before
      std::ifstream pf_again(partition_bin, std::ios::binary);
      uint64_t header_offset =
          3 * sizeof(uint64_t); // Skip C, partition_nums, nd
      uint64_t partition_list_offset = 0;
      for (uint64_t i = 0; i < partition_nums; i++) {
        partition_list_offset += sizeof(uint32_t); // Size field
        partition_list_offset +=
            graph_partitions[i].size() * sizeof(uint32_t); // Data
      }
      pf_again.seekg(header_offset + partition_list_offset, std::ios::beg);
      pf_again.read(reinterpret_cast<char *>(id2partition.data()),
                    nd * sizeof(uint32_t));
      // Error check pf_again if needed
    }

    // 3) Find target_node_id => partition_id => subIndex
    uint32_t partition_id = id2partition[target_node_id];
    if (partition_id >= partition_nums) {
      std::cerr << "Partition ID out-of-range for target node.\n";
      gf.close();
      return 1;
    }
    const auto &part_list = graph_partitions[partition_id]; // Use const ref
    auto it =
        std::find(part_list.begin(), part_list.end(), (uint32_t)target_node_id);
    if (it == part_list.end()) {
      std::cerr << "Cannot find node " << target_node_id << " in partition "
                << partition_id << std::endl;
      gf.close();
      return 1;
    }
    size_t j = std::distance(part_list.begin(), it);

    // 4) sector => (partition_id+1)* 4096
    uint64_t sector_offset = uint64_t(partition_id + 1) * SECTOR_SIZE;
    gf.seekg(sector_offset, std::ios::beg);
    std::vector<char> sectorBuf(SECTOR_SIZE);
    gf.read(sectorBuf.data(), SECTOR_SIZE);
    if (gf.fail() && !gf.eof()) {
      std::cerr << "Error reading sector data for partition " << partition_id
                << std::endl;
      gf.close();
      return 1;
    }
    gf.clear(); // Reset fail bits

    std::cout << "Partition #" << partition_id
              << ", nodeCount= " << part_list.size()
              << ", offset= " << sector_offset << "\n"
              << " first64 hex:\n   ";
    print_hex(sectorBuf.data(), SECTOR_SIZE, 64);

    // adjacency_offset= j* graph_node_len
    uint64_t node_offset = j * graph_node_len;
    if (node_offset + sizeof(uint32_t) >
        SECTOR_SIZE) { // Check only for neighbor_count read first
      std::cerr << "Out-of-range. j=" << j << ", node_offset=" << node_offset
                << ", node_offset+4=" << (node_offset + sizeof(uint32_t))
                << "> 4096\n";
      gf.close();
      return 1;
    }

    char *adjacency_ptr = sectorBuf.data() + node_offset;
    uint32_t neighbor_count = *reinterpret_cast<uint32_t *>(adjacency_ptr);
    std::cout << "[Node " << target_node_id << "] partition=" << partition_id
              << ", subIndex=" << j << ", adjacency_offset=" << node_offset
              << ", neighbor_count=" << neighbor_count << "\n";

    size_t needed = neighbor_count * sizeof(uint32_t);
    if (node_offset + sizeof(uint32_t) + needed > SECTOR_SIZE) {
      std::cerr << "Neighbors partly out-of-range => neighbor_count="
                << neighbor_count << "\n";
      // Option: Can still print partial list if needed, but indicating it's
      // truncated
      gf.close();
      return 1; // Or handle differently
    }
    std::vector<uint32_t> neighbors(neighbor_count);
    memcpy(neighbors.data(), adjacency_ptr + sizeof(uint32_t), needed);

    std::cout << "  neighbors=[";
    for (size_t kk = 0; kk < std::min<size_t>(10, neighbor_count); kk++) {
      std::cout << neighbors[kk];
      if (kk + 1 < std::min<size_t>(10, neighbor_count))
        std::cout << ", ";
    }
    if (neighbor_count > 10)
      std::cout << " ... (total " << neighbor_count << ")";
    std::cout << "]\n";
  } // End of else (single node lookup mode)

  gf.close();
  return 0;
}