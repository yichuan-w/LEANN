import faiss
hnsw_index = faiss.read_index("/opt/dlami/nvme/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/hnsw/hnsw_IP_M30_efC128.index", faiss.IO_FLAG_ONDISK_SAME_DIR)

# print total number of nodes
print(hnsw_index.ntotal)

# print  stats of the graph
print(hnsw_index.hnsw.print_neighbor_stats(0))


# save_degree_distribution
hnsw_index.hnsw.save_degree_distribution(0, "degree_distribution_HNSW_M30.txt")