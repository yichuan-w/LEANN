import faiss
nsg_index = faiss.read_index("/opt/dlami/nvme/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/nsg_R16.index", faiss.IO_FLAG_ONDISK_SAME_DIR)

# print total number of nodes
print(nsg_index.ntotal)

# print stats of the graph
print(nsg_index.nsg.print_neighbor_stats(0))

# save degree distribution
nsg_index.nsg.save_degree_distribution("degree_distribution_NSG_R60.txt")