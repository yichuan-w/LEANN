set -l index_dirs \
/opt/dlami/nvme/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/hnsw/hnsw_IP_M30_efC128.index \
/opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/indices/99_4_degree_based_hnsw_IP_M32_efC256/index.faiss \
/opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/indices/d9_hnsw_IP_M8_efC128/index.faiss \
/opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/indices/half_edges_IP_M32_efC128/index.faiss
# /opt/dlami/nvme/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/nsg_R16.index

set -l index_labels \
origin \
0.01per_M4_degree_based \
M8_merge_edge \
random_delete50
# nsg_R16

set -gx CUDA_VISIBLE_DEVICES 3

for i in (seq (count $index_dirs))
    set -l index_file $index_dirs[$i]
    set -l index_label $index_labels[$i]
    echo "Building HNSW index with $index_label..." >> ./large_graph_simple_build.log
    python -u large_graph_simple_build.py --index-file $index_file | tee -a ./large_graph_simple_build.log
end
