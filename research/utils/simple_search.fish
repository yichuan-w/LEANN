

set -l BIG_GRAPH /opt/dlami/nvme/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/hnsw/hnsw_IP_M30_efC128.index \
    /opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/indices/99_4_degree_based_hnsw_IP_M32_efC256/index.faiss \
    /opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/indices/d9_hnsw_IP_M8_efC128/index.faiss \
    /opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/indices/half_edges_IP_M32_efC128/index.faiss

set -l LABELS hnsw_IP_M30_efC128 \
    99_4_degree_based_hnsw_IP_M32_efC256 \
    d9_hnsw_IP_M8_efC128 \
    half_edges_IP_M32_efC128

for i in (seq 1 4)
    set graph (string join / $BIG_GRAPH[$i])
    set label (string join / $LABELS[$i])
    echo "Building HNSW index with $label..."
    python ./faiss/demo/large_graph_simple_build.py --index-file $graph
end
