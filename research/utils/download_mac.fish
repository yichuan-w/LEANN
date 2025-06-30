#!/usr/bin/env fish

set scaling_out_dir "/Users/ec2-user/scaling_out"

# Define an array of paths to download
set paths \
    "examples/" \
    "indices/rpj_wiki/facebook/contriever-msmarco/diskann/_disk_graph.index" \
    "indices/rpj_wiki/facebook/contriever-msmarco/diskann/_partition.bin" \
    "indices/rpj_wiki/facebook/contriever-msmarco/diskann/ann_disk.index_medoids.bin" \
    "indices/rpj_wiki/facebook/contriever-msmarco/diskann/ann_disk.index_centroids.bin" \
    "indices/rpj_wiki/facebook/contriever-msmarco/diskann/ann_disk.index_max_base_norm.bin" \
    "embeddings/facebook/contriever-msmarco/rpj_wiki/compressed_10/" \
    "passages/rpj_wiki/8-shards/" \
    "indices/rpj_wiki/facebook/contriever-msmarco/flat_results_nq_k3.json"

# Download each path using a for loop
for path in $paths
    echo "Downloading $path..."
    # if ends with /, then create the directory
    if string match -q "*/" $path
        echo "Creating directory $scaling_out_dir/$path"
        mkdir -p "$scaling_out_dir/$path"
        aws s3 cp "s3://retrieval-scaling-out/$path" "$scaling_out_dir/$path" --recursive
    else
        aws s3 cp "s3://retrieval-scaling-out/$path" "$scaling_out_dir/$path"
    end
end

echo "Download completed."