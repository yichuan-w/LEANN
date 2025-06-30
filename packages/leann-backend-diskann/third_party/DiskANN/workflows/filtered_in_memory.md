**Usage for filtered indices**
================================
## Building a filtered Index
DiskANN provides two algorithms for building an index with filters support: filtered-vamana and stitched-vamana. Here, we describe the parameters for building both. `apps/build_memory_index.cpp` and `apps/build_stitched_index.cpp` are respectively used to build each kind of index. 

### 1. filtered-vamana

1. **`--data_type`**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **`--dist_fn`**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips).
3. **`--data_file`**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
4. **`--index_path_prefix`**: The constructed index components will be saved to this path prefix.
5. **`-R (--max_degree)`** (default is 64): the degree of the graph index, typically between 32 and 150. Larger R will result in larger indices and longer indexing times, but might yield better search quality. 
6. **`-L (--Lbuild)`** (default is 100): the size of search list we maintain during index building. Typical values are between 75 to 400. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Ensure that value of L is at least that of R value unless you need to build indices really quickly and can somewhat compromise on quality. Note that this is to be used only for building an unfiltered index. The corresponding search list parameter for a filtered index is managed by `--FilteredLbuild`.
7. **`--alpha`** (default is 1.2): A float value between 1.0 and 1.5 which determines the diameter of the graph, which will be approximately *log n* to the base alpha. Typical values are between 1 to 1.5. 1 will yield the sparsest graph, 1.5 will yield denser graphs. 
8. **`-T (--num_threads)`** (default is to get_omp_num_procs()): number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).
9. **`--build_PQ_bytes`** (default is 0): Set to a positive value less than the dimensionality of the data to enable faster index build with PQ based distance comparisons. Defaults to using full precision vectors for distance comparisons.
10. **`--use_opq`**: use the flag to use OPQ rather than PQ compression. OPQ is more space efficient for some high dimensional datasets, but also needs a bit more build time.
11. **`--label_file`**: Filter data for each point, in `.txt` format. Line `i` of the file consists of a comma-separated list of filters corresponding to point `i` in the file passed via `--data_file`.
12. **`--universal_label`**: Optionally, the the filter data may contain a "wild-card" filter corresponding to all filters. This is referred to as a universal label. Note that if a point has the universal label, then the filter data must only have the universal label on the line corresponding to said point.
13. **`--FilteredLbuild`**: If building a filtered index, we maintain a separate search list from the one provided by `--Lbuild`. 

### 2. stitched-vamana
1. **`--data_type`**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **`--data_path`**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
3. **`--index_path_prefix`**: The constructed index components will be saved to this path prefix.
4. **`-R (--max_degree)`** (default is 64): Recall that stitched-vamana first builds a sub-index for each filter. This parameter sets the max degree for each sub-index.
5. **`-L (--Lbuild)`** (default is 100): the size of search list we maintain during sub-index building. Typical values are between 75 to 400. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Ensure that value of L is at least that of R value unless you need to build indices really quickly and can somewhat compromise on quality. 
6. **`--alpha`** (default is 1.2): A float value between 1.0 and 1.5 which determines the diameter of the graph, which will be approximately *log n* to the base alpha. Typical values are between 1 to 1.5. 1 will yield the sparsest graph, 1.5 will yield denser graphs. 
7. **`-T (--num_threads)`** (default is to get_omp_num_procs()): number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).
8. **`--label_file`**: Filter data for each point, in `.txt` format. Line `i` of the file consists of a comma-separated list of filters corresponding to point `i` in the file passed via `--data_file`.
9. **`--universal_label`**: Optionally, the the filter data may contain a "wild-card" filter corresponding to all filters. This is referred to as a universal label. Note that if a point has the universal label, then the filter data must only have the universal label on the line corresponding to said point.
10. **`--Stitched_R`**: Once all sub-indices are "stitched" together, we prune the resulting graph down to the degree given by this parameter.

## Computing a groundtruth file for a filtered index
In order to evaluate the performance of our algorithms, we can compare its results (i.e. the top `k` neighbors found for each query) against the results found by an exact nearest neighbor search. We provide the program `apps/utils/compute_groundtruth.cpp` to provide the results for the latter:

1. **`--data_type`** The type of dataset you built an index with. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **`--dist_fn`**: There are two distance functions supported: l2 and mips.
3. **`--base_file`**: The input data over which to build an index, in .bin format. Corresponds to the `--data_path` argument from above.
4. **`--query_file`**: The queries to be searched on, which are stored in the same .bin format.
5. **`--label_file`**: Filter data for each point, in `.txt` format. Line `i` of the file consists of a comma-separated list of filters corresponding to point `i` in the file passed via `--data_file`. 
6. **`--filter_label`**: Filter for each query. For each query, a search is performed with this filter.
7. **`--universal_label`**: Corresponds to the universal label passed when building an index with filter support.
8. **`--gt_file`**: File to output results to. The binary file starts with `n`, the number of queries (4 bytes), followed by `d`, the number of ground truth elements per query (4 bytes), followed by `n*d` entries per query representing the `d` closest IDs per query in integer format,  followed by `n*d` entries representing the corresponding distances (float). Total file size is `8 + 4*n*d + 4*n*d` bytes.
9. **`-K`**: The number of nearest neighbors to compute for each query. 



## Searching a Filtered Index

Searching a filtered index uses the `apps/search_memory_index.cpp`:

1. **`--data_type`**: The type of dataset you built the index on. float(32 bit), signed int8 and unsigned uint8 are supported. Use the same data type as in arg (1) above used in building the index.
2. **`--dist_fn`**: There are two distance functions supported: l2 and mips. There is an additional *fast_l2* implementation that could provide faster results for small (about a million-sized) indices. Use the same distance as in arg (2) above used in building the index. Note that stitched-vamana only supports l2.
3. **`--index_path_prefix`**: index built above in argument (4).
4. **`--result_path`**: search results will be stored in files, one per L value (see last arg), with specified prefix, in binary format.
5. **`-T (--num_threads)`**: The number of threads used for searching. Threads run in parallel and one thread handles one query at a time. More threads will result in higher aggregate query throughput, but may lead to higher per-query latency, especially if the DRAM bandwidth is a bottleneck. So find the balance depending on throughput and latency required for your application.
6. **`--query_file`**: The queries to be searched on in same binary file format as the data file (ii) above. The query file must be the same type as in argument (1).
7. **`--filter_label`**: The filter to be used when searching an index with filters. For each query, a search is performed with this filter.
8. **`--gt_file`**: The ground truth file for the queries and data file used in index construction.  Use "null" if you do not have this file and if you do not want to compute recall. Note that if building a filtered index, a special groundtruth must be computed, as described above.
9. **`-K`**: search for *K* neighbors and measure *K*-recall@*K*, meaning the intersection between the retrieved top-*K* nearest neighbors and ground truth *K* nearest neighbors.
10. **`-L (--search_list)`**: A list of search_list sizes to perform search with. Larger parameters will result in slower latencies, but higher accuracies. Must be atleast the value of *K* in (7).

Example with SIFT10K:
--------------------
We demonstrate how to work through this pipeline using the SIFT10K dataset (http://corpus-texmex.irisa.fr/). Before starting, make sure you have compiled diskANN according to the instructions in the README and can see the following binaries (paths with respect to repository root):
- `build/apps/utils/compute_groundtruth`
- `build/apps/utils/fvecs_to_bin`
- `build/apps/build_memory_index`
- `build/apps/build_stitched_index`
- `build/apps/search_memory_index`

Now, download the base and query set and convert the data to binary format:
```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -zxvf siftsmall.tar.gz
build/apps/utils/fvecs_to_bin float siftsmall/siftsmall_base.fvecs siftsmall/siftsmall_base.bin
build/apps/utils/fvecs_to_bin float siftsmall/siftsmall_query.fvecs siftsmall/siftsmall_query.bin
```

We now need to make label file for our vectors. For convenience, we've included a synthetic label generator through which we can generate label file as follow
```bash
  build/apps/utils/generate_synthetic_labels  --num_labels 50 --num_points 10000  --output_file ./rand_labels_50_10K.txt --distribution_type zipf
```
Note : `distribution_type` can be `rand` or `zipf`

This will genearate label file with 10000 data points with 50 distinct labels, ranging from 1 to 50 assigned using zipf distribution (0 is the universal label).

Label count for each unique label in the generated label file can be printed with help of following command
```bash
  build/apps/utils/stats_label_data.exe --labels_file ./rand_labels_50_10K.txt --universal_label 0
```
	
Note that neither approach is designed for use with random synthetic labels, which will lead to unpredictable accuracy at search time.

Now build and search the index and measure the recall using ground truth computed using bruteforce. We search for results with the filter 35.
```bash
build/apps/utils/compute_groundtruth --data_type float --dist_fn l2 --base_file siftsmall/siftsmall_base.bin --query_file siftsmall/siftsmall_query.bin --gt_file siftsmall/siftsmall_gt_35.bin --K 100 --label_file ./rand_labels_50_10K.txt --filter_label 35 --universal_label 0
build/apps/build_memory_index  --data_type float --dist_fn l2 --data_path siftsmall/siftsmall_base.bin --index_path_prefix siftsmall/siftsmall_R32_L50_filtered_index -R 32 --FilteredLbuild 50 --alpha 1.2 --label_file ./rand_labels_50_10K.txt --universal_label 0
build/apps/build_stitched_index --data_type float --data_path siftsmall/siftsmall_base.bin --index_path_prefix siftsmall/siftsmall_R20_L40_SR32_stitched_index -R 20 -L 40 --stitched_R 32 --alpha 1.2 --label_file ./rand_labels_50_10K.txt --universal_label 0
build/apps/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/siftsmall_R20_L40_SR32_filtered_index --query_file siftsmall/siftsmall_query.bin --gt_file siftsmall/siftsmall_gt_35.bin --filter_label 35 -K 10 -L 10 20 30 40 50 100 --result_path siftsmall/filtered_search_results
build/apps/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/siftsmall_R20_L40_SR32_stitched_index --query_file siftsmall/siftsmall_query.bin --gt_file siftsmall/siftsmall_gt_35.bin --filter_label 35 -K 10 -L 10 20 30 40 50 100 --result_path siftsmall/stitched_search_results
```

 The output of both searches is listed below. The throughput (Queries/sec) as well as mean and 99.9 latency in microseconds for each `L` parameter provided. (Measured on a physical machine with a Intel(R) Xeon(R) W-2145 CPU and 64 GB RAM)
 ```
 Stitched Index
  Ls         QPS     Avg dist cmps  Mean Latency (mus)   99.9 Latency   Recall@10
=================================================================================
  10    31324.39             37.33              116.79         311.90       17.80
  20    91357.57             44.36              193.06        1042.30       17.90
  30    69314.48             49.89              258.09        1398.00       18.20
  40    61421.29             60.52              289.08        1515.00       18.60
  50    54203.48             70.27              294.26         685.10       19.40
 100    52904.45             79.00              336.26        1018.80       19.50

Filtered Index
 Ls         QPS     Avg dist cmps  Mean Latency (mus)   99.9 Latency   Recall@10
=================================================================================
  10    69671.84             21.48               45.25         146.20       11.60
  20   168577.20             38.94              100.54         547.90       18.20
  30   127129.41             52.95              126.83         768.40       19.70
  40   106349.04             62.38              167.23         899.10       20.90
  50    89952.33             70.95              189.12        1070.80       22.10
 100    56899.00            112.26              304.67         636.60       23.80
 ```
