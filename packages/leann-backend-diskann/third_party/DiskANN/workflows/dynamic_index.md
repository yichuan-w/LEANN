<!-- Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license. -->

**Usage for dynamic indices**
================================

A "dynamic" index refers to an index which supports insertion of new points into a (possibly previously built) index as well as deletions of points.
While eager deletes can be supported by DiskANN, `lazy_deletes` are the preferred method. 
A sequence of lazy deletions must be followed by an invocation of the `consolidate_deletes` method that frees up slots in the index and edits the graph to maintain good recall.


The program `apps/test_insert_deletes_consolidate` demonstrates this functionality. It allows the user to specify which points from the data file will be used
to initially build the index, which points will be deleted from the index, and which points will be inserted into the index.
Insertions, searches and lazy deletions can be performed concurrently.
Conslolidation of lazy deletes can be performed synchnronously or concurrently with insertions and deletions.
When modifying the index sequentially, the user has the ability to take *snapshots*--
that is, save the index to memory for every *m* insertions or deletions instead of only at the end of the build.

The program `apps/test_streaming_scenario` simulates a scenario where the index actively maintains a sliding window of active points from a larger dataset.
The program starts with an index build over the first `active_window` set of points from a data file. 
The program then simultaneously inserts newer points drawn from the file and deletes older points from the index
in chunks of `consolidate_interval` points so that the number of active points in the index is approximately `active_window`.
It terminates when the end of data file is reached, and the final index has `active_window + consolidate_interval` number of points.

The index also supports filters on steaming index, you can use `insert_point` function overloads to either insert points as before or insert points with labels.
Additional options are added to support this in `apps/test_streaming_scenario` and `apps/test_streaming_scenario` please refer to program arguments for more details.

---
> Note
* The index does not support mixed points, that is, either all points do not have labels or all points have labels. 
* You can search the built filter index (one built with filters) without filters as well.
 
> WARNING: Deleting points in case of filtered build may cause the quality of Index to degrade and affect recall.
---

`apps/test_insert_deletes_consolidate` to try inserting, lazy deletes and consolidate_delete 
---------------------------------------------------------------------------------------------

The arguments are as follows:

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips).
3. **--data_file**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
4. **--index_path_prefix**: The constructed index components will be saved to this path prefix.
5. **-R (--max_degree)** (default is 64): the degree of the graph index, typically between 32 and 150. Larger R will result in larger indices and longer indexing times, but might yield better search quality. 
6. **-L (--Lbuild)** (default is 100): the size of search list we maintain during index building. Typical values are between 75 to 400. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Ensure that value of L is at least that of R value unless you need to build indices really quickly and can somewhat compromise on quality. 
7. **--alpha** (default is 1.2): A float value between 1.0 and 1.5 which determines the diameter of the graph, which will be approximately *log n* to the base alpha. Typical values are between 1 to 1.5. 1 will yield the sparsest graph, 1.5 will yield denser graphs. 
8. **T (--num_threads)** (default is to get_omp_num_procs()): number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).
9. **--points_to_skip**: number of points to skip from the beginning of the data file. 
10. **--max_points_to_insert**: the maximum size of the index. 
11. **--beginning_index_size**: how many points to build the initial index with. The number of points inserted dynamically will be max_points_to_insert - beginning_index_size. 
12. **--points_per_checkpoint**: when inserting and deleting sequentially, each update is handled in points_per_checkpoint batches. When updating concurrently, insertions are handled in points_per_checkpoint batches but deletions are always processed in a single batch.
13. **--checkpoints_per_snapshot**: when inserting and deleting sequentially, the graph is saved to memory every checkpoints_per_snapshot checkpoints. This is not currently supported for concurrent updates.
14. **--points_to_delete_from_beginning**: how many points to delete from the index, starting in order of insertion. If deletions are concurrent with insertions, points_to_delete_from_beginning cannot be larger than beginning_index_size.
15. **--start_point_norm**: Set the starting node to a random point on a sphere of this radius. A reasonable choice is to set this to the average norm of the data set. Use when starting an index with zero points. 
16. **--do_concurrent** (default false): whether to perform conslidate_deletes and other updates concurrently or sequentially. If concurrent is specified, half the threads are used for insertions and half the threads are used for processing deletes. Note that insertions are performed before deletions if this flag is set to false, so in this case is possible to delete more than beginning_index_size points.

`apps/test_streaming_scenario` to try inserting, lazy deletes and consolidate_delete 
---------------------------------------------------------------------------------------------

The arguments are as follows:

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips).
3. **--data_file**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
4. **--index_path_prefix**: The constructed index components will be saved to this path prefix.
5. **-R (--max_degree)** (default is 64): the degree of the graph index, typically between 32 and 150. Larger R will result in larger indices and longer indexing times, but might yield better search quality. 
6. **-L (--Lbuild)** (default is 100): the size of search list we maintain during index building. Typical values are between 75 to 400. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Ensure that value of L is at least that of R value unless you need to build indices really quickly and can somewhat compromise on quality. 
7. **--alpha** (default is 1.2): A float value between 1.0 and 1.5 which determines the diameter of the graph, which will be approximately *log n* to the base alpha. Typical values are between 1 to 1.5. 1 will yield the sparsest graph, 1.5 will yield denser graphs. 
8. **--insert_threads**: number of threads used for inserting points in to the index. 
9. **--consolidate_threads**: number of threads used for consolidating deletes to the index. 
10. **--max_points_to_insert**: Maximum number of points from the data file to insert in to the index.
11. **--active_window**: Approximate number of points in the index at any point.
12. **--consolidate_interval**: Granularity at which insert and delete functions are called.
13. **--start_point_norm**: Set the starting node to a random point on a sphere of this radius.  A reasonable choice is to set this to the average norm of the data stream.

** To build with filters add these optional parameters.

14. **--label_file**: Filter data for each point, in `.txt` format. Line `i` of the file consists of a comma-separated list of labels corresponding to point `i` in the file passed via `--data_file`.
15. **--FilteredLbuild**: If building a filtered index, we maintain a separate search list from the one provided by `--Lbuild/-L`.
16. **--num_start_points**: number of frozen points in this case should be more then number of unique labels. 
17. **--universal_label**: Optionally, the label data may contain a special "universal" label. A point with the universal label can be matched against a query with any label. Note that if a point has the universal label, then the filter data must only have the universal label on the line corresponding.
18. **--label_type**: Optionally, type of label to be use its either uint or short, defaulted to `uint`.

To search the generated index, use the `apps/search_memory_index` program:
---------------------------------------------------------------------------


The arguments are as follows:

1. **data_type**: The type of dataset you built the index on. float(32 bit), signed int8 and unsigned uint8 are supported. Use the same data type as in arg (1) above used in building the index.
2. **dist_fn**: There are two distance functions supported: l2 and mips. There is an additional *fast_l2* implementation that could provide faster results for small (about a million-sized) indices. Use the same distance as in arg (2) above used in building the index.
3. **memory_index_path**: index built above in argument (4).
4. **T**: The number of threads used for searching. Threads run in parallel and one thread handles one query at a time. More threads will result in higher aggregate query throughput, but may lead to higher per-query latency, especially if the DRAM bandwidth is a bottleneck. So find the balance depending on throughput and latency required for your application.
5. **query_bin**: The queries to be searched on in same binary file format as the data file (ii) above. The query file must be the same type as in argument (1).
6. **truthset.bin**: The ground truth file for the queries in arg (7) and data file used in index construction.  The binary file must start with *n*, the number of queries (4 bytes), followed by *d*, the number of ground truth elements per query (4 bytes), followed by `n*d` entries per query representing the d closest IDs per query in integer format,  followed by `n*d` entries representing the corresponding distances (float). Total file size is `8 + 4*n*d + 4*n*d` bytes. The groundtruth file, if not available, can be calculated using the program `apps/utils/compute_groundtruth`. Use "null" if you do not have this file and if you do not want to compute recall.
7. **K**: search for *K* neighbors and measure *K*-recall@*K*, meaning the intersection between the retrieved top-*K* nearest neighbors and ground truth *K* nearest neighbors.
8. **result_output_prefix**: search results will be stored in files, one per L value (see next arg), with specified prefix, in binary format.
9. **-L (--search_list)**: A list of search_list sizes to perform search with. Larger parameters will result in slower latencies, but higher accuracies. Must be at least the value of *K* in (7).
10. **--dynamic** (default false): whether the index being searched is dynamic or not.
11. **--tags** (default false): whether to search with tags. This should be used if point *i* in the ground truth file does not correspond the point in the *i*th position in the loaded index.

** to search with filters add these

12. **--filter_label**: Filter for each query. For each query, a search is performed with this filter.

Example with BIGANN:
--------------------

This example demonstrates the use of the commands above on a 100K slice of the [BIGANN dataset](http://corpus-texmex.irisa.fr/) with 128 dimensional SIFT descriptors applied to images. 

Download the base and query set and convert the data to binary format
```bash
mkdir -p DiskANN/build/data && cd DiskANN/build/data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
cd ..
./apps/utils/fvecs_to_bin float data/sift/sift_learn.fvecs data/sift/sift_learn.fbin
./apps/utils/fvecs_to_bin float data/sift/sift_query.fvecs data/sift/sift_query.fbin
```

The example below tests the following scenario: using a file with 100000 points, the index is incrementally constructed point by point. After the first 50000 ponts are inserted, another concurrent job deletes the first 25000 points from the index and consolidates the index (edit the graph and cleans up resources). At the same time an additional 25000 points (i.e. points 50001 to 75000) are concurrently inserted into the index. Note that the index should be built **before** calculating the ground truth, since the memory index returns the slice of the sift100K dataset that was used to build the final graph (that is, points 25001-75000 in the original index).
```bash
type='float'
data='data/sift/sift_learn.fbin'
query='data/sift/sift_query.fbin'
index_prefix='data/sift/index'
result='data/sift/res'
deletes=25000
inserts=75000
deletes_after=50000
pts_per_checkpoint=10000
begin=0
thr=64
index=${index_prefix}.after-concurrent-delete-del${deletes}-${inserts}
gt_file=data/sift/gt100_learn-conc-${deletes}-${inserts}

 ~/DiskANN/build/apps/test_insert_deletes_consolidate  --data_type ${type} --dist_fn l2 --data_path ${data}  --index_path_prefix ${index_prefix} -R 64 -L 300 --alpha 1.2 -T ${thr} --points_to_skip 0 --max_points_to_insert ${inserts} --beginning_index_size ${begin} --points_per_checkpoint ${pts_per_checkpoint} --checkpoints_per_snapshot 0 --points_to_delete_from_beginning ${deletes} --start_deletes_after ${deletes_after} --do_concurrent true;

 ~/DiskANN/build/apps/utils/compute_groundtruth --data_type ${type} --dist_fn l2 --base_file ${index}.data  --query_file ${query}  --K 100 --gt_file ${gt_file} --tags_file  ${index}.tags

~/DiskANN/build/apps/search_memory_index  --data_type ${type} --dist_fn l2 --index_path_prefix ${index} --result_path ${result} --query_file ${query}  --gt_file ${gt_file}  -K 10 -L 20 40 60 80 100 -T ${thr} --dynamic true --tags 1
 ```

 The example below tests the following scenario: using a file with 100000 points, insert 10000 points at a time. After the first 40000
are inserted, start deleting the first 10000 points while inserting points 40000--50000.  Then delete points 10000--20000 while inserting
points 50000--60000 and so until the index is left with points 60000-100000.


Generate labels for filtered build like this. Generating 50 unique labels zipf's distributed for 100K point dataset.
```
~/DiskANN/build/apps/utils/generate_synthetic_labels  --num_labels 50 --num_points 100000  --output_file data/zipf_labels_50_100K.txt --distribution_type zipf
```

```bash
type='float'
data='data/sift/sift_learn.fbin'
query='data/sift/sift_query.fbin'
index_prefix='data/sift/idx_learn_str'
result='data/sift/res'
ins_thr=16
cons_thr=16
inserts=100000
active=20000
cons_int=10000
index=${index_prefix}.after-streaming-act${active}-cons${cons_int}-max${inserts}
gt=data/sift/gt100_learn-act${active}-cons${cons_int}-max${inserts}
filter_label=1

## filter options
universal_label = '0'
label_file = 'data/zipf_labels_50_100K.txt'
num_start_points = 50
gt_filtered= data/sift/gt100_learn-act${active}-cons${cons_int}-max${inserts}_wlabel_${filter_label}


# Without Filters (build and search)
./apps/test_streaming_scenario  --data_type ${type} --dist_fn l2 --data_path ${data}  --index_path_prefix ${index_prefix} -R 64 -L 600 --alpha 1.2 --insert_threads ${ins_thr} --consolidate_threads ${cons_thr}  --max_points_to_insert ${inserts}  --active_window ${active} --consolidate_interval ${cons_int} --start_point_norm 508;
./apps/utils/compute_groundtruth --data_type ${type} --dist_fn l2 --base_file ${index}.data  --query_file ${query}  --K 100 --gt_file ${gt} --tags_file  ${index}.tags
./apps/search_memory_index  --data_type ${type} --dist_fn l2 --index_path_prefix ${index} --result_path ${result} --query_file ${query}  --gt_file ${gt}  -K 10 -L 20 40 60 80 100 -T 64 --dynamic true --tags 1

# With filters (build and search)

./apps/test_streaming_scenario  --data_type ${type} --num_start_points ${num_start_points} --label_file ${label_file} --universal_label {universal_label} --dist_fn l2 --data_path ${data}  --index_path_prefix ${index_prefix} -R 64 -L 600 --alpha 1.2 --insert_threads ${ins_thr} --consolidate_threads ${cons_thr}  --max_points_to_insert ${inserts}  --active_window ${active} --consolidate_interval ${cons_int} --start_point_norm 508;
./apps/utils/compute_groundtruth_for_filters --data_type ${type} --dist_fn l2 --base_file ${index}.data  --query_file ${query}  --K 100 --gt_file ${gt_filtered} --label_file  ${label_file} --universal_label {universal_label} --filter_label {filter_label}
./apps/search_memory_index  --data_type ${type} --filter_label {filter_label} --dist_fn l2 --index_path_prefix ${index} --result_path ${result} --query_file ${query}  --gt_file ${gt_filtered}  -K 10 -L 20 40 60 80 100 -T 64 --dynamic true --tags 1
```
