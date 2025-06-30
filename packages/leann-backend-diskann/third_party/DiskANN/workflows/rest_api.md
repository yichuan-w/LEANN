<!-- Copyright (c) Microsoft Corporation. All rights reserved.
   Licensed under the MIT license. -->
**REST service set up for serving DiskANN indices and query interface**
=======================================================================

Install dependencies on Ubuntu and compile
------------------------------------------
In addition to the common dependencies in the [README](/README.md), install [Microsoft C++ REST SDK](https://github.com/Microsoft/cpprestsdk).

```bash
sudo apt install libcpprest-dev
mkdir -p build && cd build
cmake -DRESTAPI=True -DCMAKE_BUILD_TYPE=Release ..
make -j
```

Starting an index hosting service
---------------------------------
Follow the instructions for [building an in-memory DiskANN index](/workflows/in_memory_index.md) or [building an SSD DiskANN index](/workflows/SSD_index.md).  Then start a service bound at the appropriate IP:port. For querying from the local machine, you may want to use `http://127.0.0.1:port`. For serving queries originating from remote machines, you may want to use `http://0.0.0.0:port`.

```bash
# To start serving an in-memory index
./apps/restapi/inmem_server --address <http://ip_addr:port> --data_type <float/int8/uint8> --data_file <data_file> --index_path_prefix <index_file> --num_threads <number of threads> --l_search <Value for L> --tags_file [tags_file]

# To start serving an SSD-based index.
./apps/restapi/ssd_server --address <http://ip_addr:port> --data_type <float/int8/uint8> --index_path_prefix <index_file_prefix> --num_nodes_to_cache <num_nodes_to_cache> --num_threads <num_threads> --tags_file [tags_file]
```
The `data_type` and the `data_file` should be the same as those used in the construction of the index. The server returns the ids and distances of the closests vector in the index to the query. The ids are implicitly defined by the order of the vector in the data file. If you wish to assign a different numbering or GUID or URL to the vectors in the index, use the optional `tags_file`. This should be a file which lists a "tag" string for each vector in the index. The file should contain one string per line. The string on the line `n` is considered the tag corresponding to the vector `n` in the index (in the implicit order defined in the `data_file`).

For an SSD-based index, specify the number of nodes to cache in-memory to make queries faster. For large indices with over 100 million vectors, a typical value for `num_nodes_to_cache` could be 500000. Increase or decrease based on DRAM footprint desired.

For an SSD-based index, also specify the number of threads used for search by setting the `num_threads` parameter.

You can also query multiple SSD based indices using the following command by listing the prefix of each index in a file (one prefix per line) and passing it through the `index_prefix_paths` parameter to the following command. 
```bash
multiple_ssdserver --address <ip_addr:port> --data_type <float/int8/uint8> --index_prefix_paths <index_prefix_paths> --num_nodes_to_cache <num_nodes_to_cache> --num_threads <num_threads> --tags_file [tags_file]
```
The service searches each of the indices and aggregate the results based on distances to find the closest neighbors across all indices.

Querying the service
--------------------
Issue a json query with the following fields
- "k" : The number of nearest neighbors needed
- "query" : The query vector with a listing of co-ordinates.
- "query_id" : An id to track the query. Use a unique number to keep track of queries, or "0" if you do not want to keep track.
- "Ls" : query complexity. Higher Ls takes more milliseconds to process but offers higher recall. Default to 256 if you don't want to tune this. 

**Post a json query using python**

```python
import requests
jsonquery = {"Ls": 256,
         "query_id": 1234,
         "query": [0.00407, 0.01534, 0.02498, ...],
         "k": 10}

response = requests.post('http://ip_addr:port', json=jsonquery)
print(response.text)
```

The response might look like the following. The partition array indicates the ID of index from which the result was found in the case of a multi-index set up. For a single index set up, the response would not contain the information on partitions. The response may or may not contain `tags` based on whether the server was started with a `tags_file`. 
```json
{"distances":[1.6947,1.6954,1.6972,1.6985,1.6991,1.7003,1.7008,1.7014,1.7021,1.7039],"indices":[8976853,8221762,30909336,13100282,30514543,11537860,7133262,34074869,50512601,17983301],"k":10,"partition":[20,7,20,20,6,6,11,6,6,20],"query_id":1234,"tags":["https://xyz1", "https://xyz2", "https://xyz3", "https://xyz4", "https://xyz5", "https://xyz6", "https://xyz7", "https://xyz8", "https://xyz9", "https://xyz10"],"time_taken_in_us":3245}
```

**Command line interface to issue multiple queries from a file**

To issue `num_queries` queries from `query_file`, run the following command
```bash
client ip_addr:port data_type<float/int8/uint8> query_file num_queries Ls"
```

