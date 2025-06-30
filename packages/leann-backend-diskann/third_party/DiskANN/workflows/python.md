# `diskannpy`

We publish (sporadic) builds of `diskann` with python bindings to `pypi.org`, which you can install via `pip install diskannpy`.

#### Caveats
Native python modules with cffi need to be built for *every* version of Python and *every* OS and *every* native-integration-library.

This makes for a complicated build matrix that only `(ana)conda` is properly fit to solve. However, we do build wheels
for python 3.9-3.11, across linux, Windows, and macOS (x86_64). These versions are also built against `numpy` 1.25 - 
which makes for a hard runtime requirement that can be challenging to use if you are using older or newer versions of numpy.

There *are* instructions for building against other versions of numpy 
[documented in this issue response](https://github.com/microsoft/DiskANN/issues/544#issuecomment-2103437976) if you require a different build.

# Basic Usage

`diskannpy` provides access to both building and reading `DiskANN` indices. In all cases, the _lingua franca_ is numpy
ndarrays. Currently, the only supported dtypes are `np.float32`, `np.int8`, and `np.uint8`. 

`diskannpy` provides a number of helpful functions, like reading or writing `diskann` style vector binary files via the 
`vectors_to_file` and `vectors_from_file` functions. For a full suite of python functions and their documentation,
please be sure to read the latest documentation @ [https://microsoft.github.io/](https://microsoft.github.io/DiskANN/docs/python/latest/diskannpy.html).


## Scenarios
The following scenarios are supported via the `diskannpy` api.


### Commonalities
```python
my_dtype = np.float32  # or np.uint8 or np.int8 ONLY
my_set_of_vectors: np.typing.NDArray[my_dtype] = ... # your vectors come from somewhere - you need to bring these!
index_to_identifiers_map: np.typing.NDArray[str] = ... # your vectors likely have some kind of external identifier - 
# you need to keep track of the external identifier -> index relationship somehow
identifiers_to_index_map: dict[str, np.uint32|np.uint.64] = ... # your map of your external id to the `diskannpy` internal id
# diskannpy `query` responses will contain the _internal id only_, and if you don't have these maps you won't be able to 
# know what this relates to
```

### Build Disk Index
A disk index is a memory mapped, [vamana](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) 
index that heavily leans into the hardware speeds of modern NVMe based solid state storage. 

This means you can build performant ANN indices that overflow plausibly available system memory!

```python
import numpy as np
import diskannpy as dap

vecs = my_set_of_vectors / np.linalg.norm(my_set_of_vectors, axis=1)  # useful if your intention is to rank by a directionless 
# cosine angle distance

dap.build_disk_index(
    data=vecs,
    distance_metric="l2", # can also be cosine, especially if you don't normalize your vectors like above
    index_directory="/tmp/my_index",
    complexity=128,  # the larger this is, the more candidate points we consider when ranking
    graph_degree=64,  # the beauty of a vamana index is it's ability to shard and be able to transfer long distances across the grpah without navigating the whole thing. the larger this value is, the higher quality your results, but the longer it will take to build 
    search_memory_maximum=16.0, # a floating point number to represent how much memory in GB we want to optimize for @ query time
    build_memory_maximum=100.0, # a floating point number to represent how much memory in GB we are allocating for the index building process
    num_threads=0,  # 0 means use all available threads - but if you are in a shared environment you may need to restrict how greedy you are
    vector_dtype=my_dtype,  # we specified this in the Commonalities section above
    index_prefix="ann",  # ann is the default anyway. all files generated will have the prefix `ann_`, in the form of `f"{index_prefix}_"`
    pq_disk_bytes=0  # using product quantization of your vectors can still achieve excellent recall characteristics at a fraction of the latency, but we'll do it without PQ for now
)
```

### Search Disk Index

Now we want to search our disk index - using a completely different set of vectors that aren't necessarily guaranteed to
be in our index. We will call this set of vectors `q`, and it is *critical* that they are the same dtype and 
dimensionality as the disk index we have just built.

**Note**: If you manually normalized your indexed vectors prior to building the index, you will *also* need to normalize 
them prior to query!

#### Common index query setup

```python
index = dap.StaticDiskIndex(
    index_directory="/tmp/my_index",
    num_threads=0,
    num_nodes_to_cache=1_000_000,
    index_prefix="ann"  
)
```

#### Individual Vectors
```python
some_index: np.uint32 = ... # the index in our `q` array of points that we will be using to query on an individual basis
my_query_vector: np.typing.NDArray[my_dtype] = q[some_index] # make sure this is a 1-d array of the same dimensionality as your index!
# normalize if required by my_query_vector /= np.linalg.norm(my_query_vector)
internal_indices, distances = index.search(
    query=my_query_vector,
    k_neighbors=25,
    complexity=50,  # must be as big or bigger than `k_neighbors`
) 
```

#### Mapping to our External Ids
The internal IDs that diskann returns via query aren't necessarily directly useful to you, and the onus is on you
to figure out what they actually link to via your `index_to_identifiers_map` map.
```python
actual_identifiers = index_to_identifiers_map[internal_indices]  # using np fancy indexing (advanced indexing?) to map them all to ids you actually understand
```

#### Batch Vectors
```python
import multiprocessing

internal_indices, distances = index.batch_search(
    queries=q,
    k_neighbors=25,
    complexity=50,
    num_threads=multiprocessing.cpu_count(), # there's a current bug where this is not handling the value 0 properly
    beam_width=8 # beamwidth is the parameter that indicates our parallelism of individual searches, whereas num_threads 
    # indicates the number of threads *per* query item in the batch
)
# note that in batch_query form, our internal_indices and distances are 2d arrays
```

#### Mapping to our External Ids
Unlike the previous entry, I have yet to get the fancy awesome advanced indexing to work in one shot, we will have
to do this the not-numpy-paragon way.

```python
actual_neighbors = np.full(shape=internal_indices.shape, dtype=str, fill_value="")
for row in range(internal_indices.shape[0]):
    actual_neighbors[row] = index_to_identifiers_map[internal_indices[row]]
```

This is only scratching the surface of what `diskannpy` can offer. Please read the API documentation @ [https://microsoft.github.io/](https://microsoft.github.io/DiskANN/docs/python/latest/diskannpy.html) 
for more details.
