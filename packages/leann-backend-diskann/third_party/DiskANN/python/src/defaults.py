# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
# Parameter Defaults
These parameter defaults are re-exported from the C++ extension module, and used to keep the pythonic wrapper in sync with the C++.
"""
from ._diskannpy import defaults as _defaults

ALPHA = _defaults.ALPHA
""" 
Note that, as ALPHA is a `float32` (single precision float) in C++, when converted into Python it becomes a 
`float64` (double precision float). The actual value is 1.2f. The alpha parameter (>=1) is used to control the nature 
and number of points that are added to the graph. A higher alpha value (e.g., 1.4) will result in fewer hops (and IOs) 
to convergence, but probably more distance comparisons compared to a lower alpha value.
"""
NUM_THREADS = _defaults.NUM_THREADS
""" Number of threads to use. `0` will use all available detected logical processors """
MAX_OCCLUSION_SIZE = _defaults.MAX_OCCLUSION_SIZE
""" 
The maximum number of points that can be occluded by a single point. This is used to  prevent a single point from 
dominating the graph structure. If a point has more than `max_occlusion_size` neighbors closer to it than the current 
point, it will not be added to the graph. This is a tradeoff between index build time and search quality. 
"""
FILTER_COMPLEXITY = _defaults.FILTER_COMPLEXITY
""" 
Complexity (a.k.a. `L`) references the size of the list we store candidate approximate neighbors in while doing a 
filtered search. This value must be larger than `k_neighbors`, and larger values tend toward higher recall in the 
resultant ANN search at the cost of more time. 
"""
NUM_FROZEN_POINTS_STATIC = _defaults.NUM_FROZEN_POINTS_STATIC
""" Number of points frozen by default in a StaticMemoryIndex """
NUM_FROZEN_POINTS_DYNAMIC = _defaults.NUM_FROZEN_POINTS_DYNAMIC
""" Number of points frozen by default in a DynamicMemoryIndex """
SATURATE_GRAPH = _defaults.SATURATE_GRAPH
""" Whether to saturate the graph or not. Default is `True` """
GRAPH_DEGREE = _defaults.GRAPH_DEGREE
""" 
Graph degree (a.k.a. `R`) is the maximum degree allowed for a node in the index's graph structure. This degree will be 
pruned throughout the course of the index build, but it will never grow beyond this value. Higher R values require 
longer index build times, but may result in an index showing excellent recall and latency characteristics. 
"""
COMPLEXITY = _defaults.COMPLEXITY
""" 
Complexity (a.k.a `L`) references the size of the list we store candidate approximate neighbors in while doing build
or search tasks. It's used during index build as part of the index optimization processes. It's used in index search 
classes both to help mitigate poor latencies during cold start, as well as on subsequent queries to conduct the search. 
Large values will likely increase latency but also may improve recall, and tuning these values for your particular 
index is certainly a reasonable choice.
"""
PQ_DISK_BYTES = _defaults.PQ_DISK_BYTES
""" 
Use `0` to store uncompressed data on SSD. This allows the index to asymptote to 100% recall. If your vectors are 
too large to store in SSD, this parameter provides the option to compress the vectors using PQ for storing on SSD. 
This will trade off recall. You would also want this to be greater than the number of bytes used for the PQ 
compressed data stored in-memory. Default is `0`. 
"""
USE_PQ_BUILD = _defaults.USE_PQ_BUILD
"""
 Whether to use product quantization in the index building process. Product quantization is an approximation 
technique that can vastly speed up vector computations and comparisons in a spatial neighborhood, but it is still an 
approximation technique. It should be preferred when index creation times take longer than you can afford for your 
use case.
"""
NUM_PQ_BYTES = _defaults.NUM_PQ_BYTES
""" 
The number of product quantization bytes to use. More bytes requires more resources in both memory and time, but is 
like to result in better approximations. 
"""
USE_OPQ = _defaults.USE_OPQ
""" Whether to use Optimized Product Quantization or not. """
