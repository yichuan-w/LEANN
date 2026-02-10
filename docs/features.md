# ✨ Detailed Features

## 🔥 Core Features

- **🔄 Real-time Embeddings** - Eliminate heavy embedding storage with dynamic computation using optimized ZMQ servers and highly optimized search paradigm (overlapping and batching) with highly optimized embedding engine
- **🧠 AST-Aware Code Chunking** - Intelligent code chunking that preserves semantic boundaries (functions, classes, methods) for Python, Java, C#, and TypeScript files
- **📈 Scalable Architecture** - Handles millions of documents on consumer hardware; the larger your dataset, the more LEANN can save
- **🎯 Graph Pruning** - Advanced techniques to minimize the storage overhead of vector search to a limited footprint
- **🏗️ Pluggable Backends** - HNSW/FAISS (default), with optional DiskANN for large-scale deployments

## 🛠️ Technical Highlights
- **🔄 Recompute Mode** - Highest accuracy scenarios while eliminating vector storage overhead
- **⚡ Zero-copy Operations** - Minimize IPC overhead by transferring distances instead of embeddings
- **🚀 High-throughput Embedding Pipeline** - Optimized batched processing for maximum efficiency
- **🎯 Two-level Search** - Novel coarse-to-fine search overlap for accelerated query processing (optional)
- **💾 Memory-mapped Indices** - Fast startup with raw text mapping to reduce memory overhead
- **🚀 MLX Support** - Ultra-fast recompute/build with quantized embedding models, accelerating building and search ([minimal example](../examples/mlx_demo.py))

## 🎨 Developer Experience

- **Simple Python API** - Get started in minutes
- **Extensible backend system** - Easy to add new algorithms
- **Comprehensive examples** - From basic usage to production deployment
