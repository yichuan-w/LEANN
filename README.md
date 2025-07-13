# 🚀 LEANN: A Low-Storage Vector Index

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Platform">
</p>

<p align="center">
  <strong>⚡ Storage Saving RAG sytem on Consumer Device</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-paper">Paper</a>
</p>

---

## 🌟 What is Leann?

**Leann** revolutionizes Retrieval-Augmented Generation (RAG) by eliminating the storage bottleneck of traditional vector databases. Instead of pre-computing and storing billions of embeddings, Leann dynamically computes embeddings at query time using optimized graph-based search algorithms.

### 🎯 Why Leann?

Traditional RAG systems face a fundamental trade-off:

- **💾 Storage**: Storing embeddings for millions of documents requires massive disk space
- **🔄 Memory overhead**: The indexes LlamaIndex uses usually face high memory overhead (e.g., in-memory vector databases)
- **💰 Cost**: Vector databases are expensive to scale

**Leann revolutionizes this with Graph-based recomputation and cutting-edge system optimizations:**

- ✅ **Zero embedding storage** - Only graph structure is persisted, reducing storage by 94-97%
- ✅ **Real-time computation** - Embeddings computed on-demand with low latency
- ✅ **Memory efficient** - Runs on consumer hardware with theoretical zero memory overhead
- ✅ **Graph-based optimization** - Advanced pruning techniques for efficient search while keeping low storage cost, with batching and overlapping strategies using low-precision search to optimize latency
- ✅ **Pluggable backends** - Support for DiskANN, HNSW, and other ANN algorithms (welcome contributions!)

## 🚀 Quick Start

### Installation

```bash
git clone git@github.com:yichuan520030910320/LEANN-RAG.git leann
cd leann
git submodule update --init --recursive
```

**macOS:**
```bash
brew install llvm libomp boost protobuf
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
uv sync
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libomp-dev libboost-all-dev protobuf-compiler libabsl-dev libmkl-full-dev libaio-dev
uv sync
```

### 30-Second Example

```python
from leann.api import LeannBuilder, LeannSearcher

# 1. Build index (no embeddings stored!)
builder = LeannBuilder(backend_name="diskann")
builder.add_text("Python is a powerful programming language")
builder.add_text("Machine learning transforms industries")  
builder.add_text("Neural networks process complex data")
builder.build_index("knowledge.leann")

# 2. Search with real-time embeddings
searcher = LeannSearcher("knowledge.leann")
results = searcher.search("programming languages", top_k=2)

for result in results:
    print(f"Score: {result['score']:.3f} - {result['text']}")
```

### Run the Demo

```bash
uv run examples/document_search.py
```

or you want to use python 

```bash
source .venv/bin/activate
python ./examples/main_cli_example.py
```
**PDF RAG Demo (using LlamaIndex for document parsing and Leann for indexing/search)**

This demo showcases how to build a RAG system for PDF documents using Leann.

1. Place your PDF files (and other supported formats like .docx, .pptx, .xlsx) into the `examples/data/` directory.
2. Ensure you have an `OPENAI_API_KEY` set in your environment variables or in a `.env` file for the LLM to function.

```bash
uv run examples/main_cli_example.py
```

### Regenerating Protobuf Files

If you modify any `.proto` files (such as `embedding.proto`), or if you see errors about protobuf version mismatch, **regenerate the C++ protobuf files** to match your installed version:

```bash
cd packages/leann-backend-diskann
protoc --cpp_out=third_party/DiskANN/include --proto_path=third_party embedding.proto
protoc --cpp_out=third_party/DiskANN/src --proto_path=third_party embedding.proto
```

This ensures the generated files are compatible with your system's protobuf library.

## ✨ Features

### 🔥 Core Features

- **📊 Multiple Distance Functions**: L2, Cosine, MIPS (Maximum Inner Product Search)
- **🏗️ Pluggable Backends**: DiskANN, HNSW/FAISS with unified API
- **🔄 Real-time Embeddings**: Dynamic computation using optimized ZMQ servers
- **📈 Scalable Architecture**: Handles millions of documents on consumer hardware
- **🎯 Graph Pruning**: Advanced techniques for memory-efficient search

### 🛠️ Technical Highlights

- **Zero-copy operations** for maximum performance
- **SIMD-optimized** distance computations (AVX2/AVX512)
- **Async embedding pipeline** with batched processing
- **Memory-mapped indices** for fast startup
- **Recompute mode** for highest accuracy scenarios

### 🎨 Developer Experience

- **Simple Python API** - Get started in minutes
- **Extensible backend system** - Easy to add new algorithms
- **Comprehensive examples** - From basic usage to production deployment
- **Rich debugging tools** - Built-in performance profiling

## 📊 Benchmarks

### Memory Usage Comparison

| System                | 1M Documents     | 10M Documents    | 100M Documents   |
| --------------------- | ---------------- | ---------------- | ---------------- |
| Traditional Vector DB | 3.1 GB           | 31 GB            | 310 GB           |
| **Leann**       | **180 MB** | **1.2 GB** | **8.4 GB** |
| **Reduction**   | **94.2%**  | **96.1%**  | **97.3%**  |

### Query Performance

| Backend             | Index Size | Query Time | Recall@10 |
| ------------------- | ---------- | ---------- | --------- |
| DiskANN             | 1M docs    | 12ms       | 0.95      |
| DiskANN + Recompute | 1M docs    | 145ms      | 0.98      |
| HNSW                | 1M docs    | 8ms        | 0.93      |

*Benchmarks run on AMD Ryzen 7 with 32GB RAM*

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Query Text    │───▶│  Embedding       │───▶│   Graph-based   │
│                 │    │  Computation     │    │     Search      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────┐         ┌──────────────┐
                       │ ZMQ Server   │         │ Pruned Graph │
                       │ (Cached)     │         │ Index        │
                       └──────────────┘         └──────────────┘
```

### Key Components

1. **🧠 Embedding Engine**: Real-time transformer inference with caching
2. **📊 Graph Index**: Memory-efficient navigation structures
3. **🔄 Search Coordinator**: Orchestrates embedding + graph search
4. **⚡ Backend Adapters**: Pluggable algorithm implementations

## 🎓 Supported Models & Backends

### 🤖 Embedding Models

- **sentence-transformers/all-mpnet-base-v2** (default)
- **sentence-transformers/all-MiniLM-L6-v2** (lightweight)
- Any HuggingFace sentence-transformer model
- Custom model support via API

### 🔧 Search Backends

- **DiskANN**: Microsoft's billion-scale ANN algorithm
- **HNSW**: Hierarchical Navigable Small World graphs
- **Coming soon**: ScaNN, Faiss-IVF, NSG

### 📏 Distance Functions

- **L2**: Euclidean distance for precise similarity
- **Cosine**: Angular similarity for normalized vectors
- **MIPS**: Maximum Inner Product Search for recommendation systems

## 🔬 Paper

If you find Leann useful, please cite:

**[LEANN: A Low-Storage Vector Index](https://arxiv.org/abs/2506.08276)**

```bibtex
@misc{wang2025leannlowstoragevectorindex,
      title={LEANN: A Low-Storage Vector Index}, 
      author={Yichuan Wang and Shu Liu and Zhifei Li and Yongji Wu and Ziming Mao and Yilong Zhao and Xiao Yan and Zhiying Xu and Yang Zhou and Ion Stoica and Sewon Min and Matei Zaharia and Joseph E. Gonzalez},
      year={2025},
      eprint={2506.08276},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2506.08276}, 
}
```

## 🌍 Use Cases

### 💼 Enterprise RAG

```python
# Handle millions of documents with limited resources
builder = LeannBuilder(
    backend_name="diskann",
    distance_metric="cosine",
    graph_degree=64,
    memory_budget="4GB"
)
```

### 🔬 Research & Experimentation

```python
# Quick prototyping with different algorithms
for backend in ["diskann", "hnsw"]:
    searcher = LeannSearcher(index_path, backend=backend)
    evaluate_recall(searcher, queries, ground_truth)
```

### 🚀 Real-time Applications

```python
# Sub-second response times
chat = LeannChat("knowledge.leann")
response = chat.ask("What is quantum computing?")
# Returns in <100ms with recompute mode
```

## 🤝 Contributing

We welcome contributions! Leann is built by the community, for the community.

### Ways to Contribute

- 🐛 **Bug Reports**: Found an issue? Let us know!
- 💡 **Feature Requests**: Have an idea? We'd love to hear it!
- 🔧 **Code Contributions**: PRs welcome for all skill levels
- 📖 **Documentation**: Help make Leann more accessible
- 🧪 **Benchmarks**: Share your performance results

### Development Setup

```bash
git clone git@github.com:yichuan520030910320/LEANN-RAG.git leann
cd leann
git submodule update --init --recursive
uv sync --dev
uv run pytest tests/
```

### Quick Tests

```bash
# Sanity check all distance functions
uv run python tests/sanity_checks/test_distance_functions.py

# Verify L2 implementation
uv run python tests/sanity_checks/test_l2_verification.py
```

## ❓ FAQ

### Common Issues

#### NCCL Topology Error

**Problem**: You encounter `ncclTopoComputePaths` error during document processing:

```
ncclTopoComputePaths (system=<optimized out>, comm=comm@entry=0x5555a82fa3c0) at graph/paths.cc:688
```

**Solution**: Set these environment variables before running your script:

```bash
export NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
export NCCL_IB_DISABLE=1
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=ens5
```

## 📈 Roadmap

### 🎯 Q2 2025

- [X] DiskANN backend with MIPS/L2/Cosine support
- [X] HNSW backend integration
- [X] Real-time embedding pipeline
- [X] Memory-efficient graph pruning

### 🚀 Q3 2025


- [ ] Advanced caching strategies
- [ ] GPU-accelerated embedding computation
- [ ] Add sleep-time-compute and summarize agent! to summarilze the file on computer!
- [ ] Add OpenAI recompute API

### 🌟 Q4 2025

- [ ] Integration with LangChain/LlamaIndex
- [ ] Visual similarity search
- [ ] Query rewrtiting, rerank and expansion

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Microsoft Research** for the DiskANN algorithm
- **Meta AI** for FAISS and optimization insights
- **HuggingFace** for the transformer ecosystem
- **Our amazing contributors** who make this possible

---

<p align="center">
  <strong>⭐ Star us on GitHub if Leann is useful for your research or applications!</strong>
</p>

<p align="center">
  Made with ❤️ by the Leann team
</p>

