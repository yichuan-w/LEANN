<h1 align="center">🚀 LEANN: A Low-Storage Vector Index</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
<img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey" alt="Platform">
</p>

<p align="center">
  <strong>💾 Extreme Storage Saving • 🔒 100% Private • 📚 RAG Everything • ⚡ Easy & Accurate</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="https://arxiv.org/abs/2506.08276" target="_blank">Paper</a>
</p>

---

## 🌟 What is LEANN-RAG?

**LEANN-RAG** is a lightweight, locally deployable **Retrieval-Augmented Generation (RAG)** engine designed for personal devices. It combines **compact storage**, **clean usability**, and **privacy-by-design**, making it easy to build personalized retrieval systems over your own data — emails, notes, documents, chats, or anything else.

Unlike traditional vector databases that rely on massive embedding storage, LEANN reduces storage needs dramatically by using **graph-based recomputation** and **pruned HNSW search**, while maintaining responsive and reliable performance — all without sending any data to the cloud.

---

## 🔥 Key Highlights

### 💾 1. Extreme Storage Efficiency  
LEANN reduces storage usage by **up to 97%** compared to conventional vector DBs (e.g., FAISS), by storing only pruned graph structures and computing embeddings at query time.  
> For example: 60M chunks can be indexed in just **6GB**, compared to **200GB+** with dense storage.

### 🔒 2. Fully Private, Cloud-Free  
LEANN runs entirely locally. No cloud services, no API keys, and no risk of leaking sensitive data.  
> Converse with your own files **without compromising privacy**.

### 🧠 3. RAG Everything  
Build truly personalized assistants by querying over **your own** chat logs, email archives, browser history, or agent memory.  
> LEANN makes it easy to integrate personal context into RAG workflows.

### ⚡ 4. Easy, Accurate, and Fast  
LEANN is designed to be **easy to install**, with a **clean API** and minimal setup. It runs efficiently on consumer hardware without sacrificing retrieval accuracy.  
> One command to install, one click to run.

---

## 🚀 Why Choose LEANN?

Traditional RAG systems often require trade-offs between storage, privacy, and usability. **LEANN-RAG aims to simplify the stack** with a more practical design:

- ✅ **No embedding storage** — compute on demand, save disk space  
- ✅ **Low memory footprint** — lightweight and hardware-friendly  
- ✅ **Privacy-first** — 100% local, no network dependency  
- ✅ **Simple to use** — developer-friendly API and seamless setup  

> 📄 For more details, see our [academic paper](https://arxiv.org/abs/2506.08276)
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

**Ollama Setup (Optional for Local LLM):**

*macOS:*
```bash
# Install Ollama
brew install ollama

# Pull a lightweight model (recommended for consumer hardware)
ollama pull llama3.2:1b

# For better performance but higher memory usage
ollama pull llama3.2:3b
```

*Linux:*
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service manually
ollama serve &

# Pull a lightweight model (recommended for consumer hardware)
ollama pull llama3.2:1b

# For better performance but higher memory usage
ollama pull llama3.2:3b
```

**Note:** For Hugging Face models >1B parameters, you may encounter OOM errors on consumer hardware. Consider using smaller models like Qwen3-0.6B or switch to Ollama for better memory management.

### 30-Second Example
Try it out in [**demo.ipynb**](demo.ipynb)

```python
from leann.api import LeannBuilder, LeannSearcher
# 1. Build index (no embeddings stored!)
builder = LeannBuilder(backend_name="hnsw")
builder.add_text("C# is a powerful programming language")
builder.add_text("Python is a powerful programming language")
builder.add_text("Machine learning transforms industries")  
builder.add_text("Neural networks process complex data")
builder.add_text("Leann is a great storage saving engine for RAG on your macbook")
builder.build_index("knowledge.leann")
# 2. Search with real-time embeddings
searcher = LeannSearcher("knowledge.leann")
results = searcher.search("C++ programming languages", top_k=2, recompute_beighbor_embeddings=True)
print(results)
```

### Run the Demo (support .pdf,.txt,.docx, .pptx, .csv, .md etc)

```bash
uv run ./examples/main_cli_example.py
```

or you want to use python 

```bash
source .venv/bin/activate
python ./examples/main_cli_example.py
```
**PDF RAG Demo (using LlamaIndex for document parsing and Leann for indexing/search)**

This demo showcases how to build a RAG system for PDF/md documents using Leann.

1. Place your PDF files (and other supported formats like .docx, .pptx, .xlsx) into the `examples/data/` directory.
2. Ensure you have an `OPENAI_API_KEY` set in your environment variables or in a `.env` file for the LLM to function.



## ✨ Features

### 🔥 Core Features

- **🔄 Real-time Embeddings** - Eliminate heavy embedding storage with dynamic computation using optimized ZMQ servers and highly optimized search paradigm (overlapping and batching) with highly optimized embedding engine
- **📈 Scalable Architecture** - Handles millions of documents on consumer hardware; the larger your dataset, the more LEANN can save
- **🎯 Graph Pruning** - Advanced techniques to minimize the storage overhead of vector search to a limited footprint
- **🏗️ Pluggable Backends** - DiskANN, HNSW/FAISS with unified API

### 🛠️ Technical Highlights
- **🔄 Recompute Mode** - Highest accuracy scenarios while eliminating vector storage overhead
- **⚡ Zero-copy Operations** - Minimize IPC overhead by transferring distances instead of embeddings
- **🚀 High-throughput Embedding Pipeline** - Optimized batched processing for maximum efficiency
- **🎯 Two-level Search** - Novel coarse-to-fine search overlap for accelerated query processing (optional)
- **💾 Memory-mapped Indices** - Fast startup with raw text mapping to reduce memory overhead
- **🚀 MLX Support** - Ultra-fast recompute with quantized embedding models, accelerating building and search by 10-100x ([minimal example](test/build_mlx_index.py))

### 🎨 Developer Experience

- **Simple Python API** - Get started in minutes
- **Extensible backend system** - Easy to add new algorithms
- **Comprehensive examples** - From basic usage to production deployment

## Applications on your MacBook

### 📧 Lightweight RAG on your Apple Mail

LEANN can create a searchable index of your Apple Mail emails, allowing you to query your email history using natural language.

#### Quick Start

<details>
<summary><strong>📋 Click to expand: Command Examples</strong></summary>

```bash
# Use default mail path (works for most macOS setups)
python examples/mail_reader_leann.py

# Specify your own mail path
python examples/mail_reader_leann.py --mail-path "/Users/yourname/Library/Mail/V10/..."

# Run with custom index directory
python examples/mail_reader_leann.py --index-dir "./my_mail_index"

# embedd and search all of your email(this may take a long preprocessing time but it will encode all your emails)
python examples/mail_reader_leann.py --max-emails -1

# Limit number of emails processed (useful for testing)
python examples/mail_reader_leann.py --max-emails 1000

# Run a single query
python examples/mail_reader_leann.py --query "Find emails about project deadlines"
```

</details>

#### Finding Your Mail Path

<details>
<summary><strong>🔍 Click to expand: How to find your mail path</strong></summary>

The default mail path is configured for a typical macOS setup. If you need to find your specific mail path:

1. Open Terminal
2. Run: `find ~/Library/Mail -name "Messages" -type d | head -5`
3. Use the parent directory(ended with Data) of the Messages folder as your `--mail-path`

</details>

#### Example Queries

<details>
<summary><strong>💬 Click to expand: Example queries you can try</strong></summary>

Once the index is built, you can ask questions like:
- "Show me emails about meeting schedules"
- "Find emails from my boss about deadlines"
- "What did John say about the project timeline?"
- "Show me emails about travel expenses"

</details>

### 🌐 Lightweight RAG on your Google Chrome History

LEANN can create a searchable index of your Chrome browser history, allowing you to query your browsing history using natural language.

#### Quick Start

<details>
<summary><strong>📋 Click to expand: Command Examples</strong></summary>

Note you need to quit google right now to successfully run this.

```bash
# Use default Chrome profile (auto-finds all profiles) and recommand method to run this because usually default file is enough
python examples/google_history_reader_leann.py


# Run with custom index directory
python examples/google_history_reader_leann.py --index-dir "./my_chrome_index"

# Limit number of history entries processed (useful for testing)
python examples/google_history_reader_leann.py --max-entries 500

# Run a single query
python examples/google_history_reader_leann.py --query "What websites did I visit about machine learning?"

# Use only a specific profile (disable auto-find)
python examples/google_history_reader_leann.py --chrome-profile "~/Library/Application Support/Google/Chrome/Default" --no-auto-find-profiles
```

</details>

#### Finding Your Chrome Profile

<details>
<summary><strong>🔍 Click to expand: How to find your Chrome profile</strong></summary>

The default Chrome profile path is configured for a typical macOS setup. If you need to find your specific Chrome profile:

1. Open Terminal
2. Run: `ls ~/Library/Application\ Support/Google/Chrome/`
3. Look for folders like "Default", "Profile 1", "Profile 2", etc.
4. Use the full path as your `--chrome-profile` argument

**Common Chrome profile locations:**
- macOS: `~/Library/Application Support/Google/Chrome/Default`
- Linux: `~/.config/google-chrome/Default`

</details>

#### Example Queries

<details>
<summary><strong>💬 Click to expand: Example queries you can try</strong></summary>

Once the index is built, you can ask questions like:
- "What websites did I visit about machine learning?"
- "Find my search history about programming"
- "What YouTube videos did I watch recently?"
- "Show me websites I visited about travel planning"

</details>


### 💬 Lightweight RAG on your WeChat History

LEANN can create a searchable index of your WeChat chat history, allowing you to query your conversations using natural language.

#### Prerequisites

<details>
<summary><strong>🔧 Click to expand: Installation Requirements</strong></summary>

First, you need to install the WeChat exporter:

```bash
sudo packages/wechat-exporter/wechattweak-cli install
```

**Troubleshooting**: If you encounter installation issues, check the [WeChatTweak-CLI issues page](https://github.com/sunnyyoung/WeChatTweak-CLI/issues/41).

</details>

#### Quick Start

<details>
<summary><strong>📋 Click to expand: Command Examples</strong></summary>

```bash
# Use default settings (recommended for first run)
python examples/wechat_history_reader_leann.py

# Run with custom export directory and wehn we run the first time, LEANN will export all chat history automatically for you
python examples/wechat_history_reader_leann.py --export-dir "./my_wechat_exports"

# Run with custom index directory
python examples/wechat_history_reader_leann.py --index-dir "./my_wechat_index"

# Limit number of chat entries processed (useful for testing)
python examples/wechat_history_reader_leann.py --max-entries 1000

# Run a single query
python examples/wechat_history_reader_leann.py --query "Show me conversations about travel plans"

```

</details>

#### Example Queries

<details>
<summary><strong>💬 Click to expand: Example queries you can try</strong></summary>

Once the index is built, you can ask questions like:
- "我想买魔术师约翰逊的球衣，给我一些对应聊天记录?" (Chinese: Show me chat records about buying Magic Johnson's jersey)

</details>


## ⚡ Performance Comparison

### LEANN vs Faiss HNSW

We benchmarked LEANN against the popular Faiss HNSW implementation to demonstrate the significant memory and storage savings our approach provides:

```bash
# Run the comparison benchmark
python examples/compare_faiss_vs_leann.py
```

#### 🎯 Results Summary

| Metric | Faiss HNSW | LEANN HNSW | **Improvement** |
|--------|------------|-------------|-----------------|
| **Storage Size** | 5.5 MB | 0.5 MB | **11.4x smaller** (5.0 MB saved) |

#### 📈 Key Takeaways


- **💾 Storage Optimization**: LEANN requires **91% less storage** for the same dataset  

- **⚖️ Fair Comparison**: Both systems tested on identical hardware with the same 2,573 document dataset and the same embedding model and chunk method

> **Note**: Results may vary based on dataset size, hardware configuration, and query patterns. The comparison excludes text storage to focus purely on index structures.



*Benchmark results obtained on Apple Silicon with consistent environmental conditions*

## 📊 Benchmarks

### How to Reproduce Evaluation Results

Reproducing our benchmarks is straightforward. The evaluation script is designed to be self-contained, automatically downloading all necessary data on its first run.

#### 1. Environment Setup

First, ensure you have followed the installation instructions in the [Quick Start](#-quick-start) section. This will install all core dependencies.

Next, install the optional development dependencies, which include the `huggingface-hub` library required for automatic data download:

```bash
# This command installs all development dependencies
uv pip install -e ".[dev]"
```

#### 2. Run the Evaluation

Simply run the evaluation script. The first time you run it, it will detect that the data is missing, download it from Hugging Face Hub, and then proceed with the evaluation.

**To evaluate the DPR dataset:**
```bash
python examples/run_evaluation.py data/indices/dpr/dpr_diskann
```

**To evaluate the RPJ-Wiki dataset:**
```bash
python examples/run_evaluation.py data/indices/rpj_wiki/rpj_wiki.index
```

The script will print the recall and search time for each query, followed by the average results.

### Storage Usage Comparison

| System                | DPR (2.1M chunks) | RPJ-wiki (60M chunks) | Chat history (400K messages) | Apple emails (90K messages chunks) |
|-----------------------|------------------|------------------------|-----------------------------|------------------------------|
| Traditional Vector DB | 3.8 GB           | 201 GB                 | 1.8G                     | 305.8 MB                     |
| **LEANN**             | **324 MB**       | **6 GB**               | **64 MB**                 | **14.8 MB**                  |
| **Reduction**         | **91% smaller**  | **97% smaller**        | **97% smaller**             | **95% smaller**              |

<!-- ### Memory Usage Comparison

| System          j      | DPR(2M docs)     | RPJ-wiki(60M docs)    | Chat history()   |
| --------------------- | ---------------- | ---------------- | ---------------- |
| Traditional Vector DB(LLamaindex faiss) | x GB           | x GB            | x GB           |
| **Leann**       | **xx MB** | **x GB** | **x GB** |
| **Reduction**   | **x%**  | **x%**  | **x%**  |

### Query Performance of LEANN

| Backend             | Index Size | Query Time | Recall@3 |
| ------------------- | ---------- | ---------- | --------- |
| DiskANN             | 1M docs    | xms       | 0.95      |
| HNSW                | 1M docs    | xms        | 0.95      | -->

*Benchmarks run on Apple M3 Pro 36 GB*


## 🏗️ Architecture

<p align="center">
  <img src="asset/arch.png" alt="LEANN Architecture" width="800">
</p>

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


## 🤝 Contributing

We welcome contributions! Leann is built by the community, for the community.

### Ways to Contribute

- 🐛 **Bug Reports**: Found an issue? Let us know!
- 💡 **Feature Requests**: Have an idea? We'd love to hear it!
- 🔧 **Code Contributions**: PRs welcome for all skill levels
- 📖 **Documentation**: Help make Leann more accessible
- 🧪 **Benchmarks**: Share your performance results


<!-- ## ❓ FAQ

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
``` -->

## 📈 Roadmap

### 🎯 Q2 2025

- [X] DiskANN backend with MIPS/L2/Cosine support
- [X] HNSW backend integration
- [X] Real-time embedding pipeline
- [X] Memory-efficient graph pruning

### 🚀 Q3 2025


- [ ] Advanced caching strategies
- [ ] Add contextual-retrieval https://www.anthropic.com/news/contextual-retrieval
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

