import argparse
import asyncio
from pathlib import Path

import dotenv
from leann.api import LeannBuilder, LeannChat
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

dotenv.load_dotenv()


async def main(args):
    INDEX_DIR = Path(args.index_dir)
    INDEX_PATH = str(INDEX_DIR / "pdf_documents.leann")

    if not INDEX_DIR.exists():
        node_parser = SentenceSplitter(
            chunk_size=256, chunk_overlap=128, separator=" ", paragraph_separator="\n\n"
        )

        print("Loading documents...")
        documents = SimpleDirectoryReader(
            args.data_dir,
            recursive=True,
            encoding="utf-8",
            required_exts=[".pdf", ".txt", ".md"],
        ).load_data(show_progress=True)
        print("Documents loaded.")
        all_texts = []
        for doc in documents:
            nodes = node_parser.get_nodes_from_documents([doc])
            if nodes:
                all_texts.extend(node.get_content() for node in nodes)

        print("--- Index directory not found, building new index ---")

        print("\n[PHASE 1] Building Leann index...")

        # LeannBuilder now automatically detects normalized embeddings and sets appropriate distance metric
        print(f"Using {args.embedding_model} with {args.embedding_mode} mode")

        # Use HNSW backend for better macOS compatibility
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            # distance_metric is automatically set based on embedding model
            graph_degree=32,
            complexity=64,
            is_compact=True,
            is_recompute=True,
            num_threads=1,  # Force single-threaded mode
        )

        print(f"Loaded {len(all_texts)} text chunks from documents.")
        for chunk_text in all_texts:
            builder.add_text(chunk_text)

        builder.build_index(INDEX_PATH)
        print(f"\nLeann index built at {INDEX_PATH}!")
    else:
        print(f"--- Using existing index at {INDEX_DIR} ---")

    print("\n[PHASE 2] Starting Leann chat session...")

    # Build llm_config based on command line arguments
    if args.llm == "simulated":
        llm_config = {"type": "simulated"}
    elif args.llm == "ollama":
        llm_config = {"type": "ollama", "model": args.model, "host": args.host}
    elif args.llm == "hf":
        llm_config = {"type": "hf", "model": args.model}
    elif args.llm == "openai":
        llm_config = {"type": "openai", "model": args.model}
    else:
        raise ValueError(f"Unknown LLM type: {args.llm}")

    print(f"Using LLM: {args.llm} with model: {args.model if args.llm != 'simulated' else 'N/A'}")

    chat = LeannChat(index_path=INDEX_PATH, llm_config=llm_config)
    # query = (
    #     "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面,任务令一般在什么城市颁发"
    # )
    query = args.query

    print(f"You: {query}")
    chat_response = chat.ask(query, top_k=20, recompute_embeddings=True, complexity=32)
    print(f"Leann chat response: \033[36m{chat_response}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Leann Chat with various LLM backends.")
    parser.add_argument(
        "--llm",
        type=str,
        default="hf",
        choices=["simulated", "ollama", "hf", "openai"],
        help="The LLM backend to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="The model name to use (e.g., 'llama3:8b' for ollama, 'deepseek-ai/deepseek-llm-7b-chat' for hf, 'gpt-4o' for openai).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="facebook/contriever",
        help="The embedding model to use (e.g., 'facebook/contriever', 'text-embedding-3-small').",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "mlx"],
        help="The embedding backend mode.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="The host for the Ollama API.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./test_doc_files",
        help="Directory where the Leann index will be stored.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="examples/data",
        help="Directory containing documents to index (PDF, TXT, MD files).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Based on the paper, what are the main techniques LEANN explores to reduce the storage overhead and DLPM explore to achieve Fairness and Efiiciency trade-off?",
        help="The query to ask the Leann chat system.",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
