import argparse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import asyncio
import dotenv
from leann.api import LeannBuilder, LeannChat
from pathlib import Path

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
            "examples/data",
            recursive=True,
            encoding="utf-8",
            required_exts=[".pdf", ".txt", ".md"],
        ).load_data(show_progress=True)
        print("Documents loaded.")
        all_texts = []
        for doc in documents:
            nodes = node_parser.get_nodes_from_documents([doc])
            for node in nodes:
                all_texts.append(node.get_content())

        print("--- Index directory not found, building new index ---")

        print("\n[PHASE 1] Building Leann index...")

        # Use HNSW backend for better macOS compatibility
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
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

    print(f"\n[PHASE 2] Starting Leann chat session...")

    llm_config = {"type": "hf", "model": "Qwen/Qwen3-4B"}
    llm_config = {"type": "ollama", "model": "qwen3:8b"}
    llm_config = {"type": "openai", "model": "gpt-4o"}

    chat = LeannChat(index_path=INDEX_PATH, llm_config=llm_config)

    query = "Based on the paper, what are the main techniques LEANN explores to reduce the storage overhead and DLPM explore to achieve Fairness and Efiiciency trade-off?"

    # query = (
    #     "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面，任务令一般在什么城市颁发"
    # )

    print(f"You: {query}")
    chat_response = chat.ask(query, top_k=20, recompute_embeddings=True, complexity=32)
    print(f"Leann: {chat_response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Leann Chat with various LLM backends."
    )
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
    args = parser.parse_args()

    asyncio.run(main(args))
