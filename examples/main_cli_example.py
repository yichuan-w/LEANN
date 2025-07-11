import faulthandler
faulthandler.enable()

import argparse
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
import asyncio
import os
import dotenv
from leann.api import LeannBuilder, LeannSearcher, LeannChat
import shutil
from pathlib import Path

dotenv.load_dotenv()

reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
file_extractor: dict[str, BaseReader] = {
    ".docx": reader,
    ".pptx": reader, 
    ".pdf": reader,
    ".xlsx": reader,
}
node_parser = DoclingNodeParser(
    chunker=HybridChunker(tokenizer="Qwen/Qwen3-Embedding-4B", max_tokens=64)
)
print("Loading documents...")
documents = SimpleDirectoryReader(
    "examples/data", 
    recursive=True, 
    file_extractor=file_extractor,
    encoding="utf-8",
    required_exts=[".pdf", ".docx", ".pptx", ".xlsx"]
).load_data(show_progress=True)
print("Documents loaded.")
all_texts = []
for doc in documents:
    nodes = node_parser.get_nodes_from_documents([doc])
    for node in nodes:
        all_texts.append(node.get_content())

INDEX_DIR = Path("./test_pdf_index")
INDEX_PATH = str(INDEX_DIR / "pdf_documents.leann")

if not INDEX_DIR.exists():
    print(f"--- Index directory not found, building new index ---")
    
    print(f"\n[PHASE 1] Building Leann index...")

    # CSR compact mode with recompute
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model="facebook/contriever",
        graph_degree=32, 
        complexity=64,
        is_compact=True,
        is_recompute=True
    )

    print(f"Loaded {len(all_texts)} text chunks from documents.")
    for chunk_text in all_texts:
        builder.add_text(chunk_text)
        
    builder.build_index(INDEX_PATH)
    print(f"\nLeann index built at {INDEX_PATH}!")
else:
    print(f"--- Using existing index at {INDEX_DIR} ---")

async def main(args):
    print(f"\n[PHASE 2] Starting Leann chat session...")
    
    llm_config = {
        "type": args.llm,
        "model": args.model,
        "host": args.host
    }

    chat = LeannChat(index_path=INDEX_PATH, llm_config=llm_config)
    
    query = "Based on the paper, what are the main techniques LEANN explores to reduce the storage overhead and DLPM explore to achieve Fairness and Efiiciency trade-off?"
    print(f"You: {query}")
    chat_response = chat.ask(query, top_k=3, recompute_beighbor_embeddings=True)
    print(f"Leann: {chat_response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Leann Chat with various LLM backends.")
    parser.add_argument("--llm", type=str, default="hf", choices=["simulated", "ollama", "hf"], help="The LLM backend to use.")
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.2-3B-Instruct', help="The model name to use (e.g., 'llama3:8b' for ollama, 'deepseek-ai/deepseek-llm-7b-chat' for hf).")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="The host for the Ollama API.")
    args = parser.parse_args()

    asyncio.run(main(args))