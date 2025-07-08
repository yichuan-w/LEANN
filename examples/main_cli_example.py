import faulthandler
faulthandler.enable()

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
import asyncio
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
    ".txt": reader,
    ".md": reader,
}
node_parser = DoclingNodeParser(
    chunker=HybridChunker(tokenizer="facebook/contriever", max_tokens=128)
)
print("Loading documents...")
documents = SimpleDirectoryReader(
    "examples/data", 
    recursive=True, 
    file_extractor=file_extractor,
    encoding="utf-8",
    required_exts=[".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"]
).load_data(show_progress=True)
print("Documents loaded.")
all_texts = []
for doc in documents:
    nodes = node_parser.get_nodes_from_documents([doc])
    for node in nodes:
        all_texts.append(node.get_content())

INDEX_DIR = Path("./test_pdf_index_pangu_test")
INDEX_PATH = str(INDEX_DIR / "pdf_documents.leann")

if not INDEX_DIR.exists():
    print(f"--- Index directory not found, building new index ---")
    
    print(f"\n[PHASE 1] Building Leann index...")

    # Use HNSW backend for better macOS compatibility
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model="facebook/contriever",
        graph_degree=32, 
        complexity=64,
        is_compact=True,
        is_recompute=True,
        num_threads=1  # Force single-threaded mode
    )

    print(f"Loaded {len(all_texts)} text chunks from documents.")
    for chunk_text in all_texts:
        builder.add_text(chunk_text)
        
    builder.build_index(INDEX_PATH)
    print(f"\nLeann index built at {INDEX_PATH}!")
else:
    print(f"--- Using existing index at {INDEX_DIR} ---")

async def main():
    print(f"\n[PHASE 2] Starting Leann chat session...")
    chat = LeannChat(index_path=INDEX_PATH)
    
    query = "Based on the paper, what are the main techniques LEANN explores to reduce the storage overhead and DLPM explore to achieve Fairness and Efiiciency trade-off?"
    query = "What is the main idea of RL and give me 5 exapmle of classic RL algorithms?"
    query = "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面，任务令一般在什么城市颁发"

    print(f"You: {query}")
    chat_response = chat.ask(query, top_k=20, recompute_beighbor_embeddings=True,complexity=32,beam_width=1)
    print(f"Leann: {chat_response}")

if __name__ == "__main__":
    asyncio.run(main())