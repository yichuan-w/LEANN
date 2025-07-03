from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
import asyncio
import os
import dotenv
from leann.api import LeannBuilder, LeannSearcher, LeannChat
import leann_backend_diskann # Import to ensure backend registration
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
    chunker=HybridChunker(tokenizer="Qwen/Qwen3-Embedding-4B", max_tokens=256)
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
# Extract text from documents and prepare for Leann
all_texts = []
for doc in documents:
    # DoclingNodeParser returns Node objects, which have a text attribute
    nodes = node_parser.get_nodes_from_documents([doc])
    for node in nodes:
        all_texts.append(node.text)

INDEX_DIR = Path("./test_pdf_index")
INDEX_PATH = str(INDEX_DIR / "pdf_documents.leann")

if INDEX_DIR.exists():
    print(f"--- Cleaning up old index directory: {INDEX_DIR} ---")
    shutil.rmtree(INDEX_DIR)

print(f"\n[PHASE 1] Building Leann index...")

builder = LeannBuilder(
    backend_name="diskann",
    embedding_model="facebook/contriever", # Using a common sentence transformer model
    graph_degree=32, 
    complexity=64
)

print(f"Loaded {len(all_texts)} text chunks from documents.")
for chunk_text in all_texts:
    builder.add_text(chunk_text)
    
builder.build_index(INDEX_PATH)
print(f"\nLeann index built at {INDEX_PATH}!")

async def main():
    print(f"\n[PHASE 2] Starting Leann chat session...")
    chat = LeannChat(index_path=INDEX_PATH)
    
    query = "Based on the paper, what are the main techniques LEANN explores to reduce the storage overhead and DLPM explore to achieve Fairness and Efiiciency trade-off?"
    # query = "What is the Off-policy training in RL?"
    print(f"You: {query}")
    chat_response = chat.ask(query, top_k=20, recompute_beighbor_embeddings=True)
    print(f"Leann: {chat_response}")

if __name__ == "__main__":
    asyncio.run(main())