import os
import asyncio
import dotenv
from pathlib import Path
from typing import List, Any
from leann.api import LeannBuilder, LeannSearcher, LeannChat
from mail_reader_llamaindex import EmlxReader
from llama_index.core.node_parser import SentenceSplitter

dotenv.load_dotenv()

def create_leann_index(mail_path: str, index_path: str = "mail_index.leann", max_count: int = 1000):
    """
    Create LEANN index from mail data.
    
    Args:
        mail_path: Path to the mail directory
        index_path: Path to save the LEANN index
        max_count: Maximum number of emails to process
    """
    print("Creating LEANN index from mail data...")
    
    # Load documents using EmlxReader from mail_reader_llamaindex
    reader = EmlxReader()
    documents = reader.load_data(mail_path, max_count=max_count)
    
    if not documents:
        print("No documents loaded. Exiting.")
        return None
    
    print(f"Loaded {len(documents)} email documents")
    
    # Create text splitter with 256 chunk size
    text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=25)
    
    # Convert Documents to text strings and chunk them
    all_texts = []
    for doc in documents:
        # Split the document into chunks
        nodes = text_splitter.get_nodes_from_documents([doc])
        for node in nodes:
            all_texts.append(node.get_content())
    
    print(f"Created {len(all_texts)} text chunks from {len(documents)} documents")
    
    # Create LEANN index directory
    INDEX_DIR = Path(index_path).parent
    
    if not INDEX_DIR.exists():
        print(f"--- Index directory not found, building new index ---")
        INDEX_DIR.mkdir(exist_ok=True)

        print(f"--- Building new LEANN index ---")
        
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

        print(f"Adding {len(all_texts)} email chunks to index...")
        for chunk_text in all_texts:
            builder.add_text(chunk_text)
            
        builder.build_index(index_path)
        print(f"\nLEANN index built at {index_path}!")
    else:
        print(f"--- Using existing index at {INDEX_DIR} ---")
    
    return index_path

async def query_leann_index(index_path: str, query: str):
    """
    Query the LEANN index.
    
    Args:
        index_path: Path to the LEANN index
        query: The query string
    """
    print(f"\n[PHASE 2] Starting Leann chat session...")
    chat = LeannChat(index_path=index_path)
    
    print(f"You: {query}")
    chat_response = chat.ask(
        query, 
        top_k=5, 
        recompute_beighbor_embeddings=True,
        complexity=32,
        beam_width=1
    )
    print(f"Leann: {chat_response}")

async def main():
    mail_path = "/Users/yichuan/Library/Mail/V10/0FCA0879-FD8C-4B7E-83BF-FDDA930791C5/[Gmail].mbox/All Mail.mbox/78BA5BE1-8819-4F9A-9613-EB63772F1DD0/Data/9/Messages"
    
    INDEX_DIR = Path("./mail_index_leann")
    INDEX_PATH = str(INDEX_DIR / "mail_documents.leann")
    
    # Create or load the LEANN index
    index_path = create_leann_index(mail_path, INDEX_PATH, max_count=1000)
    
    if index_path:
        # Example queries
        queries = [
            "Hows Berkeley Graduate Student Instructor",
            "how's the icloud related advertisement saying"
        ]
        
        for query in queries:
            print("\n" + "="*60)
            await query_leann_index(index_path, query)

if __name__ == "__main__":
    asyncio.run(main()) 