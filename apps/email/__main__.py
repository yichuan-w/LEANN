import os
import sys
import asyncio
import dotenv
import argparse
from pathlib import Path
from typing import List, Any

from leann.api import LeannBuilder, LeannSearcher, LeannChat
from llama_index.core.node_parser import SentenceSplitter

dotenv.load_dotenv()

# Auto-detect user's mail path
def get_mail_path():
    """Get the mail path for the current user"""
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, "Library", "Mail")

def create_leann_index_from_multiple_sources(messages_dirs: List[Path], index_path: str = "mail_index.leann", max_count: int = -1, include_html: bool = False, embedding_model: str = "facebook/contriever"):
    """
    Create LEANN index from multiple mail data sources.
    
    Args:
        messages_dirs: List of Path objects pointing to Messages directories
        index_path: Path to save the LEANN index
        max_count: Maximum number of emails to process per directory
        include_html: Whether to include HTML content in email processing
    """
    print("Creating LEANN index from multiple mail data sources...")
    
    # Load documents using EmlxReader from local readers module
    from .readers import EmlxReader, find_all_messages_directories
    reader = EmlxReader(include_html=include_html)
    INDEX_DIR = Path(index_path).parent
    
    if not INDEX_DIR.exists():
        print(f"--- Index directory not found, building new index ---")
        all_documents = []
        total_processed = 0
        
        # Process each Messages directory
        for i, messages_dir in enumerate(messages_dirs):
            print(f"\nProcessing Messages directory {i+1}/{len(messages_dirs)}: {messages_dir}")
            
            try:
                documents = reader.load_data(messages_dir)
                if documents:
                    print(f"Loaded {len(documents)} email documents from {messages_dir}")
                    all_documents.extend(documents)
                    total_processed += len(documents)
                    
                    # Check if we've reached the max count
                    if max_count > 0 and total_processed >= max_count:
                        print(f"Reached max count of {max_count} documents")
                        break
                else:
                    print(f"No documents loaded from {messages_dir}")
            except Exception as e:
                print(f"Error processing {messages_dir}: {e}")
                continue
        
        if not all_documents:
            print("No documents loaded from any source. Exiting.")
            return None
        
        print(f"\nTotal loaded {len(all_documents)} email documents from {len(messages_dirs)} directories")
        
        # Create text splitter with 256 chunk size
        text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=25)
        
        # Convert Documents to text strings and chunk them
        all_texts = []
        for doc in all_documents:
            # Split the document into chunks
            nodes = text_splitter.get_nodes_from_documents([doc])
            for node in nodes:
                all_texts.append(node.get_content())
        
        print(f"Created {len(all_texts)} text chunks from {len(all_documents)} documents")
        
        # Create LEANN index directory
        print(f"--- Index directory not found, building new index ---")
        INDEX_DIR.mkdir(exist_ok=True)

        print(f"--- Building new LEANN index ---")
        
        print(f"\n[PHASE 1] Building Leann index...")

        # Use HNSW backend for better macOS compatibility
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=embedding_model,
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
    chat = LeannChat(index_path=index_path,
                     llm_config={"type": "openai", "model": "gpt-4o"})
    
    print(f"You: {query}")
    import time
    start_time = time.time()
    chat_response = chat.ask(
        query, 
        top_k=10, 
        recompute_beighbor_embeddings=True,
        complexity=12,
        beam_width=1,
        
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Leann: {chat_response}")

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LEANN Mail Reader - Create and query email index')
    parser.add_argument('--index-dir', type=str, default="./mail_index_leann_raw_text_all_dicts",
                       help='Directory to store the LEANN index (default: ./mail_index_leann_raw_text_all_dicts)')
    parser.add_argument('--max-emails', type=int, default=1000,
                       help='Maximum number of emails to process (-1 means all)')
    parser.add_argument('--query', type=str, default="Give me some funny advertisement about apple or other companies",
                       help='Single query to run (default: runs example queries)')
    parser.add_argument('--include-html', action='store_true', default=False,
                       help='Include HTML content in email processing (default: False)')
    parser.add_argument('--embedding-model', type=str, default="facebook/contriever",
                       help='Embedding model to use (default: facebook/contriever)')
    
    args = parser.parse_args()

    print(f"args: {args}")
    
    # Automatically find all Messages directories under the current user's Mail directory
    from .readers import find_all_messages_directories
    mail_path = get_mail_path()
    print(f"Searching for email data in: {mail_path}")
    messages_dirs = find_all_messages_directories(mail_path)
    
    print('len(messages_dirs): ', len(messages_dirs))
    
    if not messages_dirs:
        print("No Messages directories found. Exiting.")
        return
    
    INDEX_DIR = Path(args.index_dir)
    INDEX_PATH = str(INDEX_DIR / "mail_documents.leann")
    print(f"Index directory: {INDEX_DIR}")
    print(f"Found {len(messages_dirs)} Messages directories.")
    
    # Create or load the LEANN index from all sources
    index_path = create_leann_index_from_multiple_sources(messages_dirs, INDEX_PATH, args.max_emails, args.include_html, args.embedding_model)
    
    if index_path:
        if args.query:
            # Run single query
            await query_leann_index(index_path, args.query)
        else:
            # Example queries
            queries = [
                "Hows Berkeley Graduate Student Instructor",
                "how's the icloud related advertisement saying",
                "Whats the number of class recommend to take per semester for incoming EECS students"
            ]
            for query in queries:
                print("\n" + "="*60)
                await query_leann_index(index_path, query)

if __name__ == "__main__":
    asyncio.run(main())