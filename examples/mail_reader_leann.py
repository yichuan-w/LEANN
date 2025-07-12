import os
import asyncio
import dotenv
from pathlib import Path
from typing import List, Any
from leann.api import LeannBuilder, LeannSearcher, LeannChat
from llama_index.core.node_parser import SentenceSplitter

dotenv.load_dotenv()

def create_leann_index_from_multiple_sources(messages_dirs: List[Path], index_path: str = "mail_index.leann", max_count: int = -1):
    """
    Create LEANN index from multiple mail data sources.
    
    Args:
        messages_dirs: List of Path objects pointing to Messages directories
        index_path: Path to save the LEANN index
        max_count: Maximum number of emails to process per directory
    """
    print("Creating LEANN index from multiple mail data sources...")
    
    # Load documents using EmlxReader from LEANN_email_reader
    from LEANN_email_reader import EmlxReader
    reader = EmlxReader()
    # from email_data.email import EmlxMboxReader
    # from pathlib import Path
    # reader = EmlxMboxReader()
    
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

def create_leann_index(mail_path: str, index_path: str = "mail_index.leann", max_count: int = 1000):
    """
    Create LEANN index from mail data.
    
    Args:
        mail_path: Path to the mail directory
        index_path: Path to save the LEANN index
        max_count: Maximum number of emails to process
    """
    print("Creating LEANN index from mail data...")
    
    # Load documents using EmlxReader from LEANN_email_reader
    from LEANN_email_reader import EmlxReader
    reader = EmlxReader()
    # from email_data.email import EmlxMboxReader
    # from pathlib import Path
    # reader = EmlxMboxReader()
    documents = reader.load_data(Path(mail_path))
    
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
    # Base path to the mail data directory
    base_mail_path = "/Users/yichuan/Library/Mail/V10/0FCA0879-FD8C-4B7E-83BF-FDDA930791C5/[Gmail].mbox/All Mail.mbox/78BA5BE1-8819-4F9A-9613-EB63772F1DD0/Data"
    
    INDEX_DIR = Path("./mail_index_leann_raw_text_all")
    INDEX_PATH = str(INDEX_DIR / "mail_documents.leann")
    
    # Find all Messages directories
    from LEANN_email_reader import EmlxReader
    messages_dirs = EmlxReader.find_all_messages_directories(base_mail_path)
    
    if not messages_dirs:
        print("No Messages directories found. Exiting.")
        return
    
    # Create or load the LEANN index from all sources
    index_path = create_leann_index_from_multiple_sources(messages_dirs, INDEX_PATH)
    
    if index_path:
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