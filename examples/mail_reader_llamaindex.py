import os
import sys
import argparse
from pathlib import Path
from typing import List, Any

# Add the project root to Python path so we can import from examples
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter

# --- EMBEDDING MODEL ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

# --- END EMBEDDING MODEL ---

# Import EmlxReader from the new module
from examples.email_data.LEANN_email_reader import EmlxReader

def create_and_save_index(mail_path: str, save_dir: str = "mail_index_embedded", max_count: int = 1000, include_html: bool = False):
    print("Creating index from mail data with embedded metadata...")
    documents = EmlxReader(include_html=include_html).load_data(mail_path, max_count=max_count)
    if not documents:
        print("No documents loaded. Exiting.")
        return None
    text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=25)
    # Use facebook/contriever as the embedder
    embed_model = HuggingFaceEmbedding(model_name="facebook/contriever")
    # set on device
    import torch
    if torch.cuda.is_available():
        embed_model._model.to("cuda")
    # set mps
    elif torch.backends.mps.is_available():
        embed_model._model.to("mps")
    else:
        embed_model._model.to("cpu")
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model
    )
    os.makedirs(save_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=save_dir)
    print(f"Index saved to {save_dir}")
    return index

def load_index(save_dir: str = "mail_index_embedded"):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=save_dir)
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context
        )
        print(f"Index loaded from {save_dir}")
        return index
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def query_index(index, query: str):
    if index is None:
        print("No index available for querying.")
        return
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LlamaIndex Mail Reader - Create and query email index')
    parser.add_argument('--mail-path', type=str, 
                       default="/Users/yichuan/Library/Mail/V10/0FCA0879-FD8C-4B7E-83BF-FDDA930791C5/[Gmail].mbox/All Mail.mbox/78BA5BE1-8819-4F9A-9613-EB63772F1DD0/Data/9/Messages",
                       help='Path to mail data directory')
    parser.add_argument('--save-dir', type=str, default="mail_index_embedded",
                       help='Directory to store the index (default: mail_index_embedded)')
    parser.add_argument('--max-emails', type=int, default=10000,
                       help='Maximum number of emails to process')
    parser.add_argument('--include-html', action='store_true', default=False,
                       help='Include HTML content in email processing (default: False)')
    
    args = parser.parse_args()
    
    mail_path = args.mail_path
    save_dir = args.save_dir
    
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "vector_store.json")):
        print("Loading existing index...")
        index = load_index(save_dir)
    else:
        print("Creating new index...")
        index = create_and_save_index(mail_path, save_dir, max_count=args.max_emails, include_html=args.include_html)
    if index:
        queries = [
            "Hows Berkeley Graduate Student Instructor",
            "how's the icloud related advertisement saying",
            "Whats the number of class recommend to take per semester for incoming EECS students"
        ]
        for query in queries:
            print("\n" + "="*50)
            query_index(index, query)

if __name__ == "__main__":
    main() 