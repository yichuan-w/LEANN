import os
import email
from pathlib import Path
from typing import List, Any
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter

# --- EMBEDDING MODEL ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

# --- END EMBEDDING MODEL ---

class EmlxReader(BaseReader):
    """
    Apple Mail .emlx file reader with embedded metadata.
    
    Reads individual .emlx files from Apple Mail's storage format.
    """
    
    def __init__(self) -> None:
        """Initialize."""
        pass
    
    def load_data(self, input_dir: str, **load_kwargs: Any) -> List[Document]:
        """
        Load data from the input directory containing .emlx files.
        
        Args:
            input_dir: Directory containing .emlx files
            **load_kwargs:
                max_count (int): Maximum amount of messages to read.
        """
        docs: List[Document] = []
        max_count = load_kwargs.get('max_count', 1000)
        count = 0
        
        # Walk through the directory recursively
        for dirpath, dirnames, filenames in os.walk(input_dir):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            
            for filename in filenames:
                if count >= max_count:
                    break
                    
                if filename.endswith(".emlx"):
                    filepath = os.path.join(dirpath, filename)
                    try:
                        # Read the .emlx file
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # .emlx files have a length prefix followed by the email content
                        # The first line contains the length, followed by the email
                        lines = content.split('\n', 1)
                        if len(lines) >= 2:
                            email_content = lines[1]
                            
                            # Parse the email using Python's email module
                            try:
                                msg = email.message_from_string(email_content)
                                
                                # Extract email metadata
                                subject = msg.get('Subject', 'No Subject')
                                from_addr = msg.get('From', 'Unknown')
                                to_addr = msg.get('To', 'Unknown')
                                date = msg.get('Date', 'Unknown')
                                
                                # Extract email body
                                body = ""
                                if msg.is_multipart():
                                    for part in msg.walk():
                                        if part.get_content_type() == "text/plain" or part.get_content_type() == "text/html":
                                            if part.get_content_type() == "text/html":
                                                continue
                                            body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                            # break
                                else:
                                    body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                                
                                # Create document content with metadata embedded in text
                                doc_content = f"""
[EMAIL METADATA]
File: {filename}
From: {from_addr}
To: {to_addr}
Subject: {subject}
Date: {date}
[END METADATA]

{body}
"""
                                
                                # No separate metadata - everything is in the text
                                doc = Document(text=doc_content, metadata={})
                                docs.append(doc)
                                count += 1
                                
                            except Exception as e:
                                print(f"Error parsing email from {filepath}: {e}")
                                continue
                                
                    except Exception as e:
                        print(f"Error reading file {filepath}: {e}")
                        continue
        
        print(f"Loaded {len(docs)} email documents")
        return docs

def create_and_save_index(mail_path: str, save_dir: str = "mail_index_embedded", max_count: int = 1000):
    print("Creating index from mail data with embedded metadata...")
    documents = EmlxReader().load_data(mail_path, max_count=max_count)
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
    mail_path = "/Users/yichuan/Library/Mail/V10/0FCA0879-FD8C-4B7E-83BF-FDDA930791C5/[Gmail].mbox/All Mail.mbox/78BA5BE1-8819-4F9A-9613-EB63772F1DD0/Data/9/Messages"
    save_dir = "mail_index_embedded"
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "vector_store.json")):
        print("Loading existing index...")
        index = load_index(save_dir)
    else:
        print("Creating new index...")
        index = create_and_save_index(mail_path, save_dir, max_count=10000)
    if index:
        queries = [
            "Hows Berkeley Graduate Student Instructor",
            "how's the icloud related advertisement saying"
            "Whats the number of class recommend to take per semester for incoming EECS students"
        ]
        for query in queries:
            print("\n" + "="*50)
            query_index(index, query)

if __name__ == "__main__":
    main() 