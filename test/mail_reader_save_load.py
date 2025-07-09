import os
import email
from pathlib import Path
from typing import List, Any
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter

class EmlxReader(BaseReader):
    """
    Apple Mail .emlx file reader.
    
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
                                        if part.get_content_type() == "text/plain":
                                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                            break
                                else:
                                    body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                                
                                # Create document content
                                doc_content = f"""
From: {from_addr}
To: {to_addr}
Subject: {subject}
Date: {date}

{body}
"""
                                
                                # Create metadata
                                metadata = {
                                    'file_path': filepath,
                                    'subject': subject,
                                    'from': from_addr,
                                    'to': to_addr,
                                    'date': date,
                                    'filename': filename
                                }
                                
                                doc = Document(text=doc_content, metadata=metadata)
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

def create_and_save_index(mail_path: str, save_dir: str = "mail_index", max_count: int = 1000):
    """
    Create the index from mail data and save it to disk.
    
    Args:
        mail_path: Path to the mail directory
        save_dir: Directory to save the index
        max_count: Maximum number of emails to process
    """
    print("Creating index from mail data...")
    
    # Load documents
    documents = EmlxReader().load_data(mail_path, max_count=max_count)
    
    if not documents:
        print("No documents loaded. Exiting.")
        return None
    
    # Create text splitter
    text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=0)
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter]
    )
    
    # Save the index
    os.makedirs(save_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=save_dir)
    print(f"Index saved to {save_dir}")
    
    return index

def load_index(save_dir: str = "mail_index"):
    """
    Load the saved index from disk.
    
    Args:
        save_dir: Directory where the index is saved
    
    Returns:
        Loaded index or None if loading fails
    """
    try:
        # Load storage context
        storage_context = StorageContext.from_defaults(persist_dir=save_dir)
        
        # Load index
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
    """
    Query the loaded index.
    
    Args:
        index: The loaded index
        query: The query string
    """
    if index is None:
        print("No index available for querying.")
        return
    
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")

def main():
    mail_path = "/Users/yichuan/Library/Mail/V10/0FCA0879-FD8C-4B7E-83BF-FDDA930791C5/[Gmail].mbox/All Mail.mbox/78BA5BE1-8819-4F9A-9613-EB63772F1DD0/Data/9/Messages"
    save_dir = "mail_index"
    
    # Check if index already exists
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "vector_store.json")):
        print("Loading existing index...")
        index = load_index(save_dir)
    else:
        print("Creating new index...")
        index = create_and_save_index(mail_path, save_dir, max_count=1000)
    
    if index:
        # Example queries
        queries = [
            "Hows Berkeley Graduate Student Instructor",
            "What emails mention GSR appointments?",
            "Find emails about deadlines"
        ]
        
        for query in queries:
            print("\n" + "="*50)
            query_index(index, query)

if __name__ == "__main__":
    main() 