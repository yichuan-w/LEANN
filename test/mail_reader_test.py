import os
import email
from pathlib import Path
from typing import List, Any
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.readers.base import BaseReader

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
        
        # Check if directory exists and is accessible
        if not os.path.exists(input_dir):
            print(f"Error: Directory '{input_dir}' does not exist")
            return docs
        
        if not os.access(input_dir, os.R_OK):
            print(f"Error: Directory '{input_dir}' is not accessible (permission denied)")
            print("This is likely due to macOS security restrictions on Mail app data")
            return docs
        
        print(f"Scanning directory: {input_dir}")
        
        # Walk through the directory recursively
        for dirpath, dirnames, filenames in os.walk(input_dir):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            
            for filename in filenames:
                if count >= max_count:
                    break
                    
                if filename.endswith(".emlx"):
                    filepath = os.path.join(dirpath, filename)
                    print(f"Found .emlx file: {filepath}")
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

def main():
    # Use the current directory where the sample.emlx file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Testing EmlxReader with sample .emlx file...")
    print(f"Scanning directory: {current_dir}")
    
    # Use the custom EmlxReader
    documents = EmlxReader().load_data(current_dir, max_count=1000)
    
    if not documents:
        print("No documents loaded. Make sure sample.emlx exists in the examples directory.")
        return
    
    print(f"\nSuccessfully loaded {len(documents)} document(s)")
    
    # Initialize index with documents
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    print("\nTesting query: 'Hows Berkeley Graduate Student Instructor'")
    res = query_engine.query("Hows Berkeley Graduate Student Instructor")
    print(f"Response: {res}")

if __name__ == "__main__":
    main() 