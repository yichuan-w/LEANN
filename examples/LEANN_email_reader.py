import os
import email
from pathlib import Path
from typing import List, Any
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

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
                                            # if part.get_content_type() == "text/html":
                                            #     continue
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

    @staticmethod
    def find_all_messages_directories(base_path: str) -> List[Path]:
        """
        Find all Messages directories under the given base path.
        
        Args:
            base_path: Base path to search for Messages directories
            
        Returns:
            List of Path objects pointing to Messages directories
        """
        base_path_obj = Path(base_path)
        messages_dirs = []
        
        if not base_path_obj.exists():
            print(f"Base path {base_path} does not exist")
            return messages_dirs
        
        # Find all Messages directories recursively
        for messages_dir in base_path_obj.rglob("Messages"):
            if messages_dir.is_dir():
                messages_dirs.append(messages_dir)
                print(f"Found Messages directory: {messages_dir}")
        
        print(f"Found {len(messages_dirs)} Messages directories")
        return messages_dirs 