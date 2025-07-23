import os
import email
from pathlib import Path
from typing import List, Any
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

def find_all_messages_directories(root: str = None) -> List[Path]:
    """
    Recursively find all 'Messages' directories under the given root.
    Returns a list of Path objects.
    """
    if root is None:
        # Auto-detect user's mail path
        home_dir = os.path.expanduser("~")
        root = os.path.join(home_dir, "Library", "Mail")
    
    messages_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "Messages":
            messages_dirs.append(Path(dirpath))
    return messages_dirs

class EmlxReader(BaseReader):
    """
    Apple Mail .emlx file reader with embedded metadata.
    
    Reads individual .emlx files from Apple Mail's storage format.
    """
    
    def __init__(self, include_html: bool = False) -> None:
        """
        Initialize.
        
        Args:
            include_html: Whether to include HTML content in the email body (default: False)
        """
        self.include_html = include_html
    
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
                                            if part.get_content_type() == "text/html" and not self.include_html:
                                                continue
                                            body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                            # break
                                else:
                                    body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                                
                                # Create document content with metadata embedded in text
                                doc_content = f"""
[File]: {filename}
[From]: {from_addr}
[To]: {to_addr}
[Subject]: {subject}
[Date]: {date}
[EMAIL BODY Start]:
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