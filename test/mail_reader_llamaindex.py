import email
import os
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.readers.base import BaseReader


class EmlxReader(BaseReader):
    """
    Apple Mail .emlx file reader.

    Reads individual .emlx files from Apple Mail's storage format.
    """

    def __init__(self) -> None:
        """Initialize."""
        pass

    def load_data(self, input_dir: str, **load_kwargs: Any) -> list[Document]:
        """
        Load data from the input directory containing .emlx files.

        Args:
            input_dir: Directory containing .emlx files
            **load_kwargs:
                max_count (int): Maximum amount of messages to read.
        """
        docs: list[Document] = []
        max_count = load_kwargs.get("max_count", 1000)
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
                        with open(filepath, encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        # .emlx files have a length prefix followed by the email content
                        # The first line contains the length, followed by the email
                        lines = content.split("\n", 1)
                        if len(lines) >= 2:
                            email_content = lines[1]

                            # Parse the email using Python's email module
                            try:
                                msg = email.message_from_string(email_content)

                                # Extract email metadata
                                subject = msg.get("Subject", "No Subject")
                                from_addr = msg.get("From", "Unknown")
                                to_addr = msg.get("To", "Unknown")
                                date = msg.get("Date", "Unknown")

                                # Extract email body
                                body = ""
                                if msg.is_multipart():
                                    for part in msg.walk():
                                        if (
                                            part.get_content_type() == "text/plain"
                                            or part.get_content_type() == "text/html"
                                        ):
                                            body += part.get_payload(decode=True).decode(
                                                "utf-8", errors="ignore"
                                            )
                                            # break
                                else:
                                    body = msg.get_payload(decode=True).decode(
                                        "utf-8", errors="ignore"
                                    )

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
                                    "file_path": filepath,
                                    "subject": subject,
                                    "from": from_addr,
                                    "to": to_addr,
                                    "date": date,
                                    "filename": filename,
                                }
                                if count == 0:
                                    print("--------------------------------")
                                    print("dir path", dirpath)
                                    print(metadata)
                                    print(doc_content)
                                    print("--------------------------------")
                                    body = []
                                    if msg.is_multipart():
                                        for part in msg.walk():
                                            print(
                                                "--------------------------------  get content type -------------------------------"
                                            )
                                            print(part.get_content_type())
                                            print(part)
                                            # body.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
                                            print(
                                                "--------------------------------  get content type -------------------------------"
                                            )
                                    else:
                                        body = msg.get_payload(decode=True).decode(
                                            "utf-8", errors="ignore"
                                        )
                                        print(body)

                                    print(body)
                                    print("--------------------------------")
                                doc = Document(text=doc_content, metadata=metadata)
                                docs.append(doc)
                                count += 1

                            except Exception as e:
                                print(f"!!!!!!! Error parsing email from {filepath}: {e} !!!!!!!!")
                                continue

                    except Exception as e:
                        print(f"!!!!!!! Error reading file !!!!!!!! {filepath}: {e}")
                        continue

        print(f"Loaded {len(docs)} email documents")
        return docs


# Use the custom EmlxReader instead of MboxReader
documents = EmlxReader().load_data(
    "/Users/yichuan/Library/Mail/V10/0FCA0879-FD8C-4B7E-83BF-FDDA930791C5/[Gmail].mbox/All Mail.mbox/78BA5BE1-8819-4F9A-9613-EB63772F1DD0/Data/9/Messages",
    max_count=1000,
)  # Returns list of documents

# Configure the index with larger chunk size to handle long metadata
from llama_index.core.node_parser import SentenceSplitter

# Create a custom text splitter with larger chunk size
text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=200)

index = VectorStoreIndex.from_documents(
    documents, transformations=[text_splitter]
)  # Initialize index with documents

query_engine = index.as_query_engine()
res = query_engine.query("Hows Berkeley Graduate Student Instructor")
print(res)
