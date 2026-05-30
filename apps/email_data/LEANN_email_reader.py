"""Standard .eml email reader — parses RFC 2822 .eml files from a directory."""

import email
import os
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


def find_all_messages_directories(root: str | None = None) -> list[Path]:
    """Recursively find all 'Messages' directories under the given root."""
    if root is None:
        home_dir = os.path.expanduser("~")
        root = os.path.join(home_dir, "Library", "Mail")

    messages_dirs = []
    for dirpath, _dirnames, _filenames in os.walk(root):
        if os.path.basename(dirpath) == "Messages":
            messages_dirs.append(Path(dirpath))
    return messages_dirs


def _payload_to_text(payload: object) -> str:
    """Safely decode an email payload to text."""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="ignore")
    if isinstance(payload, str):
        return payload
    return ""


def _parse_email_message(msg: Any, include_html: bool = False) -> str:
    """Extract body text from an email.message.Message, handling multipart."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype not in ("text/plain", "text/html"):
                continue
            if ctype == "text/html" and not include_html:
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    body += _payload_to_text(payload)
            except Exception:
                continue
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body = _payload_to_text(payload)
        except Exception:
            body = ""
    return body


class EmlxReader(BaseReader):
    """Apple Mail .emlx file reader.

    Reads individual .emlx files from Apple Mail's storage format.
    .emlx files have a length prefix line followed by the RFC 2822 content.
    """

    def __init__(self, include_html: bool = False) -> None:
        self.include_html = include_html

    def load_data(self, input_dir: str, **load_kwargs: Any) -> list[Document]:
        docs: list[Document] = []
        max_count = load_kwargs.get("max_count", 1000)
        count = 0
        total_files = 0
        successful_files = 0
        failed_files = 0

        print(f"Starting to process directory: {input_dir}")

        for dirpath, dirnames, filenames in os.walk(input_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if max_count > 0 and count >= max_count:
                    break
                if not filename.endswith(".emlx"):
                    continue

                total_files += 1
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # .emlx: first line is byte length, rest is email
                    lines = content.split("\n", 1)
                    if len(lines) < 2:
                        continue
                    email_content = lines[1]
                    msg = email.message_from_string(email_content)

                    subject = msg.get("Subject", "No Subject")
                    from_addr = msg.get("From", "Unknown")
                    to_addr = msg.get("To", "Unknown")
                    date = msg.get("Date", "Unknown")
                    body = _parse_email_message(msg, self.include_html)

                    if body.strip() or subject != "No Subject":
                        doc_content = (
                            f"\n[File]: {filename}\n"
                            f"[From]: {from_addr}\n"
                            f"[To]: {to_addr}\n"
                            f"[Subject]: {subject}\n"
                            f"[Date]: {date}\n"
                            f"[EMAIL BODY Start]:\n{body}\n"
                        )
                        docs.append(Document(text=doc_content, metadata={}))
                        count += 1
                        successful_files += 1
                        if successful_files <= 3:
                            print(f"Loaded: {filename} - Subject: {subject[:50]}...")

                except Exception as e:
                    failed_files += 1
                    if failed_files <= 5:
                        print(f"Error parsing {filepath}: {e}")
                    continue

        print(f"  Total .emlx: {total_files}  Loaded: {successful_files}  Failed: {failed_files}")
        return docs


class EmlReader(BaseReader):
    """Standard .eml (RFC 2822) email file reader.

    Reads individual .eml files from a directory.
    Unlike .emlx, .eml files have no length prefix — the entire file is the email.
    """

    def __init__(self, include_html: bool = False) -> None:
        self.include_html = include_html

    def load_data(self, input_dir: str, **load_kwargs: Any) -> list[Document]:
        docs: list[Document] = []
        max_count = load_kwargs.get("max_count", 1000)
        count = 0
        total_files = 0
        successful_files = 0
        failed_files = 0

        print(f"Starting to process directory: {input_dir}")

        for dirpath, dirnames, filenames in os.walk(input_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if max_count > 0 and count >= max_count:
                    break
                if not filename.endswith(".eml"):
                    continue

                total_files += 1
                filepath = os.path.join(dirpath, filename)
                try:
                    # .eml: no length prefix — whole file is the email
                    with open(filepath, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    msg = email.message_from_string(content)

                    subject = msg.get("Subject", "No Subject")
                    from_addr = msg.get("From", "Unknown")
                    to_addr = msg.get("To", "Unknown")
                    date = msg.get("Date", "Unknown")
                    body = _parse_email_message(msg, self.include_html)

                    if body.strip() or subject != "No Subject":
                        doc_content = (
                            f"\n[File]: {filename}\n"
                            f"[From]: {from_addr}\n"
                            f"[To]: {to_addr}\n"
                            f"[Subject]: {subject}\n"
                            f"[Date]: {date}\n"
                            f"[EMAIL BODY Start]:\n{body}\n"
                        )
                        docs.append(Document(text=doc_content, metadata={}))
                        count += 1
                        successful_files += 1
                        if successful_files <= 3:
                            print(f"Loaded: {filename} - Subject: {subject[:50]}...")

                except Exception as e:
                    failed_files += 1
                    if failed_files <= 5:
                        print(f"Error parsing {filepath}: {e}")
                    continue

        print(f"  Total .eml: {total_files}  Loaded: {successful_files}  Failed: {failed_files}")
        return docs
