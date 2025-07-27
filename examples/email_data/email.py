"""
Mbox parser.

Contains simple parser for mbox files.

"""

import logging
from pathlib import Path
from typing import Any

from fsspec import AbstractFileSystem
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class MboxReader(BaseReader):
    """
    Mbox parser.

    Extract messages from mailbox files.
    Returns string including date, subject, sender, receiver and
    content for each message.

    """

    DEFAULT_MESSAGE_FORMAT: str = (
        "Date: {_date}\nFrom: {_from}\nTo: {_to}\nSubject: {_subject}\nContent: {_content}"
    )

    def __init__(
        self,
        *args: Any,
        max_count: int = 0,
        message_format: str = DEFAULT_MESSAGE_FORMAT,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            from bs4 import BeautifulSoup  # noqa
        except ImportError:
            raise ImportError("`beautifulsoup4` package not found: `pip install beautifulsoup4`")

        super().__init__(*args, **kwargs)
        self.max_count = max_count
        self.message_format = message_format

    def load_data(
        self,
        file: Path,
        extra_info: dict | None = None,
        fs: AbstractFileSystem | None = None,
    ) -> list[Document]:
        """Parse file into string."""
        # Import required libraries
        import mailbox
        from email.parser import BytesParser
        from email.policy import default

        from bs4 import BeautifulSoup

        if fs:
            logger.warning(
                "fs was specified but MboxReader doesn't support loading "
                "from fsspec filesystems. Will load from local filesystem instead."
            )

        i = 0
        results: list[str] = []
        # Load file using mailbox
        bytes_parser = BytesParser(policy=default).parse
        mbox = mailbox.mbox(file, factory=bytes_parser)  # type: ignore

        # Iterate through all messages
        for _, _msg in enumerate(mbox):
            try:
                msg: mailbox.mboxMessage = _msg
                # Parse multipart messages
                if msg.is_multipart():
                    for part in msg.walk():
                        ctype = part.get_content_type()
                        cdispo = str(part.get("Content-Disposition"))
                        if "attachment" in cdispo:
                            print(f"Attachment found: {part.get_filename()}")
                        if ctype == "text/plain" and "attachment" not in cdispo:
                            content = part.get_payload(decode=True)  # decode
                            break
                # Get plain message payload for non-multipart messages
                else:
                    content = msg.get_payload(decode=True)

                # Parse message HTML content and remove unneeded whitespace
                soup = BeautifulSoup(content)
                stripped_content = " ".join(soup.get_text().split())
                # Format message to include date, sender, receiver and subject
                msg_string = self.message_format.format(
                    _date=msg["date"],
                    _from=msg["from"],
                    _to=msg["to"],
                    _subject=msg["subject"],
                    _content=stripped_content,
                )
                # Add message string to results
                results.append(msg_string)
            except Exception as e:
                logger.warning(f"Failed to parse message:\n{_msg}\n with exception {e}")

            # Increment counter and return if max count is met
            i += 1
            if self.max_count > 0 and i >= self.max_count:
                break

        return [Document(text=result, metadata=extra_info or {}) for result in results]


class EmlxMboxReader(MboxReader):
    """
    EmlxMboxReader - Modified MboxReader that handles directories of .emlx files.

    Extends MboxReader to work with Apple Mail's .emlx format by:
    1. Reading .emlx files from a directory
    2. Converting them to mbox format in memory
    3. Using the parent MboxReader's parsing logic
    """

    def load_data(
        self,
        directory: Path,
        extra_info: dict | None = None,
        fs: AbstractFileSystem | None = None,
    ) -> list[Document]:
        """Parse .emlx files from directory into strings using MboxReader logic."""
        import os
        import tempfile

        if fs:
            logger.warning(
                "fs was specified but EmlxMboxReader doesn't support loading "
                "from fsspec filesystems. Will load from local filesystem instead."
            )

        # Find all .emlx files in the directory
        emlx_files = list(directory.glob("*.emlx"))
        logger.info(f"Found {len(emlx_files)} .emlx files in {directory}")

        if not emlx_files:
            logger.warning(f"No .emlx files found in {directory}")
            return []

        # Create a temporary mbox file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mbox", delete=False) as temp_mbox:
            temp_mbox_path = temp_mbox.name

            # Convert .emlx files to mbox format
            for emlx_file in emlx_files:
                try:
                    # Read the .emlx file
                    with open(emlx_file, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # .emlx format: first line is length, rest is email content
                    lines = content.split("\n", 1)
                    if len(lines) >= 2:
                        email_content = lines[1]  # Skip the length line

                        # Write to mbox format (each message starts with "From " and ends with blank line)
                        temp_mbox.write(f"From {emlx_file.name} {email_content}\n\n")

                except Exception as e:
                    logger.warning(f"Failed to process {emlx_file}: {e}")
                    continue

            # Close the temporary file so MboxReader can read it
            temp_mbox.close()

            try:
                # Use the parent MboxReader's logic to parse the mbox file
                return super().load_data(Path(temp_mbox_path), extra_info, fs)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_mbox_path)
                except OSError:
                    pass
