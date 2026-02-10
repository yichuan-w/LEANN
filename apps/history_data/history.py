import os
import sqlite3
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class ChromeHistoryReader(BaseReader):
    """
    Chrome browser history reader that extracts browsing data from SQLite database.

    Reads Chrome history from the default Chrome profile location and creates documents
    with embedded metadata similar to the email reader structure.
    """

    def __init__(self) -> None:
        """Initialize."""
        pass

    def load_data(self, input_dir: str | None = None, **load_kwargs: Any) -> list[Document]:
        """
        Load Chrome history data from the default Chrome profile location.

        Args:
            input_dir: Not used for Chrome history (kept for compatibility)
            **load_kwargs:
                max_count (int): Maximum amount of history entries to read.
                chrome_profile_path (str): Custom path to Chrome profile directory.
        """
        docs: list[Document] = []
        max_count = load_kwargs.get("max_count", 1000)
        chrome_profile_path = load_kwargs.get("chrome_profile_path", None)

        # Default Chrome profile path on macOS
        if chrome_profile_path is None:
            chrome_profile_path = os.path.expanduser(
                "~/Library/Application Support/Google/Chrome/Default"
            )

        history_db_path = os.path.join(chrome_profile_path, "History")

        if not os.path.exists(history_db_path):
            print(f"Chrome history database not found at: {history_db_path}")
            return docs

        try:
            # Connect to the Chrome history database
            print(f"Connecting to database: {history_db_path}")
            conn = sqlite3.connect(history_db_path)
            cursor = conn.cursor()

            # Query to get browsing history with metadata (removed created_time column)
            query = """
            SELECT
                datetime(last_visit_time/1000000-11644473600,'unixepoch','localtime') as last_visit,
                url,
                title,
                visit_count,
                typed_count,
                hidden
            FROM urls
            ORDER BY last_visit_time DESC
            """

            print(f"Executing query on database: {history_db_path}")
            cursor.execute(query)
            rows = cursor.fetchall()
            print(f"Query returned {len(rows)} rows")

            count = 0
            for row in rows:
                if count >= max_count and max_count > 0:
                    break

                last_visit, url, title, visit_count, typed_count, _hidden = row

                # Create document content with metadata embedded in text
                doc_content = f"""
[Title]: {title}
[URL of the page]: {url}
[Last visited time]: {last_visit}
[Visit times]: {visit_count}
[Typed times]: {typed_count}
"""

                # Create document with embedded metadata
                doc = Document(text=doc_content, metadata={"title": title[0:150]})
                # if len(title) > 150:
                #     print(f"Title is too long: {title}")
                docs.append(doc)
                count += 1

            conn.close()
            print(f"Loaded {len(docs)} Chrome history documents")

        except Exception as e:
            print(f"Error reading Chrome history: {e}")
            # add you may need to close your browser to make the database file available
            # also highlight in red
            print(
                "\033[91mYou may need to close your browser to make the database file available\033[0m"
            )
            return docs

        return docs

    @staticmethod
    def find_chrome_profiles() -> list[Path]:
        """
        Find all Chrome profile directories.

        Returns:
            List of Path objects pointing to Chrome profile directories
        """
        chrome_base_path = Path(os.path.expanduser("~/Library/Application Support/Google/Chrome"))
        profile_dirs = []

        if not chrome_base_path.exists():
            print(f"Chrome directory not found at: {chrome_base_path}")
            return profile_dirs

        # Find all profile directories
        for profile_dir in chrome_base_path.iterdir():
            if profile_dir.is_dir() and profile_dir.name != "System Profile":
                history_path = profile_dir / "History"
                if history_path.exists():
                    profile_dirs.append(profile_dir)
                    print(f"Found Chrome profile: {profile_dir}")

        print(f"Found {len(profile_dirs)} Chrome profiles")
        return profile_dirs

    @staticmethod
    def export_history_to_file(
        output_file: str = "chrome_history_export.txt", max_count: int = 1000
    ):
        """
        Export Chrome history to a text file using the same SQL query format.

        Args:
            output_file: Path to the output file
            max_count: Maximum number of entries to export
        """
        chrome_profile_path = os.path.expanduser(
            "~/Library/Application Support/Google/Chrome/Default"
        )
        history_db_path = os.path.join(chrome_profile_path, "History")

        if not os.path.exists(history_db_path):
            print(f"Chrome history database not found at: {history_db_path}")
            return

        try:
            conn = sqlite3.connect(history_db_path)
            cursor = conn.cursor()

            query = """
            SELECT
                datetime(last_visit_time/1000000-11644473600,'unixepoch','localtime') as last_visit,
                url,
                title,
                visit_count,
                typed_count,
                hidden
            FROM urls
            ORDER BY last_visit_time DESC
            LIMIT ?
            """

            cursor.execute(query, (max_count,))
            rows = cursor.fetchall()

            with open(output_file, "w", encoding="utf-8") as f:
                for row in rows:
                    last_visit, url, title, visit_count, typed_count, hidden = row
                    f.write(
                        f"{last_visit}\t{url}\t{title}\t{visit_count}\t{typed_count}\t{hidden}\n"
                    )

            conn.close()
            print(f"Exported {len(rows)} history entries to {output_file}")

        except Exception as e:
            print(f"Error exporting Chrome history: {e}")
