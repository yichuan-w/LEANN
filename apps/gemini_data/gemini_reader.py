import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GeminiReader:
    """Reader for Gemini CLI history files."""

    def __init__(self):
        pass

    def load_data(self, history_dir: str, max_count: int = -1) -> list[dict[str, Any]]:
        """
        Load data from Gemini history directory.

        Args:
            history_dir: Path to .gemini directory
            max_count: Max number of conversations to load

        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        history_path = Path(history_dir).expanduser()
        if not history_path.exists():
            print(f"Gemini history directory not found: {history_path}")
            return []

        documents = []

        # 1. Load Memory (GEMINI.md)
        memory_file = history_path / "GEMINI.md"
        if memory_file.exists():
            try:
                text = memory_file.read_text(encoding="utf-8")
                if text.strip():
                    documents.append(
                        {
                            "text": f"Gemini Memory:\n{text}",
                            "metadata": {"source": str(memory_file), "type": "memory"},
                        }
                    )
            except Exception as e:
                print(f"Error reading memory file: {e}")

        # 2. Find Session Files
        # Legacy JSON sessions
        session_files = list(history_path.glob("session-*.json"))
        # New JSONL sessions
        session_files.extend(list(history_path.glob("session-*.jsonl")))
        # Checkpoints
        session_files.extend(list(history_path.glob("checkpoint-*.json")))

        # Sort by modification time (newest first)
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        print(f"Found {len(session_files)} session files.")

        count = 0
        for file_path in session_files:
            if max_count > 0 and count >= max_count:
                break

            try:
                content = ""
                if file_path.suffix == ".jsonl":
                    content = self._parse_jsonl_session(file_path)
                elif file_path.suffix == ".json":
                    content = self._parse_json_session(file_path)

                if content:
                    documents.append(
                        {
                            "text": content,
                            "metadata": {
                                "source": str(file_path),
                                "type": "session",
                                "filename": file_path.name,
                            },
                        }
                    )
                    count += 1
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

        print(f"Successfully loaded {len(documents)} items from Gemini history.")
        return documents

    def _parse_json_session(self, file_path: Path) -> str:
        """Parse legacy JSON session file."""
        data = json.loads(file_path.read_text(encoding="utf-8"))

        # Handle dict format (standard session)
        messages = []
        if isinstance(data, dict):
            # Check for 'messages' key (standard format)
            if "messages" in data:
                for msg in data["messages"]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if content:
                        messages.append(f"{role.upper()}: {content}")
            # Check for 'parts' key (checkpoint format sometimes)
            elif "parts" in data:
                messages.append(f"Saved Session Content: {data['parts']}")

        # Handle list format (some older array-based sessions)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    role = item.get("role", "unknown")
                    content = item.get("content", "") or item.get("parts", "")
                    if content:
                        messages.append(f"{role.upper()}: {content}")

        if not messages:
            return ""

        return f"File: {file_path.name}\n\n" + "\n\n".join(messages)

    def _parse_jsonl_session(self, file_path: Path) -> str:
        """Parse JSONL session file."""
        messages = []
        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Skip metadata lines if they don't have content
                        if "role" in data and "content" in data:
                            messages.append(f"{data['role'].upper()}: {data['content']}")
                        elif "parts" in data:  # sometimes parts is used
                            messages.append(
                                f"{data.get('role', 'unknown').upper()}: {data['parts']}"
                            )
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return ""

        if not messages:
            return ""

        return f"File: {file_path.name}\n\n" + "\n\n".join(messages)
