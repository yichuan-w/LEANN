#!/usr/bin/env python3
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from leann import LeannSearcher

INDEX_PATH = str(Path("./").resolve() / "demo.leann")


class TimeParser:
    def __init__(self):
        # Main pattern: captures optional fuzzy modifier, number, unit, and optional "ago"
        self.pattern = r"(?:(around|about|roughly|approximately)\s+)?(\d+)\s+(hour|day|week|month|year)s?(?:\s+ago)?"

        # Compile for performance
        self.regex = re.compile(self.pattern, re.IGNORECASE)

        # Stop words to remove before regex parsing
        self.stop_words = {
            "in",
            "at",
            "of",
            "by",
            "as",
            "me",
            "the",
            "a",
            "an",
            "and",
            "any",
            "find",
            "search",
            "list",
            "ago",
            "back",
            "past",
            "earlier",
        }

    def clean_text(self, text):
        """Remove stop words from text"""
        words = text.split()
        cleaned = " ".join(word for word in words if word.lower() not in self.stop_words)
        return cleaned

    def parse(self, text):
        """Extract all time expressions from text"""
        # Clean text first
        cleaned_text = self.clean_text(text)

        matches = []
        for match in self.regex.finditer(cleaned_text):
            fuzzy = match.group(1)  # "around", "about", etc.
            number = int(match.group(2))
            unit = match.group(3).lower()

            matches.append(
                {
                    "full_match": match.group(0),
                    "fuzzy": bool(fuzzy),
                    "number": number,
                    "unit": unit,
                    "range": self.calculate_range(number, unit, bool(fuzzy)),
                }
            )

        return matches

    def calculate_range(self, number, unit, is_fuzzy):
        """Convert to actual datetime range and return ISO format strings"""
        units = {
            "hour": timedelta(hours=number),
            "day": timedelta(days=number),
            "week": timedelta(weeks=number),
            "month": timedelta(days=number * 30),
            "year": timedelta(days=number * 365),
        }

        delta = units[unit]
        now = datetime.now()
        target = now - delta

        if is_fuzzy:
            buffer = delta * 0.2  # 20% buffer for fuzzy
            start = (target - buffer).isoformat()
            end = (target + buffer).isoformat()
        else:
            start = target.isoformat()
            end = now.isoformat()

        return (start, end)


def search_files(query, top_k=15):
    """Search the index and return results"""
    # Parse time expressions
    parser = TimeParser()
    time_matches = parser.parse(query)

    # Remove time expressions from query for semantic search
    clean_query = query
    if time_matches:
        for match in time_matches:
            clean_query = clean_query.replace(match["full_match"], "").strip()

    # Check if clean_query is less than 4 characters
    if len(clean_query) < 4:
        print("Error: add more input for accurate results.")
        return

    # Single query to vector DB
    searcher = LeannSearcher(INDEX_PATH)
    results = searcher.search(
        clean_query if clean_query else query, top_k=top_k, recompute_embeddings=False
    )

    # Filter by time if time expression found
    if time_matches:
        time_range = time_matches[0]["range"]  # Use first time expression
        start_time, end_time = time_range

        filtered_results = []
        for result in results:
            # Access metadata attribute directly (not .get())
            metadata = result.metadata if hasattr(result, "metadata") else {}

            if metadata:
                # Check modification date first, fall back to creation date
                date_str = metadata.get("modification_date") or metadata.get("creation_date")

                if date_str:
                    # Convert strings to datetime objects for proper comparison
                    try:
                        file_date = datetime.fromisoformat(date_str)
                        start_dt = datetime.fromisoformat(start_time)
                        end_dt = datetime.fromisoformat(end_time)

                        # Compare dates properly
                        if start_dt <= file_date <= end_dt:
                            filtered_results.append(result)
                    except (ValueError, TypeError):
                        # Handle invalid date formats
                        print(f"Warning: Invalid date format in metadata: {date_str}")
                        continue

        results = filtered_results

    # Print results
    print(f"\nSearch results for: '{query}'")
    if time_matches:
        print(
            f"Time filter: {time_matches[0]['number']} {time_matches[0]['unit']}(s) {'(fuzzy)' if time_matches[0]['fuzzy'] else ''}"
        )
        print(
            f"Date range: {time_matches[0]['range'][0][:10]} to {time_matches[0]['range'][1][:10]}"
        )
    print("-" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.4f}")
        print(f"Content: {result.text}")

        # Show metadata if present
        metadata = result.metadata if hasattr(result, "metadata") else None
        if metadata:
            if "creation_date" in metadata:
                print(f"Created: {metadata['creation_date']}")
            if "modification_date" in metadata:
                print(f"Modified: {metadata['modification_date']}")
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python search_index.py "<search query>" [top_k]')
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    search_files(query, top_k)
