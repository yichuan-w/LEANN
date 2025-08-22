#!/usr/bin/env python3
"""
Spoiler-Free Book RAG Example using LEANN Metadata Filtering

This example demonstrates how to use LEANN's metadata filtering to create
a spoiler-free book RAG system where users can search for information
up to a specific chapter they've read.

Usage:
    python spoiler_free_book_rag.py
"""

import os
import sys
from typing import Any, Optional

# Add LEANN to path (adjust path as needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../packages/leann-core/src"))

from leann.api import LeannBuilder, LeannSearcher


def chunk_book_with_metadata(book_title: str = "Sample Book") -> list[dict[str, Any]]:
    """
    Create sample book chunks with metadata for demonstration.

    In a real implementation, this would parse actual book files (epub, txt, etc.)
    and extract chapter boundaries, character mentions, etc.

    Args:
        book_title: Title of the book

    Returns:
        List of chunk dictionaries with text and metadata
    """
    # Sample book chunks with metadata
    # In practice, you'd use proper text processing libraries

    sample_chunks = [
        {
            "text": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.",
            "metadata": {
                "book": book_title,
                "chapter": 1,
                "page": 1,
                "characters": ["Alice", "Sister"],
                "themes": ["boredom", "curiosity"],
                "location": "riverbank",
            },
        },
        {
            "text": "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
            "metadata": {
                "book": book_title,
                "chapter": 1,
                "page": 2,
                "characters": ["Alice", "White Rabbit"],
                "themes": ["decision", "surprise", "magic"],
                "location": "riverbank",
            },
        },
        {
            "text": "Alice found herself falling down a very deep well. Either the well was very deep, or she fell very slowly, for she had plenty of time as she fell to look about her and to wonder what was going to happen next.",
            "metadata": {
                "book": book_title,
                "chapter": 2,
                "page": 15,
                "characters": ["Alice"],
                "themes": ["falling", "wonder", "transformation"],
                "location": "rabbit hole",
            },
        },
        {
            "text": "Alice meets the Cheshire Cat, who tells her that everyone in Wonderland is mad, including Alice herself.",
            "metadata": {
                "book": book_title,
                "chapter": 6,
                "page": 85,
                "characters": ["Alice", "Cheshire Cat"],
                "themes": ["madness", "philosophy", "identity"],
                "location": "Duchess's house",
            },
        },
        {
            "text": "At the Queen's croquet ground, Alice witnesses the absurd trial that reveals the arbitrary nature of Wonderland's justice system.",
            "metadata": {
                "book": book_title,
                "chapter": 8,
                "page": 120,
                "characters": ["Alice", "Queen of Hearts", "King of Hearts"],
                "themes": ["justice", "absurdity", "authority"],
                "location": "Queen's court",
            },
        },
        {
            "text": "Alice realizes that Wonderland was all a dream, even the Rabbit, as she wakes up on the riverbank next to her sister.",
            "metadata": {
                "book": book_title,
                "chapter": 12,
                "page": 180,
                "characters": ["Alice", "Sister", "Rabbit"],
                "themes": ["revelation", "reality", "growth"],
                "location": "riverbank",
            },
        },
    ]

    return sample_chunks


def build_spoiler_free_index(book_chunks: list[dict[str, Any]], index_name: str) -> str:
    """
    Build a LEANN index with book chunks that include spoiler metadata.

    Args:
        book_chunks: List of book chunks with metadata
        index_name: Name for the index

    Returns:
        Path to the built index
    """
    print(f"📚 Building spoiler-free book index: {index_name}")

    # Initialize LEANN builder
    builder = LeannBuilder(
        backend_name="hnsw", embedding_model="text-embedding-3-small", embedding_mode="openai"
    )

    # Add each chunk with its metadata
    for chunk in book_chunks:
        builder.add_text(text=chunk["text"], metadata=chunk["metadata"])

    # Build the index
    index_path = f"{index_name}_book_index"
    builder.build_index(index_path)

    print(f"✅ Index built successfully: {index_path}")
    return index_path


def spoiler_free_search(
    index_path: str,
    query: str,
    max_chapter: int,
    character_filter: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Perform a spoiler-free search on the book index.

    Args:
        index_path: Path to the LEANN index
        query: Search query
        max_chapter: Maximum chapter number to include
        character_filter: Optional list of characters to focus on

    Returns:
        List of search results safe for the reader
    """
    print(f"🔍 Searching: '{query}' (up to chapter {max_chapter})")

    searcher = LeannSearcher(index_path)

    metadata_filters = {"chapter": {"<=": max_chapter}}

    if character_filter:
        metadata_filters["characters"] = {"contains": character_filter[0]}

    results = searcher.search(query=query, top_k=10, metadata_filters=metadata_filters)

    return results


def demo_spoiler_free_rag():
    """
    Demonstrate the spoiler-free book RAG system.
    """
    print("🎭 Spoiler-Free Book RAG Demo")
    print("=" * 40)

    # Step 1: Prepare book data
    book_title = "Alice's Adventures in Wonderland"
    book_chunks = chunk_book_with_metadata(book_title)

    print(f"📖 Loaded {len(book_chunks)} chunks from '{book_title}'")

    # Step 2: Build the index (in practice, this would be done once)
    try:
        index_path = build_spoiler_free_index(book_chunks, "alice_wonderland")
    except Exception as e:
        print(f"❌ Failed to build index (likely missing dependencies): {e}")
        print(
            "💡 This demo shows the filtering logic - actual indexing requires LEANN dependencies"
        )
        return

    # Step 3: Demonstrate various spoiler-free searches
    search_scenarios = [
        {
            "description": "Reader who has only read Chapter 1",
            "query": "What can you tell me about the rabbit?",
            "max_chapter": 1,
        },
        {
            "description": "Reader who has read up to Chapter 5",
            "query": "Tell me about Alice's adventures",
            "max_chapter": 5,
        },
        {
            "description": "Reader who has read most of the book",
            "query": "What does the Cheshire Cat represent?",
            "max_chapter": 10,
        },
        {
            "description": "Reader who has read the whole book",
            "query": "What can you tell me about the rabbit?",
            "max_chapter": 12,
        },
    ]

    for scenario in search_scenarios:
        print(f"\n📚 Scenario: {scenario['description']}")
        print(f"   Query: {scenario['query']}")

        try:
            results = spoiler_free_search(
                index_path=index_path,
                query=scenario["query"],
                max_chapter=scenario["max_chapter"],
            )

            print(f"   📄 Found {len(results)} results:")
            for i, result in enumerate(results[:3], 1):  # Show top 3
                chapter = result.metadata.get("chapter", "?")
                location = result.metadata.get("location", "?")
                print(f"      {i}. Chapter {chapter} ({location}): {result.text[:80]}...")

        except Exception as e:
            print(f"   ❌ Search failed: {e}")


if __name__ == "__main__":
    print("📚 LEANN Spoiler-Free Book RAG Example")
    print("=====================================")

    try:
        demo_spoiler_free_rag()
    except ImportError as e:
        print(f"❌ Cannot run demo due to missing dependencies: {e}")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
