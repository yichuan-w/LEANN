#!/usr/bin/env python3
"""
Spotlight Metadata Dumper for Vector DB
Extracts only essential metadata for semantic search embeddings
Output is optimized for vector database storage with minimal fields
"""

import json
import sys
from datetime import datetime

# Check platform before importing macOS-specific modules
if sys.platform != "darwin":
    print("This script requires macOS (uses Spotlight)")
    sys.exit(1)

from Foundation import NSDate, NSMetadataQuery, NSPredicate, NSRunLoop

# EDIT THIS LIST: Add or remove folders to search
# Can be either:
# - Folder names relative to home directory (e.g., "Desktop", "Downloads")
# - Absolute paths (e.g., "/Applications", "/System/Library")
SEARCH_FOLDERS = [
    "Desktop",
    "Downloads",
    "Documents",
    "Music",
    "Pictures",
    "Movies",
    # "Library",  # Uncomment to include
    # "/Applications",  # Absolute path example
    # "Code/Projects",  # Subfolder example
    # Add any other folders here
]


def convert_to_serializable(obj):
    """Convert NS objects to Python serializable types"""
    if obj is None:
        return None

    # Handle NSDate
    if hasattr(obj, "timeIntervalSince1970"):
        return datetime.fromtimestamp(obj.timeIntervalSince1970()).isoformat()

    # Handle NSArray
    if hasattr(obj, "count") and hasattr(obj, "objectAtIndex_"):
        return [convert_to_serializable(obj.objectAtIndex_(i)) for i in range(obj.count())]

    # Convert to string
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def dump_spotlight_data(max_items=10, output_file="spotlight_dump.json"):
    """
    Dump Spotlight data using public.item predicate
    """
    # Build full paths from SEARCH_FOLDERS
    import os

    home_dir = os.path.expanduser("~")
    search_paths = []

    print("Search locations:")
    for folder in SEARCH_FOLDERS:
        # Check if it's an absolute path or relative
        if folder.startswith("/"):
            full_path = folder
        else:
            full_path = os.path.join(home_dir, folder)

        if os.path.exists(full_path):
            search_paths.append(full_path)
            print(f"  ✓ {full_path}")
        else:
            print(f"  ✗ {full_path} (not found)")

    if not search_paths:
        print("No valid search paths found!")
        return []

    print(f"\nDumping {max_items} items from Spotlight (public.item)...")

    # Create query with public.item predicate
    query = NSMetadataQuery.alloc().init()
    predicate = NSPredicate.predicateWithFormat_("kMDItemContentTypeTree CONTAINS 'public.item'")
    query.setPredicate_(predicate)

    # Set search scopes to our specific folders
    query.setSearchScopes_(search_paths)

    print("Starting query...")
    query.startQuery()

    # Wait for gathering to complete
    run_loop = NSRunLoop.currentRunLoop()
    print("Gathering results...")

    # Let it gather for a few seconds
    for i in range(50):  # 5 seconds max
        run_loop.runMode_beforeDate_(
            "NSDefaultRunLoopMode", NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )
        # Check gathering status periodically
        if i % 10 == 0:
            current_count = query.resultCount()
            if current_count > 0:
                print(f"  Found {current_count} items so far...")

    # Continue while still gathering (up to 2 more seconds)
    timeout = NSDate.dateWithTimeIntervalSinceNow_(2.0)
    while query.isGathering() and timeout.timeIntervalSinceNow() > 0:
        run_loop.runMode_beforeDate_(
            "NSDefaultRunLoopMode", NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )

    query.stopQuery()

    total_results = query.resultCount()
    print(f"Found {total_results} total items")

    if total_results == 0:
        print("No results found")
        return []

    # Process items
    items_to_process = min(total_results, max_items)
    results = []

    # ONLY relevant attributes for vector embeddings
    # These provide essential context for semantic search without bloat
    attributes = [
        "kMDItemPath",  # Full path for file retrieval
        "kMDItemFSName",  # Filename for display & embedding
        "kMDItemFSSize",  # Size for filtering/ranking
        "kMDItemContentType",  # File type for categorization
        "kMDItemKind",  # Human-readable type for embedding
        "kMDItemFSCreationDate",  # Temporal context
        "kMDItemFSContentChangeDate",  # Recency for ranking
    ]

    print(f"Processing {items_to_process} items...")

    for i in range(items_to_process):
        try:
            item = query.resultAtIndex_(i)
            metadata = {}

            # Extract ONLY the relevant attributes
            for attr in attributes:
                try:
                    value = item.valueForAttribute_(attr)
                    if value is not None:
                        # Keep the attribute name clean (remove kMDItem prefix for cleaner JSON)
                        clean_key = attr.replace("kMDItem", "").replace("FS", "")
                        metadata[clean_key] = convert_to_serializable(value)
                except (AttributeError, ValueError, TypeError):
                    continue

            # Only add if we have at least a path
            if metadata.get("Path"):
                results.append(metadata)

        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(results)} items to {output_file}")

    # Show summary
    print("\nSample items:")
    import os

    home_dir = os.path.expanduser("~")

    for i, item in enumerate(results[:3]):
        print(f"\n[Item {i + 1}]")
        print(f"  Path: {item.get('Path', 'N/A')}")
        print(f"  Name: {item.get('Name', 'N/A')}")
        print(f"  Type: {item.get('ContentType', 'N/A')}")
        print(f"  Kind: {item.get('Kind', 'N/A')}")

        # Handle size properly
        size = item.get("Size")
        if size:
            try:
                size_int = int(size)
                if size_int > 1024 * 1024:
                    print(f"  Size: {size_int / (1024 * 1024):.2f} MB")
                elif size_int > 1024:
                    print(f"  Size: {size_int / 1024:.2f} KB")
                else:
                    print(f"  Size: {size_int} bytes")
            except (ValueError, TypeError):
                print(f"  Size: {size}")

        # Show dates
        if "CreationDate" in item:
            print(f"  Created: {item['CreationDate']}")
        if "ContentChangeDate" in item:
            print(f"  Modified: {item['ContentChangeDate']}")

    # Count by type
    type_counts = {}
    for item in results:
        content_type = item.get("ContentType", "unknown")
        type_counts[content_type] = type_counts.get(content_type, 0) + 1

    print(f"\nTotal items saved: {len(results)}")

    if type_counts:
        print("\nTop content types:")
        for ct, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {ct}: {count} items")

    # Count by folder
    folder_counts = {}
    for item in results:
        path = item.get("Path", "")
        for folder in SEARCH_FOLDERS:
            # Build the full folder path
            if folder.startswith("/"):
                folder_path = folder
            else:
                folder_path = os.path.join(home_dir, folder)

            if path.startswith(folder_path):
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
                break

    if folder_counts:
        print("\nItems by location:")
        for folder, count in sorted(folder_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {folder}: {count} items")

    return results


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        try:
            max_items = int(sys.argv[1])
        except ValueError:
            print("Usage: python spot.py [number_of_items]")
            print("Default: 10 items")
            sys.exit(1)
    else:
        max_items = 10

    output_file = sys.argv[2] if len(sys.argv) > 2 else "spotlight_dump.json"

    # Run dump
    dump_spotlight_data(max_items=max_items, output_file=output_file)


if __name__ == "__main__":
    main()
