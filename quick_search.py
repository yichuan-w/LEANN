#!/usr/bin/env python3
"""
Quick search script for LEANN - faster and more reliable than CLI
Usage: python quick_search.py "your query"
"""
from leann import LeannSearcher
import sys

# Index path
index_path = ".leann/indexes/my-docs/documents.leann"

# Get query from command line or use default
query = sys.argv[1] if len(sys.argv) > 1 else "project X"
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

print(f"🔍 Searching for: '{query}'")
print(f"📚 Index: my-docs")
print(f"📊 Top {top_k} results\n")
print("=" * 80)

try:
    searcher = LeannSearcher(index_path)
    results = searcher.search(query, top_k=top_k)
    
    if not results:
        print("No results found.")
    else:
        print(f"Found {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. Score: {r.score:.3f}")
            # Show first 200 characters
            text_preview = r.text[:200].replace('\n', ' ')
            if len(r.text) > 200:
                text_preview += "..."
            print(f"   {text_preview}")
            print()
            
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure:")
    print("1. You're in the LEANN directory")
    print("2. The index 'my-docs' exists (run: leann list)")
    print("3. The virtual environment is activated")
    sys.exit(1)
