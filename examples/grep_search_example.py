"""
Grep Search Example

Shows how to use grep-based text search instead of semantic search.
Useful when you need exact text matches rather than meaning-based results.
"""

from leann import LeannSearcher

# Load your index
searcher = LeannSearcher("my-documents.leann")

# Regular semantic search
print("=== Semantic Search ===")
results = searcher.search("machine learning algorithms", top_k=3)
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text[:80]}...")
    print()

# Grep-based search for exact text matches
print("=== Grep Search ===")
results = searcher.search("def train_model", top_k=3, use_grep=True)
for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.text[:80]}...")
    print()

# Find specific error messages
error_results = searcher.search("FileNotFoundError", use_grep=True)
print(f"Found {len(error_results)} files mentioning FileNotFoundError")

# Search for function definitions
func_results = searcher.search("class SearchResult", use_grep=True, top_k=5)
print(f"Found {len(func_results)} class definitions")
