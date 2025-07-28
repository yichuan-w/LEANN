"""
Comparison between Sentence Transformers and OpenAI embeddings

This example shows how different embedding models handle complex queries
and demonstrates the differences between local and API-based embeddings.
"""

import numpy as np
from leann.embedding_compute import compute_embeddings

# OpenAI API key should be set as environment variable
# export OPENAI_API_KEY="your-api-key-here"

# Test data
conference_text = "[Title]: COLING 2025 Conference\n[URL]: https://coling2025.org/"
browser_text = "[Title]: Browser Use Tool\n[URL]: https://github.com/browser-use"

# Two queries with same intent but different wording
query1 = "Tell me my browser history about some conference i often visit"
query2 = "browser history about conference I often visit"

texts = [query1, query2, conference_text, browser_text]


def cosine_similarity(a, b):
    return np.dot(a, b)  # Already normalized


def analyze_embeddings(embeddings, model_name):
    print(f"\n=== {model_name} Results ===")

    # Results for Query 1
    sim1_conf = cosine_similarity(embeddings[0], embeddings[2])
    sim1_browser = cosine_similarity(embeddings[0], embeddings[3])

    print(f"Query 1: '{query1}'")
    print(f"  → Conference similarity: {sim1_conf:.4f} {'✓' if sim1_conf > sim1_browser else ''}")
    print(
        f"  → Browser similarity:    {sim1_browser:.4f} {'✓' if sim1_browser > sim1_conf else ''}"
    )
    print(f"  Winner: {'Conference' if sim1_conf > sim1_browser else 'Browser'}")

    # Results for Query 2
    sim2_conf = cosine_similarity(embeddings[1], embeddings[2])
    sim2_browser = cosine_similarity(embeddings[1], embeddings[3])

    print(f"\nQuery 2: '{query2}'")
    print(f"  → Conference similarity: {sim2_conf:.4f} {'✓' if sim2_conf > sim2_browser else ''}")
    print(
        f"  → Browser similarity:    {sim2_browser:.4f} {'✓' if sim2_browser > sim2_conf else ''}"
    )
    print(f"  Winner: {'Conference' if sim2_conf > sim2_browser else 'Browser'}")

    # Show the impact
    print("\n=== Impact Analysis ===")
    print(f"Conference similarity change: {sim2_conf - sim1_conf:+.4f}")
    print(f"Browser similarity change:    {sim2_browser - sim1_browser:+.4f}")

    if sim1_conf > sim1_browser and sim2_browser > sim2_conf:
        print("❌ FLIP: Adding 'browser history' flips winner from Conference to Browser!")
    elif sim1_conf > sim1_browser and sim2_conf > sim2_browser:
        print("✅ STABLE: Conference remains winner in both queries")
    elif sim1_browser > sim1_conf and sim2_browser > sim2_conf:
        print("✅ STABLE: Browser remains winner in both queries")
    else:
        print("🔄 MIXED: Results vary between queries")

    return {
        "query1_conf": sim1_conf,
        "query1_browser": sim1_browser,
        "query2_conf": sim2_conf,
        "query2_browser": sim2_browser,
    }


# Test Sentence Transformers
print("Testing Sentence Transformers (facebook/contriever)...")
try:
    st_embeddings = compute_embeddings(texts, "facebook/contriever", mode="sentence-transformers")
    st_results = analyze_embeddings(st_embeddings, "Sentence Transformers (facebook/contriever)")
except Exception as e:
    print(f"❌ Sentence Transformers failed: {e}")
    st_results = None

# Test OpenAI
print("\n" + "=" * 60)
print("Testing OpenAI (text-embedding-3-small)...")
try:
    openai_embeddings = compute_embeddings(texts, "text-embedding-3-small", mode="openai")
    openai_results = analyze_embeddings(openai_embeddings, "OpenAI (text-embedding-3-small)")
except Exception as e:
    print(f"❌ OpenAI failed: {e}")
    openai_results = None

# Compare results
if st_results and openai_results:
    print("\n" + "=" * 60)
    print("=== COMPARISON SUMMARY ===")
