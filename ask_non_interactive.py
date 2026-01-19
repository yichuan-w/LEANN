#!/usr/bin/env python3
"""
Non-interactive ask script - avoids permission issues with readline history
Usage: python ask_non_interactive.py "your question"
"""
from leann import LeannChat
import sys
import os

# Index path
index_path = ".leann/indexes/my-docs/documents.leann"

# Get question from command line
if len(sys.argv) < 2:
    print("Usage: python ask_non_interactive.py 'your question'")
    print("Example: python ask_non_interactive.py 'What did I write about project X?'")
    sys.exit(1)

question = sys.argv[1]
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 20

print(f"💬 Asking: '{question}'")
print(f"📚 Index: my-docs")
print(f"🤖 Model: gemma3:4b (Ollama)")
print("=" * 80)
print()

try:
    # Use Ollama with gemma3:4b
    chat = LeannChat(
        index_path,
        llm_config={"type": "ollama", "model": "gemma3:4b"}
    )
    
    print("⏳ Processing... (this may take 30-60 seconds)")
    response = chat.ask(question, top_k=top_k)
    
    print("\n" + "=" * 80)
    print("📝 Answer:")
    print("=" * 80)
    print(response)
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure Ollama is running: ollama list")
    print("2. Make sure gemma3:4b is installed: ollama pull gemma3:4b")
    print("3. Check the index exists: leann list")
    sys.exit(1)
