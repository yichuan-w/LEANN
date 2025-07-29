#!/usr/bin/env python3
"""
This script has been replaced by document_rag.py with a unified interface.
This file is kept for backward compatibility.
"""

import sys
import os

print("=" * 70)
print("NOTICE: This script has been replaced!")
print("=" * 70)
print("\nThe examples have been refactored with a unified interface.")
print("Please use the new script instead:\n")
print("  python examples/document_rag.py")
print("\nThe new script provides:")
print("  ✓ Consistent parameters across all examples")
print("  ✓ Better error handling")
print("  ✓ Interactive mode support")
print("  ✓ More customization options")
print("\nExample usage:")
print('  python examples/document_rag.py --query "What are the main techniques?"')
print("  python examples/document_rag.py  # For interactive mode")
print("\nSee README.md for full documentation.")
print("=" * 70)

# If user passed arguments, show how to use them with new script
if len(sys.argv) > 1:
    print("\nTo use your arguments with the new script:")
    print(f"  python examples/document_rag.py {' '.join(sys.argv[1:])}")

sys.exit(1)
