#!/usr/bin/env python3
"""
Sanity check script to verify HNSW index pruning effectiveness.
Tests the difference in file sizes between pruned and non-pruned indices.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import backend packages to trigger plugin registration
import leann_backend_hnsw

from leann.api import LeannBuilder

def create_sample_documents(num_docs=1000):
    """Create sample documents for testing"""
    documents = []
    for i in range(num_docs):
        documents.append(f"Sample document {i} with some random text content for testing purposes.")
    return documents

def build_index(documents, output_dir, is_recompute=True):
    """Build HNSW index with specified recompute setting"""
    index_path = os.path.join(output_dir, "test_index.hnsw")
    
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        M=16,
        efConstruction=100,
        distance_metric="mips",
        is_compact=True,
        is_recompute=is_recompute
    )
    
    for doc in documents:
        builder.add_text(doc)
    
    builder.build_index(index_path)
    
    return index_path

def get_file_size(filepath):
    """Get file size in bytes"""
    return os.path.getsize(filepath)

def main():
    print("🔍 HNSW Pruning Sanity Check")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample documents...")
    documents = create_sample_documents(num_docs=1000)
    print(f"   Number of documents: {len(documents)}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Working in temporary directory: {temp_dir}")
        
        # Build index with pruning (is_recompute=True)
        print("\n🔨 Building index with pruning enabled (is_recompute=True)...")
        pruned_dir = os.path.join(temp_dir, "pruned")
        os.makedirs(pruned_dir, exist_ok=True)
        
        pruned_index_path = build_index(documents, pruned_dir, is_recompute=True)
        # Check what files were actually created
        print(f"   Looking for index files at: {pruned_index_path}")
        import glob
        files = glob.glob(f"{pruned_index_path}*")
        print(f"   Found files: {files}")
        
        # Try to find the actual index file
        if os.path.exists(f"{pruned_index_path}.index"):
            pruned_index_file = f"{pruned_index_path}.index"
        else:
            # Look for any .index file in the directory
            index_files = glob.glob(f"{pruned_dir}/*.index")
            if index_files:
                pruned_index_file = index_files[0]
            else:
                raise FileNotFoundError(f"No .index file found in {pruned_dir}")
        
        pruned_size = get_file_size(pruned_index_file)
        print(f"   ✅ Pruned index built successfully")
        print(f"   📏 Pruned index size: {pruned_size:,} bytes ({pruned_size/1024:.1f} KB)")
        
        # Build index without pruning (is_recompute=False)
        print("\n🔨 Building index without pruning (is_recompute=False)...")
        non_pruned_dir = os.path.join(temp_dir, "non_pruned")
        os.makedirs(non_pruned_dir, exist_ok=True)
        
        non_pruned_index_path = build_index(documents, non_pruned_dir, is_recompute=False)
        # Check what files were actually created
        print(f"   Looking for index files at: {non_pruned_index_path}")
        files = glob.glob(f"{non_pruned_index_path}*")
        print(f"   Found files: {files}")
        
        # Try to find the actual index file
        if os.path.exists(f"{non_pruned_index_path}.index"):
            non_pruned_index_file = f"{non_pruned_index_path}.index"
        else:
            # Look for any .index file in the directory
            index_files = glob.glob(f"{non_pruned_dir}/*.index")
            if index_files:
                non_pruned_index_file = index_files[0]
            else:
                raise FileNotFoundError(f"No .index file found in {non_pruned_dir}")
        
        non_pruned_size = get_file_size(non_pruned_index_file)
        print(f"   ✅ Non-pruned index built successfully")
        print(f"   📏 Non-pruned index size: {non_pruned_size:,} bytes ({non_pruned_size/1024:.1f} KB)")
        
        # Compare sizes
        print("\n📊 Comparison Results:")
        print("=" * 30)
        size_diff = non_pruned_size - pruned_size
        size_ratio = pruned_size / non_pruned_size if non_pruned_size > 0 else 0
        reduction_percent = (1 - size_ratio) * 100
        
        print(f"Non-pruned index: {non_pruned_size:,} bytes ({non_pruned_size/1024:.1f} KB)")
        print(f"Pruned index:     {pruned_size:,} bytes ({pruned_size/1024:.1f} KB)")
        print(f"Size difference:  {size_diff:,} bytes ({size_diff/1024:.1f} KB)")
        print(f"Size ratio:       {size_ratio:.3f}")
        print(f"Size reduction:   {reduction_percent:.1f}%")
        
        # Verify pruning effectiveness
        print("\n🔍 Verification:")
        if size_diff > 0:
            print("   ✅ Pruning is effective - pruned index is smaller")
            if reduction_percent > 10:
                print(f"   ✅ Significant size reduction: {reduction_percent:.1f}%")
            else:
                print(f"   ⚠️  Small size reduction: {reduction_percent:.1f}%")
        else:
            print("   ❌ Pruning appears ineffective - no size reduction")
        
        # Check if passages files were created
        pruned_passages = f"{pruned_index_path}.passages.json"
        non_pruned_passages = f"{non_pruned_index_path}.passages.json"
        
        print(f"\n📄 Passages files:")
        print(f"   Pruned passages file exists: {os.path.exists(pruned_passages)}")
        print(f"   Non-pruned passages file exists: {os.path.exists(non_pruned_passages)}")
        
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)