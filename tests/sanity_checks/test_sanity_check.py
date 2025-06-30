#!/usr/bin/env python3
"""
Sanity check script for Leann DiskANN backend
Tests different distance functions and embedding models
"""

import os
import numpy as np
from pathlib import Path
import shutil
import time

# 导入后端包以触发插件注册
import sys
sys.path.append('packages/leann-core/src')
sys.path.append('packages/leann-backend-diskann')
sys.path.append('packages/leann-backend-hnsw')

try:
    import leann_backend_diskann
    import leann_backend_hnsw
    print("INFO: Backend packages imported successfully.")
except ImportError as e:
    print(f"WARNING: Could not import backend packages. Error: {e}")

# 从 leann-core 导入上层 API
from leann.api import LeannBuilder, LeannSearcher

def test_distance_functions():
    """测试不同的距离函数"""
    print("\n=== 测试不同距离函数 ===")
    
    # 测试数据
    documents = [
        "Machine learning is a powerful technology",
        "Deep learning uses neural networks", 
        "Artificial intelligence transforms industries"
    ]
    
    distance_functions = ["mips", "l2", "cosine"]
    
    for distance_func in distance_functions:
        print(f"\n[测试 {distance_func} 距离函数]")
        try:
            index_path = f"test_indices/test_{distance_func}.diskann"
            if Path(index_path).parent.exists():
                shutil.rmtree(Path(index_path).parent)
            
            # 构建索引
            builder = LeannBuilder(
                backend_name="diskann",
                distance_metric=distance_func,
                graph_degree=16,
                complexity=32
            )
            
            for doc in documents:
                builder.add_text(doc)
            
            builder.build_index(index_path)
            
            # 测试搜索
            searcher = LeannSearcher(index_path, distance_metric=distance_func)
            results = searcher.search("neural network technology", top_k=2)
            
            print(f"✅ {distance_func} 距离函数工作正常")
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result['score']:.4f}, Text: {result['text'][:50]}...")
                
        except Exception as e:
            print(f"❌ {distance_func} 距离函数失败: {e}")

def test_embedding_models():
    """测试不同的embedding模型"""
    print("\n=== 测试不同Embedding模型 ===")
    
    documents = ["AI is transforming the world", "Technology advances rapidly"]
    
    # 测试不同的embedding模型
    models_to_test = [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        # "sentence-transformers/distilbert-base-nli-mean-tokens",  # 可能不存在
    ]
    
    for model_name in models_to_test:
        print(f"\n[测试 {model_name}]")
        try:
            index_path = f"test_indices/test_model.diskann"
            if Path(index_path).parent.exists():
                shutil.rmtree(Path(index_path).parent)
            
            # 构建索引
            builder = LeannBuilder(
                backend_name="diskann",
                embedding_model=model_name,
                distance_metric="cosine"
            )
            
            for doc in documents:
                builder.add_text(doc)
            
            builder.build_index(index_path)
            
            # 测试搜索
            searcher = LeannSearcher(index_path)
            results = searcher.search("artificial intelligence", top_k=1)
            
            print(f"✅ {model_name} 模型工作正常")
            print(f"  结果: {results[0]['text'][:50]}...")
            
        except Exception as e:
            print(f"❌ {model_name} 模型失败: {e}")

def test_search_correctness():
    """验证搜索结果的正确性"""
    print("\n=== 验证搜索结果正确性 ===")
    
    # 创建有明确相关性的测试文档
    documents = [
        "Python is a programming language used for machine learning",  # 与编程相关
        "Dogs are loyal pets that love to play fetch",                # 与动物相关  
        "Machine learning algorithms can predict future trends",       # 与ML相关
        "Cats are independent animals that sleep a lot",              # 与动物相关
        "Deep learning neural networks process complex data"          # 与ML相关
    ]
    
    try:
        index_path = "test_indices/correctness_test.diskann"
        if Path(index_path).parent.exists():
            shutil.rmtree(Path(index_path).parent)
        
        # 构建索引
        builder = LeannBuilder(
            backend_name="diskann", 
            distance_metric="cosine"
        )
        
        for doc in documents:
            builder.add_text(doc)
        
        builder.build_index(index_path)
        
        # 测试相关性查询
        searcher = LeannSearcher(index_path)
        
        test_queries = [
            ("machine learning programming", [0, 2, 4]),  # 应该返回ML相关文档
            ("pet animals behavior", [1, 3]),             # 应该返回动物相关文档
        ]
        
        for query, expected_topics in test_queries:
            print(f"\n查询: '{query}'")
            results = searcher.search(query, top_k=3)
            
            print("搜索结果:")
            for i, result in enumerate(results):
                print(f"  {i+1}. ID:{result['id']}, Score:{result['score']:.4f}")
                print(f"     Text: {result['text'][:60]}...")
            
            # 简单验证：检查前两个结果是否在预期范围内
            top_ids = [result['id'] for result in results[:2]]
            relevant_found = any(id in expected_topics for id in top_ids)
            
            if relevant_found:
                print("✅ 搜索结果相关性正确")
            else:
                print("⚠️  搜索结果相关性可能有问题")
                
    except Exception as e:
        print(f"❌ 正确性测试失败: {e}")

def main():
    print("🔍 Leann DiskANN Sanity Check")
    print("=" * 50)
    
    # 清理旧的测试数据
    if Path("test_indices").exists():
        shutil.rmtree("test_indices")
    
    # 运行测试
    test_distance_functions()
    test_embedding_models() 
    test_search_correctness()
    
    print("\n" + "=" * 50)
    print("🎉 Sanity check 完成!")

if __name__ == "__main__":
    main()