#!/usr/bin/env python3
"""
DiskANN 距离函数测试
"""

import os
from pathlib import Path
import shutil
import time

# 导入后端包以触发插件注册
try:
    import leann_backend_diskann
    import leann_backend_hnsw
    print("INFO: Backend packages imported successfully.")
except ImportError as e:
    print(f"WARNING: Could not import backend packages. Error: {e}")

# 从 leann-core 导入上层 API
from leann.api import LeannBuilder, LeannSearcher


def load_sample_documents():
    """创建用于演示的样本文档"""
    docs = [
        {"title": "Intro to Python", "content": "Python is a programming language for machine learning"},
        {"title": "ML Basics", "content": "Machine learning algorithms build intelligent systems"},
        {"title": "Data Structures", "content": "Data structures like arrays and graphs organize information"},
    ]
    return docs


def test_distance_function(distance_func, test_name):
    """测试特定距离函数"""
    print(f"\n=== 测试 {test_name} ({distance_func}) ===")
    
    INDEX_DIR = Path(f"./test_indices_{distance_func}")
    INDEX_PATH = str(INDEX_DIR / "documents.diskann")
    
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    
    # 构建索引
    print(f"构建索引 (距离函数: {distance_func})...")
    builder = LeannBuilder(
        backend_name="diskann",
        distance_metric=distance_func,
        graph_degree=16,
        complexity=32
    )
    
    documents = load_sample_documents()
    for doc in documents:
        builder.add_text(doc["content"], metadata=doc)
    
    try:
        builder.build_index(INDEX_PATH)
        print(f"✅ 索引构建成功")
        
        # 测试搜索
        searcher = LeannSearcher(INDEX_PATH, distance_metric=distance_func)
        results = searcher.search("machine learning programming", top_k=2)
        
        print(f"搜索结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Text: {result['text'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    print("🔍 DiskANN 距离函数测试")
    print("=" * 50)
    
    # 测试不同距离函数
    distance_tests = [
        ("mips", "Maximum Inner Product Search"),
        ("l2", "L2 Euclidean Distance"), 
        ("cosine", "Cosine Similarity")
    ]
    
    results = {}
    for distance_func, test_name in distance_tests:
        try:
            success = test_distance_function(distance_func, test_name)
            results[distance_func] = success
        except Exception as e:
            print(f"❌ {distance_func} 测试异常: {e}")
            results[distance_func] = False
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    for distance_func, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {distance_func:10s}: {status}")
    
    print("\n🎉 测试完成!")


if __name__ == "__main__":
    main()