#!/usr/bin/env python3
"""
验证DiskANN L2距离是否真正工作
"""

import numpy as np
from pathlib import Path
import shutil

# 导入后端包以触发插件注册
try:
    import leann_backend_diskann
    print("INFO: Backend packages imported successfully.")
except ImportError as e:
    print(f"WARNING: Could not import backend packages. Error: {e}")

from leann.api import LeannBuilder, LeannSearcher

def test_l2_verification():
    """验证L2距离是否真正被使用"""
    print("=== 验证DiskANN L2距离实现 ===")
    
    INDEX_DIR = Path("./test_l2_verification")
    INDEX_PATH = str(INDEX_DIR / "documents.diskann")
    
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    
    # 创建特殊的测试文档，使L2和cosine产生不同结果
    documents = [
        "machine learning artificial intelligence",  # 文档0
        "computer programming software development",  # 文档1  
        "data science analytics statistics"          # 文档2
    ]
    
    print("构建索引...")
    builder = LeannBuilder(
        backend_name="diskann",
        distance_metric="l2",  # 明确指定L2
        graph_degree=16,
        complexity=32
    )
    
    for i, doc in enumerate(documents):
        builder.add_text(doc, metadata={"id": i, "text": doc})
    
    builder.build_index(INDEX_PATH)
    print("✅ 索引构建完成")
    
    # 测试搜索
    searcher = LeannSearcher(INDEX_PATH, distance_metric="l2")
    
    # 用一个与文档0非常相似的查询
    query = "machine learning AI technology"
    results = searcher.search(query, top_k=3)
    
    print(f"\n查询: '{query}'")
    print("L2距离搜索结果:")
    for i, result in enumerate(results):
        print(f"  {i+1}. ID:{result['id']}, Score:{result['score']:.6f}")
        print(f"     Text: {result['text']}")
    
    # 现在用cosine重新测试同样的数据
    print(f"\n--- 用Cosine距离对比测试 ---")
    
    INDEX_DIR_COS = Path("./test_cosine_verification") 
    INDEX_PATH_COS = str(INDEX_DIR_COS / "documents.diskann")
    
    if INDEX_DIR_COS.exists():
        shutil.rmtree(INDEX_DIR_COS)
    
    builder_cos = LeannBuilder(
        backend_name="diskann",
        distance_metric="cosine",  # 使用cosine
        graph_degree=16,
        complexity=32
    )
    
    for i, doc in enumerate(documents):
        builder_cos.add_text(doc, metadata={"id": i, "text": doc})
    
    builder_cos.build_index(INDEX_PATH_COS)
    
    searcher_cos = LeannSearcher(INDEX_PATH_COS, distance_metric="cosine")
    results_cos = searcher_cos.search(query, top_k=3)
    
    print("Cosine距离搜索结果:")
    for i, result in enumerate(results_cos):
        print(f"  {i+1}. ID:{result['id']}, Score:{result['score']:.6f}")
        print(f"     Text: {result['text']}")
    
    # 对比分析
    print(f"\n--- 结果对比分析 ---")
    print("L2距离的分数是欧几里得距离平方，越小越相似")
    print("Cosine距离的分数是余弦相似度的负值，越小越相似")
    
    l2_top = results[0]
    cos_top = results_cos[0]
    
    print(f"L2最佳匹配: ID{l2_top['id']}, Score={l2_top['score']:.6f}")
    print(f"Cosine最佳匹配: ID{cos_top['id']}, Score={cos_top['score']:.6f}")
    
    if l2_top['id'] == cos_top['id']:
        print("✅ 两种距离函数返回相同的最佳匹配")
    else:
        print("⚠️  两种距离函数返回不同的最佳匹配 - 这表明它们确实使用了不同的距离计算")
        
    # 验证分数范围的合理性
    l2_scores = [r['score'] for r in results]
    cos_scores = [r['score'] for r in results_cos]
    
    print(f"L2分数范围: {min(l2_scores):.6f} 到 {max(l2_scores):.6f}")
    print(f"Cosine分数范围: {min(cos_scores):.6f} 到 {max(cos_scores):.6f}")
    
    # L2分数应该是正数，cosine分数应该在-1到0之间（因为是负的相似度）
    if all(score >= 0 for score in l2_scores):
        print("✅ L2分数都是正数，符合预期")
    else:
        print("❌ L2分数有负数，可能有问题")
        
    if all(-1 <= score <= 0 for score in cos_scores):
        print("✅ Cosine分数在合理范围内")
    else:
        print(f"⚠️  Cosine分数超出预期范围: {cos_scores}")

if __name__ == "__main__":
    test_l2_verification()