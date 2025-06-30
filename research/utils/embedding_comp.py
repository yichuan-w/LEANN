import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else 
                     ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"使用设备: {device}")

# 定义自定义比较函数（基于内积）
def compare(a, b):
    """
    计算两个向量的内积，并返回其负值作为距离度量
    数值越小表示越相似（与提供的代码一致）
    """
    result = np.dot(a, b)
    return -result  # 返回负值，与原代码一致

# 批量计算相似度
def compute_similarities(queries, corpus):
    """计算查询向量与语料库向量之间的相似度矩阵"""
    similarities = np.zeros((len(queries), len(corpus)))
    for i, query in enumerate(queries):
        for j, doc in enumerate(corpus):
            similarities[i, j] = compare(query, doc)
    return similarities

# 加载两个模型
model_names = [
    "facebook/contriever-msmarco",           # Contriever模型
    "facebook/contriever-msmarco-int4"       # Contriever模型 (int4)
]

# 扩展的样本文本 - 分为多个主题组
texts = [
    # 组1: 关于狐狸和动物 (0-9)
    "The quick brown fox jumps over the lazy dog.",
    "A rapid auburn fox leaps above the inactive canine.",
    "The sly fox outsmarts the hunting hounds in the forest.",
    "Foxes are known for their cunning behavior and bushy tails.",
    "The red fox is the largest of the true foxes and the most common fox species.",
    "Dogs have been companions to humans for thousands of years.",
    "The lazy dog slept through the commotion of the playful fox.",
    "Wolves and foxes belong to the same family, Canidae.",
    "The arctic fox changes its coat color with the seasons.",
    "Domestic dogs come in hundreds of breeds of various sizes and appearances.",
    
    # 组2: 人工智能和机器学习 (10-19)
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning is a subset of machine learning.",
    "Neural networks are computing systems inspired by biological neural networks.",
    "AI systems can now beat human champions at complex games like chess and Go.",
    "Natural language processing allows computers to understand human language.",
    "Reinforcement learning involves training agents to make sequences of decisions.",
    "Computer vision enables machines to derive information from images and videos.",
    "The Turing test measures a machine's ability to exhibit intelligent behavior.",
    "Supervised learning uses labeled training data to learn the mapping function.",
    "Unsupervised learning finds patterns in data without pre-existing labels.",
    
    # 组3: 巴黎和法国地标 (20-29)
    "The Eiffel Tower is located in Paris, France.",
    "The Louvre Museum is in the city of Paris.",
    "Notre-Dame Cathedral is a medieval Catholic cathedral on the Île de la Cité in Paris.",
    "The Arc de Triomphe stands at the center of the Place Charles de Gaulle in Paris.",
    "The Seine River flows through the heart of Paris.",
    "Montmartre is a large hill in Paris's 18th arrondissement known for its artistic history.",
    "The Palace of Versailles is located in the Île-de-France region of France.",
    "The Champs-Élysées is an avenue in Paris famous for its theatres, cafés, and luxury shops.",
    "The Sacré-Cœur Basilica offers one of the most beautiful panoramic views of Paris.",
    "The Musée d'Orsay houses the largest collection of impressionist masterpieces in the world.",
    
    # 组4: 可再生能源 (30-39)
    "Solar panels convert sunlight into electricity.",
    "Wind turbines generate power from moving air.",
    "Hydroelectric power is generated from flowing water.",
    "Geothermal energy harnesses heat from within the Earth.",
    "Biomass energy comes from organic materials like plants and waste.",
    "Tidal energy uses the natural rise and fall of coastal tidal waters.",
    "Renewable energy sources can help reduce greenhouse gas emissions.",
    "Solar farms can span hundreds of acres with thousands of panels.",
    "Offshore wind farms are built in bodies of water to harvest wind energy.",
    "Energy storage systems are crucial for balancing renewable energy supply and demand.",
    
    # 组5: 编程语言 (40-49)
    "Python is a popular programming language for data science.",
    "JavaScript is commonly used for web development.",
    "Java is known for its 'write once, run anywhere' capability.",
    "C++ provides high-performance and close hardware control.",
    "Ruby is praised for its simplicity and productivity.",
    "PHP is a server-side scripting language designed for web development.",
    "Swift is used to develop applications for Apple platforms.",
    "Rust offers memory safety without using garbage collection.",
    "Go was designed at Google to improve programming productivity.",
    "Kotlin is fully interoperable with Java and provides more concise syntax.",
]

# 扩展的查询句子
query_texts = [
    # 动物相关查询
    "A fox jumped over a dog.",
    "Wild animals and their behaviors in forests.",
    "Different species of foxes around the world.",
    
    # AI相关查询
    "Artificial intelligence and neural networks.",
    "Machine learning algorithms and applications.",
    "The future of deep learning technology.",
    
    # 巴黎相关查询
    "Famous landmarks in Paris, France.",
    "Tourist attractions along the Seine River.",
    "Historical buildings and museums in Paris.",
    
    # 能源相关查询
    "Renewable energy sources and sustainability.",
    "Solar and wind power generation technologies.",
    "Alternative clean energy solutions for the future.",
    
    # 编程相关查询
    "Computer programming languages comparison.",
    "Best languages for web development.",
    "Programming tools for data science applications."
]

# 函数：获取BGE模型的嵌入
def get_bge_embeddings(model, tokenizer, texts, device):
    # 处理大量文本时分批进行
    batch_size = 16
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                 max_length=512, return_tensors='pt').to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # BGE使用[CLS]标记
        embeddings = model_output.last_hidden_state[:, 0]
        # 归一化嵌入
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(normalized_embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

# 函数：获取Contriever模型的嵌入
def get_contriever_embeddings(model, tokenizer, texts, device, use_int4=False):
    # 处理大量文本时分批进行
    batch_size = 16
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                 max_length=512, return_tensors='pt').to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Contriever使用平均池化
        attention_mask = encoded_input['attention_mask'].unsqueeze(-1)
        embeddings = (model_output.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        # 归一化嵌入
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(normalized_embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

# 主函数
def compare_embeddings():
    results = {}
    
    for i, model_name in enumerate(model_names):
        model_display_name = model_name
        # 给第二个模型一个不同的显示名称，以便区分
        if i == 1:
            model_display_name = "facebook/contriever-msmarco-int4"
            
        print(f"\n======= 加载模型 {i+1}: {model_display_name} =======")
        tokenizer = AutoTokenizer.from_pretrained(model_names[0])  # 两个模型使用相同的tokenizer
        
        # 如果是第二个模型（int4版本），进行量化
        if i == 1:
            print("应用int4量化...")
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModel.from_pretrained(
                    model_names[0],  # 使用相同的基础模型
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                print("成功加载int4模型")
            except Exception as e:
                print(f"int4加载失败: {e}")
                print("回退到标准模型...")
                model = AutoModel.from_pretrained(model_names[0]).to(device)
        else:
            model = AutoModel.from_pretrained(model_names[0]).to(device)
        
        model.eval()
        
        print(f"使用 {model_display_name} 生成嵌入...")
        # 所有模型都使用contriever
        use_int4 = i == 1
        corpus_embeddings = get_contriever_embeddings(model, tokenizer, texts, device, use_int4)
        query_embeddings = get_contriever_embeddings(model, tokenizer, query_texts, device, use_int4)
        
        print(f"语料库嵌入形状: {corpus_embeddings.shape}")
        print(f"查询嵌入形状: {query_embeddings.shape}")
        
        # 使用自定义函数计算相似度
        similarity_scores = compute_similarities(query_embeddings, corpus_embeddings)
        
        # 对每个查询，按相似度排序文本索引（较小的值表示更相似）
        ranked_indices = {}
        for j, scores in enumerate(similarity_scores):
            # 按相似度从低到高排序（因为我们返回的是负内积值）
            sorted_indices = np.argsort(scores)
            ranked_indices[f"query_{j+1}"] = sorted_indices
        
        results[model_display_name] = {
            'corpus_embeddings': corpus_embeddings,
            'query_embeddings': query_embeddings,
            'similarity_scores': similarity_scores,
            'ranked_indices': ranked_indices
        }
        
        # 立即打印这个模型的一些结果作为验证
        print(f"\n=== {model_display_name} 初步结果 ===")
        # 显示第一个查询的前3个结果
        query_idx = 0
        ranked_idx = ranked_indices[f"query_{query_idx+1}"]
        top_texts = [texts[idx] for idx in ranked_idx[:3]]
        print(f"查询: '{query_texts[query_idx]}'")
        print(f"排名前3位的文本:")
        for j, text in enumerate(top_texts):
            idx = ranked_idx[j]
            score = similarity_scores[query_idx][idx]
            print(f"  {j+1}. [ID:{idx}] {text} (分数: {score:.4f})")
    
    return results

# 分析结果
def analyze_results(results):
    models = list(results.keys())
    
    # 1. 比较相似度分数
    print("\n=== 相似度分数比较 ===")
    for model_name, result in results.items():
        similarities = result['similarity_scores'].flatten()
        print(f"{model_name} 相似度统计:")
        print(f"  平均值: {similarities.mean():.4f}")
        print(f"  最小值: {similarities.min():.4f}")
        print(f"  最大值: {similarities.max():.4f}")
        print(f"  标准差: {similarities.std():.4f}")
    
    # 2. 比较排序结果（针对每个查询显示前5个结果）
    print("\n=== 排序结果比较 ===")
    for query_idx in range(len(query_texts)):
        query_key = f"query_{query_idx+1}"
        print(f"\n查询 {query_idx+1}: '{query_texts[query_idx]}'")
        
        for model_name in models:
            ranked_idx = results[model_name]['ranked_indices'][query_key]
            top_texts = [texts[idx] for idx in ranked_idx[:5]]
            print(f"{model_name} 排名前5位的文本:")
            for i, text in enumerate(top_texts):
                idx = ranked_idx[i]
                score = results[model_name]['similarity_scores'][query_idx][idx]
                print(f"  {i+1}. [ID:{idx}] {text} (分数: {score:.4f})")
    
    # 3. 排序一致性分析
    print("\n=== 模型间排序一致性分析 ===")
    kendall_tau_scores = []
    spearman_scores = []
    
    for query_idx in range(len(query_texts)):
        query_key = f"query_{query_idx+1}"
        
        # 获取各模型的排序结果（只比较前10个结果）
        model1_top10 = results[models[0]]['ranked_indices'][query_key][:10]
        model2_top10 = results[models[1]]['ranked_indices'][query_key][:10]
        
        # 计算排序一致性
        kt, _ = kendalltau(model1_top10, model2_top10)
        sr, _ = spearmanr(model1_top10, model2_top10)
        
        kendall_tau_scores.append(kt)
        spearman_scores.append(sr)
        
        # 计算前10个结果的重叠率
        overlap = len(set(model1_top10) & set(model2_top10))
        overlap_rate = overlap / 10.0
        
        print(f"查询 {query_idx+1} '{query_texts[query_idx]}':")
        print(f"  Kendall's Tau = {kt:.4f}, Spearman's rank correlation = {sr:.4f}")
        print(f"  前10结果重叠率: {overlap_rate:.2f} ({overlap}/10)")
    
    print(f"\n平均 Kendall's Tau: {np.mean(kendall_tau_scores):.4f}")
    print(f"平均 Spearman's rank correlation: {np.mean(spearman_scores):.4f}")
    
    # 4. 可视化相似度分布差异
    plt.figure(figsize=(12, 6))
    for i, model_name in enumerate(models):
        sns.histplot(results[model_name]['similarity_scores'].flatten(), 
                     kde=True, label=model_name, alpha=0.6)
    
    plt.title('不同模型的相似度分布')
    plt.xlabel('相似度得分（越小越相似）')
    plt.ylabel('频率')
    plt.legend()
    plt.savefig('similarity_distribution.png')
    print("已保存相似度分布图表到 'similarity_distribution.png'")

    # 5. 可视化主题相关性
    plt.figure(figsize=(15, 10))
    
    # 为每个主题组定义颜色
    topic_colors = {
        '动物': 'blue',
        'AI': 'red',
        '巴黎': 'green',
        '能源': 'purple',
        '编程': 'orange'
    }
    
    # 定义主题组范围
    topic_ranges = {
        '动物': (0, 10),
        'AI': (10, 20),
        '巴黎': (20, 30),
        '能源': (30, 40),
        '编程': (40, 50)
    }
    
    # 对每个查询显示前10个结果的主题分布
    query_groups = [
        [0, 1, 2],           # 动物查询组
        [3, 4, 5],           # AI查询组
        [6, 7, 8],           # 巴黎查询组
        [9, 10, 11],         # 能源查询组
        [12, 13, 14]         # 编程查询组
    ]
    
    for group_idx, group in enumerate(query_groups):
        plt.subplot(len(query_groups), 1, group_idx+1)
        
        # 为每个模型计算主题分布
        bar_width = 0.35
        bar_positions = np.arange(len(topic_ranges))
        
        for model_idx, model_name in enumerate(models):
            # 统计每个主题在前10个结果中的出现次数
            topic_counts = {topic: 0 for topic in topic_ranges.keys()}
            
            for query_idx in group:
                query_key = f"query_{query_idx+1}"
                top10 = results[model_name]['ranked_indices'][query_key][:10]
                
                for idx in top10:
                    for topic, (start, end) in topic_ranges.items():
                        if start <= idx < end:
                            topic_counts[topic] += 1
            
            # 绘制主题分布柱状图
            plt.bar(bar_positions + (model_idx * bar_width), 
                   list(topic_counts.values()), 
                   bar_width, 
                   label=model_name)
        
        plt.title(f"查询组 {group_idx+1}: {', '.join([query_texts[i] for i in group[:1]])}")
        plt.xticks(bar_positions + bar_width/2, list(topic_ranges.keys()))
        plt.ylabel('前10结果中的出现次数')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('topic_distribution.png')
    print("已保存主题分布图表到 'topic_distribution.png'")

    # 6. 可视化查询与相关文档的相似度热图
    plt.figure(figsize=(15, 12))
    
    for i, model_name in enumerate(models):
        plt.subplot(2, 1, i+1)
        
        # 获取相似度矩阵（负数越小表示越相似）
        sim_matrix = results[model_name]['similarity_scores']
        
        # 将负值转换为正值以便可视化（越大表示越相似）
        sim_matrix_viz = -sim_matrix
        
        # 绘制热图
        sns.heatmap(sim_matrix_viz, cmap='YlGnBu', 
                   xticklabels=[f"Doc{i}" for i in range(len(texts))], 
                   yticklabels=[f"Q{i+1}" for i in range(len(query_texts))],
                   cbar_kws={'label': '相似度（越高越相似）'})
        
        plt.title(f"{model_name} 相似度热图")
        plt.xlabel('文档ID')
        plt.ylabel('查询ID')
    
    plt.tight_layout()
    plt.savefig('similarity_heatmap.png')
    print("已保存相似度热图到 'similarity_heatmap.png'")

if __name__ == "__main__":
    print("开始比较嵌入模型...")
    results = compare_embeddings()
    analyze_results(results)
    print("\n比较完成!")