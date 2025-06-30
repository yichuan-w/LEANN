import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re

# 设置风格
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# 读取数据 - 修改为自定义读取逻辑
log_file = './top3_positions_log.txt'

# 手动解析文件
data = []
header = None
with open(log_file, 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split(',')
    
    # 检查是否存在ThreadID列
    has_thread_id = 'ThreadID' in header
    
    for line in lines[1:]:
        # 跳过非数据行，如"Search X results:"
        if 'results:' in line or not ',' in line:
            continue
        
        # 分割并解析数据行
        parts = line.strip().split(',')
        
        # 检查数据是否符合格式
        if len(parts) >= 7:  # 至少需要7个字段
            # 对于旧格式(无ThreadID)的数据
            if not has_thread_id and len(parts) == 7:
                data.append([parts[0], 0, parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]])
            # 对于新格式(有ThreadID)的数据
            elif has_thread_id and len(parts) == 8:
                data.append(parts)
            # 处理不一致的格式
            elif not has_thread_id and len(parts) == 8:
                # 假设第二列是ThreadID
                data.append(parts)
                if not has_thread_id:
                    has_thread_id = True
                    header.insert(1, 'ThreadID')

# 确保header正确
if not has_thread_id:
    header.insert(1, 'ThreadID')

# 创建DataFrame并确保列名正确
if len(header) == 8:  # 确保有8列
    df = pd.DataFrame(data, columns=header)
else:
    # 如果header不正确，则使用默认列名
    default_header = ['Search#', 'ThreadID', 'FullSetSize', 'Rank', 'ID', 'PQ_Rank', 'PQ_Distance', 'Exact_Distance']
    df = pd.DataFrame(data, columns=default_header)

# 转换数值列
df['Search#'] = pd.to_numeric(df['Search#'], errors='coerce').fillna(0).astype(int)
df['ThreadID'] = pd.to_numeric(df['ThreadID'], errors='coerce').fillna(0).astype(int)
df['FullSetSize'] = pd.to_numeric(df['FullSetSize'], errors='coerce').fillna(0).astype(int)
df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce').fillna(0).astype(int)
df['ID'] = pd.to_numeric(df['ID'], errors='coerce').fillna(0).astype(int)
df['PQ_Rank'] = pd.to_numeric(df['PQ_Rank'], errors='coerce').fillna(0).astype(int)
df['PQ_Distance'] = pd.to_numeric(df['PQ_Distance'], errors='coerce').fillna(0).astype(float)
df['Exact_Distance'] = pd.to_numeric(df['Exact_Distance'], errors='coerce').fillna(0).astype(float)

print(f"读取了 {len(df)} 行数据")
print(f"搜索次数: {df['Search#'].nunique()}")
print(f"线程数: {df['ThreadID'].nunique()}")

# 提取前3名的结果
top3_df = df[df['Rank'] <= 3].copy()

# 分析PQ Rank的分布
pq_positions = []
for rank in [1, 2, 3]:
    rank_df = top3_df[top3_df['Rank'] == rank]
    pq_positions.append(rank_df['PQ_Rank'].values)

# 创建结果目录
result_dir = './analysis_results'
os.makedirs(result_dir, exist_ok=True)

# 1. 箱型图：展示top-3结果在PQ排序中的位置分布
plt.figure(figsize=(10, 6))
box_data = [top3_df[top3_df['Rank'] == i]['PQ_Rank'].values for i in [1, 2, 3]]
sns.boxplot(data=box_data)
plt.xticks([0, 1, 2], ['Top 1', 'Top 2', 'Top 3'])
plt.ylabel('PQ Rank Position')
plt.title('Distribution of PQ Ranks for Top-3 Exact Results')
plt.savefig(os.path.join(result_dir, 'pq_rank_boxplot.png'), dpi=300)

# 2. 直方图：每个排名在PQ结果中的位置分布
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for i, rank in enumerate([1, 2, 3]):
    rank_df = top3_df[top3_df['Rank'] == rank]
    sns.histplot(x=rank_df['PQ_Rank'].values, bins=20, ax=axs[i])
    axs[i].set_title(f'Exact Rank {rank}')
    axs[i].set_xlabel('PQ Rank')
    axs[i].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'pq_rank_histogram.png'), dpi=300)

# 3. 热力图：PQ排名与精确排名的关系
plt.figure(figsize=(10, 8))
# 只关注Top 20的排名
bins = list(range(0, 22))
pq_rank_bins = pd.cut(top3_df['PQ_Rank'], bins=bins)
heatmap_data = pd.crosstab(pq_rank_bins, top3_df['Rank'])
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
plt.title('Heatmap of Exact Rank vs PQ Rank (Top 20)')
plt.xlabel('Exact Rank')
plt.ylabel('PQ Rank Range')
plt.savefig(os.path.join(result_dir, 'rank_heatmap.png'), dpi=300)

# 4. 散点图：比较PQ距离和精确距离的关系
plt.figure(figsize=(10, 8))
sns.scatterplot(x=top3_df['Exact_Distance'], y=top3_df['PQ_Distance'], hue=top3_df['Rank'], palette='viridis')
plt.title('PQ Distance vs Exact Distance')
plt.xlabel('Exact Distance')
plt.ylabel('PQ Distance')
plt.legend(title='Exact Rank')
# 添加对角线表示完美匹配
min_val = min(top3_df['Exact_Distance'].min(), top3_df['PQ_Distance'].min())
max_val = max(top3_df['Exact_Distance'].max(), top3_df['PQ_Distance'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
plt.savefig(os.path.join(result_dir, 'distance_scatter.png'), dpi=300)

# 5. 折线图：PQ Rank随结果集大小的变化
plt.figure(figsize=(12, 6))
size_grouped = top3_df.groupby(['FullSetSize', 'Rank'])['PQ_Rank'].mean().reset_index()
for rank in [1, 2, 3]:
    rank_data = size_grouped[size_grouped['Rank'] == rank]
    plt.plot(rank_data['FullSetSize'], rank_data['PQ_Rank'], marker='o', label=f'Rank {rank}')
plt.xlabel('Result Set Size')
plt.ylabel('Average PQ Rank')
plt.title('Average PQ Rank by Result Set Size')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'pq_rank_by_size.png'), dpi=300)

# 6. 百分比热力图：在PQ排名前K的概率
top_k_values = [1, 5, 10, 20, 50, 100, 200, 300, 500, 700, 800, 900]
top_k_probs = []

for rank in [1, 2, 3]:
    rank_df = top3_df[top3_df['Rank'] == rank]
    probs = []
    for k in top_k_values:
        prob = (rank_df['PQ_Rank'] <= k).mean() * 100
        probs.append(prob)
    top_k_probs.append(probs)

plt.figure(figsize=(10, 6))
sns.heatmap(top_k_probs, annot=True, fmt='.1f', cmap='YlGnBu',
            xticklabels=[f'Top-{k}' for k in top_k_values],
            yticklabels=['Rank 1', 'Rank 2', 'Rank 3'])
plt.title('Probability (%) of Finding Exact Top-K Results in PQ Top-K')
plt.xlabel('PQ Top-K')
plt.ylabel('Exact Rank')
plt.savefig(os.path.join(result_dir, 'topk_probability.png'), dpi=300)

# 7. 生成统计摘要报告
with open(os.path.join(result_dir, 'summary_report.txt'), 'w') as f:
    f.write(f"数据分析摘要\n")
    f.write(f"=================\n")
    f.write(f"总搜索次数: {df['Search#'].nunique()}\n")
    f.write(f"使用线程数: {df['ThreadID'].nunique()}\n\n")
    
    f.write("精确排名前3的结果在PQ排序中的平均位置:\n")
    for rank in [1, 2, 3]:
        avg_pq_rank = top3_df[top3_df['Rank'] == rank]['PQ_Rank'].mean()
        median_pq_rank = top3_df[top3_df['Rank'] == rank]['PQ_Rank'].median()
        f.write(f"  排名 {rank}: 平均位置 = {avg_pq_rank:.2f}, 中位数位置 = {median_pq_rank:.1f}\n")
    
    f.write("\n各排名结果在PQ排序前K的命中率:\n")
    for rank in [1, 2, 3]:
        f.write(f"  排名 {rank}:\n")
        for k in top_k_values:
            hit_rate = (top3_df[top3_df['Rank'] == rank]['PQ_Rank'] <= k).mean() * 100
            f.write(f"    在PQ前 {k} 中的命中率: {hit_rate:.2f}%\n")

print(f"分析完成! 结果已保存到 {result_dir} 目录") 