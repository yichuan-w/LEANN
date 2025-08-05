# Claude Code x LEANN 集成指南

## ✅ 现状：已经可以工作！

好消息：LEANN CLI已经完全可以在Claude Code中使用，无需任何修改！

## 🚀 立即开始

### 1. 激活环境
```bash
# 在LEANN项目目录下
source .venv/bin/activate.fish  # fish shell
# 或
source .venv/bin/activate       # bash shell
```

### 2. 基本命令

#### 查看现有索引
```bash
leann list
```

#### 搜索文档
```bash
leann search my-docs "machine learning" --recompute-embeddings
```

#### 问答对话
```bash
echo "What is machine learning?" | leann ask my-docs --llm ollama --model qwen3:8b --recompute-embeddings
```

#### 构建新索引
```bash
leann build project-docs --docs ./src --recompute-embeddings
```

## 💡 Claude Code 使用技巧

### 在Claude Code中直接使用

1. **激活环境**：
   ```bash
   cd /Users/andyl/Projects/LEANN-RAG
   source .venv/bin/activate.fish
   ```

2. **搜索代码库**：
   ```bash
   leann search my-docs "authentication patterns" --recompute-embeddings --top-k 10
   ```

3. **智能问答**：
   ```bash
   echo "How does the authentication system work?" | leann ask my-docs --llm ollama --model qwen3:8b --recompute-embeddings
   ```

### 批量操作示例

```bash
# 构建项目文档索引
leann build project-docs --docs ./docs --force

# 搜索多个关键词
leann search project-docs "API authentication" --recompute-embeddings
leann search project-docs "database schema" --recompute-embeddings
leann search project-docs "deployment guide" --recompute-embeddings

# 问答模式
echo "What are the API endpoints?" | leann ask project-docs --recompute-embeddings
```

## 🎯 Claude 可以立即执行的工作流

### 代码分析工作流
```bash
# 1. 构建代码库索引
leann build codebase --docs ./src --backend hnsw --recompute-embeddings

# 2. 分析架构
echo "What is the overall architecture?" | leann ask codebase --recompute-embeddings

# 3. 查找特定功能
leann search codebase "user authentication" --recompute-embeddings --top-k 5

# 4. 理解实现细节
echo "How is user authentication implemented?" | leann ask codebase --recompute-embeddings
```

### 文档理解工作流
```bash
# 1. 索引项目文档
leann build docs --docs ./docs --recompute-embeddings

# 2. 快速查找信息
leann search docs "installation requirements" --recompute-embeddings

# 3. 获取详细说明
echo "What are the system requirements?" | leann ask docs --recompute-embeddings
```

## ⚠️ 重要提示

1. **必须使用 `--recompute-embeddings`** - 这是关键参数，不加会报错
2. **需要先激活虚拟环境** - 确保有LEANN的Python环境
3. **Ollama需要预先安装** - ask功能需要本地LLM

## 🔥 立即可用的Claude提示词

```
Help me analyze this codebase using LEANN:

1. First, activate the environment:
   cd /Users/andyl/Projects/LEANN-RAG && source .venv/bin/activate.fish

2. Build an index of the source code:
   leann build codebase --docs ./src --recompute-embeddings

3. Search for authentication patterns:
   leann search codebase "authentication middleware" --recompute-embeddings --top-k 10

4. Ask about the authentication system:
   echo "How does user authentication work in this codebase?" | leann ask codebase --recompute-embeddings

Please execute these commands and help me understand the code structure.
```

## 📈 下一步改进计划

虽然现在已经可以用，但还可以进一步优化：

1. **简化命令** - 默认启用recompute-embeddings
2. **配置文件** - 避免重复输入参数
3. **状态管理** - 自动检测环境和索引
4. **输出格式** - 更适合Claude解析的格式

但这些都是锦上添花，现在就能用起来！

## 🎉 总结

**LEANN现在就可以在Claude Code中完美工作！**

- ✅ 搜索功能正常
- ✅ RAG问答功能正常
- ✅ 索引构建功能正常
- ✅ 支持多种数据源
- ✅ 支持本地LLM

只需要记住加上 `--recompute-embeddings` 参数就行！
