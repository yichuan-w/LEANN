# LEANN ReAct Agent Guide

## Overview

The LEANN ReAct (Reasoning + Acting) Agent enables **multiturn retrieval and reasoning** for complex queries that require multiple search iterations. Unlike the standard `leann ask` command which performs a single search and answer, the ReAct agent can:

- **Reason** about what information is needed
- **Act** by performing targeted searches
- **Observe** the results and iterate
- **Answer** based on all gathered context

This is particularly useful for questions that require:
- Multiple pieces of information from different parts of your index
- Iterative refinement of search queries
- Complex reasoning that builds on previous findings

## How It Works

The ReAct agent follows a **Thought → Action → Observation** loop:

1. **Thought**: The agent analyzes the question and determines what information is needed
2. **Action**: The agent performs a search query based on its reasoning
3. **Observation**: The agent reviews the search results
4. **Iteration**: The process repeats until the agent has enough information or reaches the maximum iteration limit
5. **Final Answer**: The agent synthesizes all gathered information into a comprehensive answer

## Basic Usage

### Command Line

```bash
# Basic usage
leann react <index_name> "your question"

# With custom LLM settings
leann react my-index "What are the main features discussed?" \
  --llm ollama \
  --model qwen3:8b \
  --max-iterations 5 \
  --top-k 5
```

### Command Options

- `index_name`: Name of the LEANN index to search
- `query`: The question to research
- `--llm`: LLM provider (`ollama`, `openai`, `anthropic`, `hf`, `simulated`) - default: `ollama`
- `--model`: Model name (default: `qwen3:8b`)
- `--host`: Override Ollama-compatible host (defaults to `LEANN_OLLAMA_HOST` or `OLLAMA_HOST`)
- `--top-k`: Number of results per search iteration (default: `5`)
- `--max-iterations`: Maximum number of search iterations (default: `5`)
- `--api-base`: Base URL for OpenAI-compatible APIs
- `--api-key`: API key for cloud LLM providers

### Python API

```python
from leann import create_react_agent, LeannSearcher

# Create a searcher
searcher = LeannSearcher(index_path="path/to/index.leann")

# Create the ReAct agent
agent = create_react_agent(
    index_path="path/to/index.leann",
    llm_config={
        "type": "ollama",
        "model": "qwen3:8b",
        "host": "http://localhost:11434"  # optional
    },
    max_iterations=5
)

# Run the agent
answer = agent.run("What are the main topics covered in the documentation?", top_k=5)
print(answer)

# Access search history
if agent.search_history:
    print(f"\nSearch History ({len(agent.search_history)} iterations):")
    for entry in agent.search_history:
        print(f"  {entry['iteration']}. {entry['action']} ({entry['results_count']} results)")
```

## Example Use Cases

### 1. Multi-faceted Questions

```bash
# Questions that need information from multiple sources
leann react docs-index "What are the differences between HNSW and DiskANN backends, and when should I use each?"
```

The agent will:
- First search for "HNSW backend features"
- Then search for "DiskANN backend features"
- Compare the results
- Provide a comprehensive answer

### 2. Iterative Research

```bash
# Questions requiring multiple search iterations
leann react codebase-index "How does the embedding computation work and what optimizations are used?"
```

The agent will:
- Search for "embedding computation"
- Based on results, search for "embedding optimizations"
- Refine queries based on findings
- Synthesize the information

### 3. Complex Reasoning

```bash
# Questions that require building understanding
leann react research-index "What are the performance characteristics of different indexing strategies?"
```

## Comparison: `leann ask` vs `leann react`

| Feature | `leann ask` | `leann react` |
|---------|-------------|---------------|
| **Search iterations** | Single search | Multiple iterations |
| **Query refinement** | No | Yes, based on observations |
| **Use case** | Simple Q&A | Complex, multi-faceted questions |
| **Speed** | Faster | Slower (multiple searches) |
| **Reasoning** | Direct answer | Iterative reasoning |

### When to Use Each

**Use `leann ask` when:**
- You have a straightforward question
- A single search should provide enough context
- You want a quick answer

**Use `leann react` when:**
- Your question requires information from multiple sources
- You need the agent to explore and refine its understanding
- The answer requires synthesizing multiple pieces of information

## Advanced Configuration

### Custom LLM Providers

```bash
# Using OpenAI
leann react my-index "question" \
  --llm openai \
  --model gpt-4 \
  --api-base https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY

# Using Anthropic
leann react my-index "question" \
  --llm anthropic \
  --model claude-3-opus-20240229 \
  --api-key $ANTHROPIC_API_KEY
```

### Adjusting Search Parameters

```bash
# More results per iteration
leann react my-index "question" --top-k 10

# More iterations for complex questions
leann react my-index "question" --max-iterations 10
```

## Understanding the Output

When you run `leann react`, you'll see:

1. **Question**: The original question being researched
2. **Iteration logs**: Each search action and its results
3. **Final Answer**: The synthesized answer based on all iterations
4. **Search History**: Summary of all search iterations performed

Example output:

```
🤖 Starting ReAct agent with index 'my-index'...
Using qwen3:8b (ollama)

🔍 Question: What are the main features of LEANN?

🔍 Action: search("LEANN features")
[Result 1] (Score: 0.923)
LEANN is a vector database that saves 97% storage...

🔍 Action: search("LEANN storage optimization")
[Result 1] (Score: 0.891)
LEANN uses compact storage and recomputation...

✅ Final Answer:
LEANN is a vector database with several key features:
1. 97% storage savings compared to traditional vector databases
2. Compact storage with recomputation capabilities
3. Support for multiple backends (HNSW and DiskANN)
...

📊 Search History (2 iterations):
  1. search("LEANN features") (5 results)
  2. search("LEANN storage optimization") (5 results)
```

## Tips for Best Results

1. **Be specific**: Clear, specific questions work better than vague ones
2. **Adjust iterations**: Complex questions may need more iterations (increase `--max-iterations`)
3. **Monitor history**: Check the search history to understand the agent's reasoning
4. **Use appropriate models**: Larger models generally provide better reasoning, but are slower
5. **Index quality**: Ensure your index is well-built with relevant content

## Limitations

- **Speed**: Multiple iterations make ReAct slower than single-search queries
- **Cost**: More LLM calls mean higher costs for cloud providers
- **Complexity**: Very complex questions may still require human review
- **Model dependency**: Reasoning quality depends on the LLM's capabilities

## Future Enhancements

This is the first implementation (1/N) of Deep-Research integration. Future enhancements may include:
- Web search integration for external information
- More sophisticated reasoning strategies
- Parallel search execution
- Better query optimization

## Related Documentation

- [Basic Usage Guide](../README.md)
- [CLI Reference](configuration-guide.md)
- [Embedding Models](normalized_embeddings.md)
