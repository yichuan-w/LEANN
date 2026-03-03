# LEANN ReAct Agent Guide

## Overview

The LEANN ReAct (Reasoning + Acting) Agent enables **multiturn retrieval and reasoning** for complex queries that require multiple search iterations. Unlike the standard `leann ask` command which performs a single search and answer, the ReAct agent can:

- **Reason** about what information is needed
- **Act** by performing targeted searches (both local and web)
- **Observe** the results and iterate
- **Answer** based on all gathered context

This is particularly useful for questions that require:
- Multiple pieces of information from different parts of your index
- **Combining local knowledge with current web information**
- **Accessing external documentation or latest updates**
- Iterative refinement of search queries
- Complex reasoning that builds on previous findings

## How It Works

The ReAct agent follows a **Thought → Action → Observation** loop:

1. **Thought**: The agent analyzes the question and determines what information is needed
2. **Action**: The agent performs a search based on its reasoning, choosing from:
   - `leann_search("query")` - Search your local private knowledge base
   - `web_search("query")` - Search the public internet for current information
   - `visit_page("url")` - Fetch and read the full content of a specific web page
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
- `--serper-api-key`: Serper API key for web search (enables web_search action)
- `--jina-api-key`: Jina AI API key for enhanced page content fetching (optional)

### Python API

```python
from leann import create_react_agent
import os

# Create the ReAct agent with web search enabled
agent = create_react_agent(
    index_path="path/to/index.leann",
    llm_config={
        "type": "ollama",
        "model": "qwen3:8b",
        "host": "http://localhost:11434"  # optional
    },
    max_iterations=5,
    serper_api_key=os.getenv("SERPER_API_KEY")  # Enable web search
)

# Run the agent - it will automatically choose between local and web search
answer = agent.run("What are the latest Python 3.13 features?", top_k=5)
print(answer)

# Access search history to see which actions were used
if agent.search_history:
    print(f"\nSearch History ({len(agent.search_history)} iterations):")
    for entry in agent.search_history:
        action_type = "Web" if "web_search" in entry['action'] else "Local"
        print(f"  {entry['iteration']}. [{action_type}] {entry['action']} ({entry['results_count']} results)")
```

## Web Search Integration

The ReAct agent can seamlessly combine searches from your local knowledge base with real-time web searches, enabling it to answer questions that require both private and public information.

### Enabling Web Search

**Get a Serper API Key:**
1. Visit https://serper.dev/
2. Sign up for a free account (2,500 queries/month free)
3. Copy your API key

**Use web search via CLI:**
```bash
# Set API key as environment variable
export SERPER_API_KEY="your-api-key-here"

# Run with web search enabled
leann react my-index "What are the latest AI developments?" \
  --llm ollama \
  --model qwen3:8b \
  --serper-api-key $SERPER_API_KEY
```

**Or pass the key directly:**
```bash
leann react my-index "question" \
  --llm ollama \
  --serper-api-key sk-your-key-here
```

### How the Agent Chooses Actions

The agent intelligently decides which action to use based on the question:

- **`leann_search("query")`** - For questions about your private data:
  - Code in your repository
  - Internal documentation
  - Chat history, notes, or private files
  - Project-specific details

- **`web_search("query")`** - For questions requiring current/public information:
  - Latest news or updates
  - Current best practices
  - Public documentation
  - General knowledge or concepts

- **`visit_page("url")`** - To read specific web pages:
  - When a URL is mentioned in the question
  - When search results contain a relevant link
  - To get detailed information from a specific source

### Web Search Examples

**Example 1: Pure Web Search**
```bash
leann react my-index "What are the latest features in Python 3.13?" \
  --serper-api-key $SERPER_API_KEY
```
Agent actions:
1. `web_search("Python 3.13 features latest")` → Gets current information from the web
2. Returns answer based on web results

**Example 2: Combining Local + Web**
```bash
leann react my-index "Compare our authentication implementation with current security best practices" \
  --serper-api-key $SERPER_API_KEY \
  --max-iterations 8
```
Agent actions:
1. `leann_search("authentication implementation")` → Finds your code
2. `web_search("authentication security best practices 2025")` → Gets current standards
3. Compares and provides recommendations

**Example 3: Following Web Links**
```bash
leann react my-index "What does the official Python async documentation say about event loops?" \
  --serper-api-key $SERPER_API_KEY \
  --jina-api-key $JINA_API_KEY
```
Agent actions:
1. `web_search("Python async event loop documentation")` → Finds official docs link
2. `visit_page("https://docs.python.org/3/library/asyncio-eventloop.html")` → Reads full page
3. Answers based on official documentation

### Web Search Output

When using web search, you'll see indicators in the output:

```
🤖 Starting ReAct agent with index 'my-index'...

--- Iteration 1/5 ---
💭 Thought: I need current information about Python 3.13
🔍 Action: web_search("Python 3.13 new features")

[Web Result 1]
Title: Python 3.13 Release Highlights
Link: https://docs.python.org/3.13/whatsnew/3.13.html
Snippet: Python 3.13 introduces a new interactive interpreter...

--- Iteration 2/5 ---
💭 Thought: I have enough information
✅ Final Answer: Python 3.13 introduces several key features...
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

This is part of ongoing Deep-Research integration. Current capabilities:
- ✅ Web search integration for external information (via Serper API)
- ✅ Page content fetching (via Jina AI Reader)
- ✅ Intelligent action selection between local and web search

Potential future enhancements:
- More sophisticated reasoning strategies
- Parallel search execution
- Better query optimization
- Additional web search backends (Tavily, Bing, etc.)
- Caching of web search results

## Related Documentation

- [Basic Usage Guide](../README.md)
- [CLI Reference](configuration-guide.md)
- [Embedding Models](normalized_embeddings.md)
