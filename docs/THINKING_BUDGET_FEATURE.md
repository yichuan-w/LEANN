# Thinking Budget Feature Implementation

## Overview

This document describes the implementation of the **thinking budget** feature for LEANN, which allows users to control the computational effort for reasoning models like GPT-Oss:20b.

## Feature Description

The thinking budget feature provides three levels of computational effort for reasoning models:
- **`low`**: Fast responses, basic reasoning (default for simple queries)
- **`medium`**: Balanced speed and reasoning depth
- **`high`**: Maximum reasoning effort, best for complex analytical questions

## Implementation Details

### 1. Command Line Interface

Added `--thinking-budget` parameter to both CLI and RAG examples:

```bash
# LEANN CLI
leann ask my-index --llm ollama --model gpt-oss:20b --thinking-budget high

# RAG Examples
python apps/email_rag.py --llm ollama --llm-model gpt-oss:20b --thinking-budget high
python apps/document_rag.py --llm openai --llm-model o3 --thinking-budget medium
```

### 2. LLM Backend Support

#### Ollama Backend (`packages/leann-core/src/leann/chat.py`)

```python
def ask(self, prompt: str, **kwargs) -> str:
    # Handle thinking budget for reasoning models
    options = kwargs.copy()
    thinking_budget = kwargs.get("thinking_budget")
    if thinking_budget:
        options.pop("thinking_budget", None)
        if thinking_budget in ["low", "medium", "high"]:
            options["reasoning"] = {"effort": thinking_budget, "exclude": False}
```

**API Format**: Uses Ollama's `reasoning` parameter with `effort` and `exclude` fields.

#### OpenAI Backend (`packages/leann-core/src/leann/chat.py`)

```python
def ask(self, prompt: str, **kwargs) -> str:
    # Handle thinking budget for reasoning models
    thinking_budget = kwargs.get("thinking_budget")
    if thinking_budget and thinking_budget in ["low", "medium", "high"]:
        # Check if this is an o-series model
        o_series_models = ["o3", "o3-mini", "o4-mini", "o1", "o3-pro", "o3-deep-research"]
        if any(model in self.model for model in o_series_models):
            params["reasoning_effort"] = thinking_budget
```

**API Format**: Uses OpenAI's `reasoning_effort` parameter for o-series models.

### 3. Parameter Propagation

The thinking budget parameter is properly propagated through the LEANN architecture:

1. **CLI** (`packages/leann-core/src/leann/cli.py`): Captures `--thinking-budget` argument
2. **Base RAG** (`apps/base_rag_example.py`): Adds parameter to argument parser
3. **LeannChat** (`packages/leann-core/src/leann/api.py`): Passes `llm_kwargs` to LLM
4. **LLM Interface**: Handles the parameter in backend-specific implementations

## Files Modified

### Core Implementation
- `packages/leann-core/src/leann/chat.py`: Added thinking budget support to OllamaChat and OpenAIChat
- `packages/leann-core/src/leann/cli.py`: Added `--thinking-budget` argument
- `apps/base_rag_example.py`: Added thinking budget parameter to RAG examples

### Documentation
- `README.md`: Added thinking budget parameter to usage examples
- `docs/configuration-guide.md`: Added detailed documentation and usage guidelines

### Examples
- `examples/thinking_budget_demo.py`: Comprehensive demo script with usage examples

## Usage Examples

### Basic Usage
```bash
# High reasoning effort for complex questions
leann ask my-index --llm ollama --model gpt-oss:20b --thinking-budget high

# Medium reasoning for balanced performance
leann ask my-index --llm openai --model gpt-4o --thinking-budget medium

# Low reasoning for fast responses
leann ask my-index --llm ollama --model gpt-oss:20b --thinking-budget low
```

### RAG Examples
```bash
# Email RAG with high reasoning
python apps/email_rag.py --llm ollama --llm-model gpt-oss:20b --thinking-budget high

# Document RAG with medium reasoning
python apps/document_rag.py --llm openai --llm-model gpt-4o --thinking-budget medium
```

## Supported Models

### Ollama Models
- **GPT-Oss:20b**: Primary target model with reasoning capabilities
- **Other reasoning models**: Any Ollama model that supports the `reasoning` parameter

### OpenAI Models
- **o3, o3-mini, o4-mini, o1**: o-series reasoning models with `reasoning_effort` parameter
- **GPT-OSS models**: Models that support reasoning capabilities

## Testing

The implementation includes comprehensive testing:
- Parameter handling verification
- Backend-specific API format validation
- CLI argument parsing tests
- Integration with existing LEANN architecture
