# LEANN HTTP Server

The LEANN HTTP Server provides REST API and WebSocket endpoints for semantic document search and Q&A with LLM integration. It also includes support for the Model Context Protocol (MCP) for integration with AI assistants.

## Features

- **REST API**: Search documents using semantic search
- **WebSocket Streaming**: Real-time Q&A with streaming LLM responses
- **MCP Protocol**: Integration with Claude Desktop and other MCP clients
- **Multiple LLM Providers**: Support for OpenAI, Anthropic Claude, and Ollama
- **HTTPS Support**: Optional SSL/TLS encryption

## Installation

The HTTP server is included in the `leann-core` package:

```bash
pip install leann-core
```

## Quick Start

### Starting the Server

```bash
# Basic usage with OpenAI
leann_http --index-path ~/.leann/indexes/my-docs/documents.leann

# With Anthropic Claude
leann_http \
  --index-path ~/.leann/indexes/legal-docs/documents.leann \
  --llm-type anthropic \
  --model claude-3-5-sonnet-20241022

# With Ollama (local)
leann_http \
  --index-path ~/.leann/indexes/contracts/documents.leann \
  --llm-type ollama \
  --model llama3.1

# With custom host and port
leann_http \
  --index-path ~/.leann/indexes/my-docs/documents.leann \
  --host 0.0.0.0 \
  --port 8080

# With HTTPS
leann_http \
  --index-path ~/.leann/indexes/my-docs/documents.leann \
  --server-ssl-cert /path/to/cert.pem \
  --server-ssl-key /path/to/key.pem
```

### Command-Line Options

```
Server Options:
  --host HOST              Host to bind to (default: 0.0.0.0)
  --port PORT              Port to bind to (default: 8000)
  --server-ssl-cert PATH   SSL certificate for HTTPS
  --server-ssl-key PATH    SSL private key for HTTPS

Index Options:
  --index-path PATH        Path to LEANN index (required)
                          Can be a full path or just the index name

LLM Options:
  --llm-type TYPE         LLM provider: openai, anthropic, ollama (default: openai)
  --model MODEL           Model name (default: gpt-4o)
  --llm-ssl-cert PATH     SSL certificate for LLM client

Other Options:
  --embedding-model NAME  Embedding model name for reference
```

### Using Index Names

You can specify just the index name instead of the full path:

```bash
# These are equivalent:
leann_http --index-path my-docs
leann_http --index-path ~/.leann/indexes/my-docs/documents.leann
```

## API Documentation

### Health Check

Check server status and configuration.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "index_loaded": true,
  "index_path": "/home/user/.leann/indexes/my-docs/documents.leann",
  "llm_type": "openai",
  "llm_model": "gpt-4o"
}
```

### List Indexes

List all available LEANN indexes.

**Endpoint**: `GET /indexes`

**Response**:
```json
{
  "indexes": [
    {
      "name": "legal-docs",
      "path": "/home/user/.leann/indexes/legal-docs/documents.leann",
      "backend": "diskann",
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
      "dimensions": 384
    }
  ]
}
```

### Search

Perform semantic search across documents.

**Endpoint**: `POST /search`

**Request**:
```json
{
  "query": "contract termination clauses",
  "top_k": 5,
  "complexity": 64,
  "show_metadata": true,
  "metadata_filters": {
    "document_type": "contract"
  }
}
```

**Parameters**:
- `query` (string, required): Search query
- `top_k` (integer, 1-100, default: 5): Number of results
- `complexity` (integer, 16-512, default: 64): Search complexity
- `show_metadata` (boolean, default: false): Include metadata in results
- `metadata_filters` (object, optional): Filter results by metadata

**Response**:
```json
{
  "results": [
    {
      "id": "doc-123",
      "score": 0.95,
      "text": "Contract termination requires 30 days written notice...",
      "metadata": {
        "source": "contract.pdf",
        "page": 12
      }
    }
  ],
  "search_time_ms": 45.2
}
```

### WebSocket Q&A

Stream Q&A responses in real-time.

**Endpoint**: `WebSocket /ws/ask`

**Send**:
```json
{
  "question": "What are the termination requirements?",
  "top_k": 5,
  "complexity": 64,
  "llm_params": {
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

**Receive** (multiple messages):

1. Search results:
```json
{
  "type": "search_results",
  "results": [...],
  "search_time_ms": 45.2
}
```

2. Streaming tokens:
```json
{
  "type": "token",
  "content": "Based on the contract, "
}
```

3. Completion:
```json
{
  "type": "done"
}
```

## MCP Protocol Support

The server implements the Model Context Protocol for integration with AI assistants.

### MCP Endpoints

- **HTTP POST**: `POST /` - Direct MCP requests
- **SSE**: `GET /sse` - Server-Sent Events for streaming
- **Message**: `POST /message?sessionId=...` - Send messages in SSE sessions

### MCP Tools

#### leann_search

Search documents using natural language.

**Input**:
```json
{
  "query": "patient consent requirements"
}
```

#### leann_list

Show information about the loaded document collection.

**Input**: None

#### leann_ask

Ask questions and get AI-powered answers.

**Input**:
```json
{
  "question": "What are the penalties for late payment?"
}
```

### Claude Desktop Integration

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "leann-legal": {
      "command": "leann_http",
      "args": [
        "--index-path", "legal-docs",
        "--llm-type", "anthropic",
        "--model", "claude-3-5-sonnet-20241022"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Client Examples

### Python - REST API

```python
import requests

# Search
response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "tax exemption rules",
        "top_k": 10,
        "show_metadata": True
    }
)
results = response.json()["results"]

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:200]}...")
    print()
```

### Python - WebSocket

```python
import asyncio
import json
import websockets

async def ask_question():
    uri = "ws://localhost:8000/ws/ask"

    async with websockets.connect(uri) as websocket:
        # Send question
        await websocket.send(json.dumps({
            "question": "What is the statute of limitations?",
            "top_k": 5
        }))

        # Receive search results
        search_msg = await websocket.recv()
        search_data = json.loads(search_msg)
        print(f"Found {len(search_data['results'])} results")

        # Stream answer
        answer = ""
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)

            if data["type"] == "token":
                answer += data["content"]
                print(data["content"], end="", flush=True)
            elif data["type"] == "done":
                break
            elif data["type"] == "error":
                print(f"\nError: {data['message']}")
                break

        print(f"\n\nComplete answer: {answer}")

asyncio.run(ask_question())
```

### JavaScript/TypeScript

```typescript
// Search
const searchResponse = await fetch('http://localhost:8000/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'data protection obligations',
    top_k: 5
  })
});

const searchData = await searchResponse.json();
console.log('Results:', searchData.results);

// WebSocket Q&A
const ws = new WebSocket('ws://localhost:8000/ws/ask');

ws.onopen = () => {
  ws.send(JSON.stringify({
    question: 'What are the GDPR requirements?',
    top_k: 5
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'search_results') {
    console.log('Search results:', data.results);
  } else if (data.type === 'token') {
    process.stdout.write(data.content);
  } else if (data.type === 'done') {
    console.log('\n\nComplete!');
    ws.close();
  }
};
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# List indexes
curl http://localhost:8000/indexes

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "contract termination clauses",
    "top_k": 5,
    "show_metadata": true
  }'

# MCP initialize
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize"
  }'

# MCP search
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "leann_search",
      "arguments": {
        "query": "patient consent requirements"
      }
    }
  }'
```

## Environment Variables

### OpenAI
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_BASE_URL`: Custom OpenAI-compatible endpoint

### Anthropic
- `ANTHROPIC_API_KEY`: Anthropic API key

### Ollama
- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)

## Security Considerations

### Authentication

The current implementation does not include authentication. For production use:

1. **Use a reverse proxy** (nginx, Caddy) with authentication
2. **Implement API keys** by modifying the server
3. **Use network-level security** (VPN, firewall rules)

### HTTPS

Always use HTTPS in production:

```bash
leann_http \
  --index-path my-docs \
  --server-ssl-cert /path/to/cert.pem \
  --server-ssl-key /path/to/key.pem
```

### Rate Limiting

Consider using a reverse proxy with rate limiting:

```nginx
# nginx example
limit_req_zone $binary_remote_addr zone=leann:10m rate=10r/s;

server {
    location / {
        limit_req zone=leann burst=20;
        proxy_pass http://localhost:8000;
    }
}
```

## Troubleshooting

### Server won't start

**Issue**: `Index not found: my-docs`

**Solution**: Use full path or ensure index exists in `~/.leann/indexes/`

```bash
# Check if index exists
ls ~/.leann/indexes/my-docs/

# Use full path
leann_http --index-path ~/.leann/indexes/my-docs/documents.leann
```

### LLM not responding

**Issue**: `Error: LLM not initialized`

**Solution**: Ensure API keys are set

```bash
# For OpenAI
export OPENAI_API_KEY="your-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-key"

# For Ollama, ensure server is running
ollama serve
```

### WebSocket disconnects

**Issue**: WebSocket closes during long LLM responses

**Solution**: The server has increased ping timeout to 600s. If still experiencing issues, check your reverse proxy timeout settings.

```nginx
# nginx example
proxy_read_timeout 600s;
proxy_send_timeout 600s;
```

### SSL Certificate Issues

**Issue**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**: Use `--llm-ssl-cert` for custom CA certificates

```bash
leann_http \
  --index-path my-docs \
  --llm-ssl-cert /path/to/custom-ca.pem
```

## Performance Tuning

### Search Parameters

- **complexity**: Higher values (128, 256) = better quality, slower
- **top_k**: More results = slower but more context for LLM

### LLM Parameters

```json
{
  "temperature": 0.7,     // Lower = more focused, higher = more creative
  "max_tokens": 10000     // Adjust based on needs
}
```

### Server Configuration

```bash
# For production, use multiple workers
uvicorn leann.http_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

## Advanced Usage

### Custom Prompts

Modify the prompt in WebSocket requests:

```python
# In your client code
question = "What are the GDPR requirements?"
context = "... search results ..."
custom_prompt = f"""You are a legal expert. Based on this context:

{context}

Please answer: {question}

Focus on specific articles and regulations."""

# Send to LLM
```

### Batch Processing

```python
import asyncio
import aiohttp

async def batch_search(queries):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for query in queries:
            task = session.post(
                'http://localhost:8000/search',
                json={'query': query, 'top_k': 5}
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        results = [await r.json() for r in responses]
        return results

queries = [
    "contract termination",
    "payment terms",
    "liability clauses"
]

results = asyncio.run(batch_search(queries))
```

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

See [LICENSE](../LICENSE) for details.
