# Local Cursor Proxy

`leann cursor` starts a local OpenAI-compatible HTTP proxy that can be used by
Cursor or other local coding assistants. The proxy retrieves relevant snippets
from a LEANN code index, injects them into the chat request, and forwards the
augmented request to a local OpenAI-compatible model server such as Ollama or
LM Studio.

By default the proxy binds to `127.0.0.1`, allows only localhost browser
origins for Cross-Origin Resource Sharing (CORS), and supports both normal and
streaming `/v1/chat/completions` responses.

## Build A Code Index

```bash
leann build my-code --docs ./src --file-types .py,.md --use-ast-chunking
```

## Start The Proxy

```bash
leann cursor --index my-code --model qwen3-coder
```

Use a non-default local model server with:

```bash
leann cursor \
  --index my-code \
  --model codestral \
  --llm-base-url http://127.0.0.1:1234
```

Then configure an OpenAI-compatible client with:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8765/v1
```

If the client runs in a browser on a custom localhost port, pass an explicit
allowed origin:

```bash
leann cursor --index my-code --allow-origin http://localhost:3000
```

Only bind to a non-local interface when you have separately protected the
machine/network:

```bash
leann cursor --bind-host 0.0.0.0 --allow-origin http://localhost:3000
```
