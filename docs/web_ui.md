# LEANN Web UI

`leann serve` starts the local HTTP API and serves a lightweight browser UI from the same
FastAPI app.

```bash
uv pip install "leann-core[server]"
leann serve
```

Open `http://127.0.0.1:8000/` to list project indexes, select an index, and run search queries.
The UI uses the existing `/indexes` and `/indexes/{index}/search` API endpoints, so it works from
the project root that contains `.leann/indexes`.

By default the server binds to `127.0.0.1`. Use `--host` and `--port` when you need a different
local address:

```bash
leann serve --host 127.0.0.1 --port 8080
```
