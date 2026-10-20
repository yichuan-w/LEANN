# ContextBench LEANN Runner

This directory keeps a small local runner around the upstream ContextBench repo.

## Kept Files

- `contextbench_official_repo/`: upstream ContextBench code and data.
- `scripts/*.py`: local preparation, run, and evaluation scripts.
- `mitmproxy_addons/trace_recorder.py`: HTTP trace recorder used while Claude runs.
- `requirements-run.txt`: extra Python dependencies for these local scripts.

Generated directories such as `.venv/`, `.mitmproxy-venv/`, `traces/`,
`logs/`, `scripts/contextbench_work_dir_*`, and `scripts/contextbench_eval_repos/`
can be deleted and regenerated.

## 1. Create Python Environment

Run from this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r contextbench_official_repo/requirements.txt
pip install -r requirements-run.txt
```

## 2. Install Runtime CLIs

Install LEANN:

```bash
uv tool install leann-core --with leann
```

Install `mitmdump` in a separate environment:

```bash
python3.11 -m venv .mitmproxy-venv
.mitmproxy-venv/bin/python -m pip install mitmproxy
```

The run script also expects:

- `claude` CLI available on `PATH`.
- Node/npm available for `npx ccusage`.
- A Claude login session or `ANTHROPIC_API_KEY` in the environment.
- If using LEANN MCP mode, a Claude MCP server named `leann-server` or
  `LEANN_MCP_SERVER`/`CLAUDE_MCP_CONFIG_PATH` configured accordingly.

## 3. Prepare Repos And LEANN Indexes

```bash
cd scripts
WORK_ROOT=contextbench_work_dir_claude python prepare_repos_with_leann.py
```

## 4. Run Selected Tasks

```bash
cd scripts
LEANN_ENABLED=1 \
WORK_ROOT=contextbench_work_dir_claude \
OUTPUT_FILE=all_predictions_claude.jsonl \
python batch_run_selected.py
```

Run without LEANN:

```bash
LEANN_ENABLED=0 \
WORK_ROOT=contextbench_work_dir_claude \
OUTPUT_FILE=all_predictions_claude_baseline.jsonl \
python batch_run_selected.py
```

Run specific IDs without editing the script:

```bash
SELECTED_IDS=id1,id2 python batch_run_selected.py
```

## 5. Evaluate Results

Context retrieval metrics:

```bash
cd ".../contextbench_official_repo"

PYTHONPATH=. python -m contextbench.evaluate \
  --gold data/full.parquet \
  --pred "../scripts/all_predictions_claude.jsonl" \
  --cache "../scripts/contextbench_eval_repos" \
  --out "../scripts/contextbench_official_eval_claude.jsonl" \
  2>&1 | tee "../scripts/contextbench_official_eval_claude.log"
```

## 6. Clean Generated Files

```bash
rm -rf .venv .mitmproxy-venv .eval-venv .leann .pycache_tmp logs traces
rm -rf scripts/.leann scripts/scripts
rm -rf scripts/contextbench_eval_repos scripts/contextbench_work_dir_claude scripts/contextbench_work_dir_claude_overlap160
```
