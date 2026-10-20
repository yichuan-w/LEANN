import json
import os
import time
from pathlib import Path

from mitmproxy import http

TASK_ID = os.environ.get("TASK_INSTANCE", "unknown_task")
TRACE_DIR = Path(
    os.environ.get("TRACE_DIR", Path(__file__).resolve().parents[1] / "traces" / "raw")
)
TRACE_DIR.mkdir(parents=True, exist_ok=True)
HOST_FILTER = os.environ.get("TRACE_HOST_FILTER", "").strip().lower()

OUTPUT_FILE = TRACE_DIR / f"{TASK_ID}_trace.jsonl"


def _should_record(flow: http.HTTPFlow) -> bool:
    host = (flow.request.host or "").lower()

    # filter out telemetry traffic
    if "statsig" in host:
        return False

    if HOST_FILTER and HOST_FILTER not in host:
        return False

    return True


def request(flow: http.HTTPFlow):
    if "statsig" in flow.request.pretty_host:
        flow.response = http.Response.make(204)
        return


def response(flow: http.HTTPFlow):
    should = _should_record(flow)

    if not should:
        return

    if flow.response and flow.response.stream:
        flow.response.stream = False

    try:
        entry = {
            "task_id": TASK_ID,
            "timestamp": time.time(),
            "id": flow.id,
            "request": {
                "method": flow.request.method,
                "url": flow.request.pretty_url,
                "headers": dict(flow.request.headers),
                "text": flow.request.get_text(strict=False) or "",
            },
            "response": {
                "status_code": flow.response.status_code if flow.response else None,
                "headers": dict(flow.response.headers) if flow.response else {},
                "text": flow.response.get_text(strict=False) if flow.response else "",
            },
        }

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"DEBUG: Error recording trace: {e}")
        import traceback

        traceback.print_exc()
        pass
