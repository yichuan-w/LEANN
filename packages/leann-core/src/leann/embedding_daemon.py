"""Persistent embedding server daemon for LEANN.

Solves the cold-start problem (#166, #159) where the first search takes 30-60s
due to model loading.  ``leann serve`` starts a background daemon that keeps the
embedding model warm.  Subsequent ``leann search`` calls detect the daemon and
skip the expensive model-load step.

The daemon writes a state file to ``~/.leann/daemon.json`` containing the PID,
ZMQ port, model name, and a heartbeat timestamp.  Clients read this file to
connect to the running daemon.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default location for daemon state
_DAEMON_STATE_DIR = Path.home() / ".leann"
_DAEMON_STATE_FILE = _DAEMON_STATE_DIR / "daemon.json"

# Heartbeat interval (seconds)
_HEARTBEAT_INTERVAL = 30
# Stale threshold — if heartbeat is older than this, daemon is dead
_STALE_THRESHOLD = 90


def _get_state_file() -> Path:
    """Return the path to the daemon state file."""
    return Path(os.environ.get("LEANN_DAEMON_STATE", str(_DAEMON_STATE_FILE)))


def _is_port_open(port: int, host: str = "localhost") -> bool:
    """Check whether a TCP port is accepting connections."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def read_daemon_state() -> Optional[dict]:
    """Read the daemon state file and validate it.

    Returns the state dict if a healthy daemon is running, else ``None``.
    """
    state_file = _get_state_file()
    if not state_file.exists():
        return None

    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    pid = state.get("pid")
    port = state.get("port")
    heartbeat = state.get("heartbeat", 0)

    if not pid or not port:
        return None

    # Check if the daemon process is still alive
    if not _is_pid_alive(pid):
        logger.debug("Daemon PID %d is not alive, removing stale state file", pid)
        _remove_state_file()
        return None

    # Check heartbeat freshness
    age = time.time() - heartbeat
    if age > _STALE_THRESHOLD:
        logger.debug("Daemon heartbeat is %.0fs old (stale), ignoring", age)
        _remove_state_file()
        return None

    # Verify the port is actually open
    if not _is_port_open(port):
        logger.debug("Daemon port %d is not open, removing stale state file", port)
        _remove_state_file()
        return None

    return state


def _write_state(state: dict) -> None:
    """Write daemon state atomically."""
    state_file = _get_state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(state_file)


def _remove_state_file() -> None:
    """Remove the daemon state file."""
    try:
        _get_state_file().unlink(missing_ok=True)
    except OSError:
        pass


def stop_daemon() -> bool:
    """Stop a running daemon.  Returns True if a daemon was stopped."""
    state = read_daemon_state()
    if state is None:
        # Also try reading raw file (daemon might be unhealthy but still running)
        state_file = _get_state_file()
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    if state is None:
        return False

    pid = state.get("pid")
    if pid and _is_pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait briefly for graceful shutdown
            for _ in range(20):
                if not _is_pid_alive(pid):
                    break
                time.sleep(0.25)
            else:
                # Force kill if still alive
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        except OSError:
            pass

    _remove_state_file()
    return True


def daemon_status() -> Optional[dict]:
    """Return current daemon status, or None if not running."""
    return read_daemon_state()


def run_daemon(
    model_name: str,
    embedding_mode: str = "sentence-transformers",
    port: int = 5557,
    provider_options: Optional[dict] = None,
    passages_file: Optional[str] = None,
    foreground: bool = False,
) -> None:
    """Start the embedding daemon.

    In foreground mode, blocks until interrupted.  In background mode, forks a
    child process and returns immediately.

    Args:
        passages_file: Path to the index meta.json file. Required for
            recompute mode — the HNSW embedding server uses it to resolve
            passage IDs during graph construction.
    """
    if not foreground:
        _start_background(model_name, embedding_mode, port, provider_options, passages_file)
        return

    _run_foreground(model_name, embedding_mode, port, provider_options, passages_file)


def _start_background(
    model_name: str,
    embedding_mode: str,
    port: int,
    provider_options: Optional[dict],
    passages_file: Optional[str] = None,
) -> None:
    """Fork a background daemon process."""
    import subprocess

    from .settings import encode_provider_options

    cmd = [
        sys.executable,
        "-m",
        "leann.embedding_daemon",
        "--model-name",
        model_name,
        "--embedding-mode",
        embedding_mode,
        "--port",
        str(port),
    ]
    if passages_file:
        cmd.extend(["--passages-file", str(Path(passages_file).resolve())])

    env = os.environ.copy()
    encoded = encode_provider_options(provider_options)
    if encoded:
        env["LEANN_EMBEDDING_OPTIONS"] = encoded

    # Log stderr to a file so daemon startup failures can be diagnosed.
    log_dir = _get_state_file().parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "daemon.log"
    log_fh = open(log_path, "a")  # noqa: SIM115

    # Start as a detached subprocess
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        env=env,
        start_new_session=True,
    )
    # The child inherited the fd — close the parent's copy to avoid a leak.
    log_fh.close()

    # Wait briefly and verify it started
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError(
            f"Daemon process exited immediately with code {proc.returncode}. "
            f"Check {log_path} for details."
        )

    # Wait for the state file and port to become ready
    for _ in range(60):
        state = read_daemon_state()
        if state and state.get("port"):
            return
        time.sleep(1)

    raise RuntimeError(
        f"Daemon did not become ready within 60 seconds. Check {log_path} for details."
    )


def _run_foreground(
    model_name: str,
    embedding_mode: str,
    port: int,
    provider_options: Optional[dict],
    passages_file: Optional[str] = None,
) -> None:
    """Run the daemon in the foreground (blocking)."""
    from .embedding_server_manager import EmbeddingServerManager, _get_available_port

    # Find an available port
    try:
        actual_port = _get_available_port(port)
    except RuntimeError:
        logger.error("No available ports starting from %d", port)
        sys.exit(1)

    logger.info("Starting persistent embedding daemon on port %d...", actual_port)

    # Determine the backend module for the embedding server.
    # Try DiskANN first if available, fall back to HNSW.
    backend_module = "leann_backend_hnsw.hnsw_embedding_server"
    try:
        import importlib
        importlib.import_module("leann_backend_diskann")
        # DiskANN available — but HNSW embedding server is the universal one
        # that works for both backends' ZMQ protocol.
    except ImportError:
        pass

    manager = EmbeddingServerManager(backend_module_name=backend_module)

    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        # Don't call sys.exit() from signal handler — it raises SystemExit
        # during arbitrary code, which can corrupt state. Just set the flag
        # and let the main loop exit cleanly.

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Start the embedding server
    server_kwargs: dict[str, Any] = {
        "port": actual_port,
        "model_name": model_name,
        "embedding_mode": embedding_mode,
        "provider_options": provider_options,
    }
    if passages_file:
        server_kwargs["passages_file"] = str(Path(passages_file).resolve())

    started, ready_port = manager.start_server(**server_kwargs)

    if not started:
        logger.error("Failed to start embedding server")
        sys.exit(1)

    logger.info("Embedding daemon ready on port %d (model: %s)", ready_port, model_name)

    # Write state file
    state: dict[str, Any] = {
        "pid": os.getpid(),
        "port": ready_port,
        "model_name": model_name,
        "embedding_mode": embedding_mode,
        "started_at": time.time(),
        "heartbeat": time.time(),
    }
    if passages_file:
        state["passages_file"] = str(Path(passages_file).resolve())
    _write_state(state)

    # Heartbeat loop — keep updating the state file so clients know we're alive
    try:
        while not shutdown_requested:
            time.sleep(_HEARTBEAT_INTERVAL)
            if shutdown_requested:
                break
            # Check if the server subprocess is still alive
            if manager.server_process and manager.server_process.poll() is not None:
                logger.error("Embedding server process died unexpectedly")
                break
            state["heartbeat"] = time.time()
            _write_state(state)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_server()
        _remove_state_file()
        logger.info("Embedding daemon shut down cleanly")


# ---------------------------------------------------------------------------
# CLI entry point (python -m leann.embedding_daemon)
# ---------------------------------------------------------------------------

def _main():
    """Entry point when run as ``python -m leann.embedding_daemon``."""
    import argparse

    parser = argparse.ArgumentParser(description="LEANN persistent embedding daemon")
    parser.add_argument("--model-name", required=True, help="Embedding model name")
    parser.add_argument(
        "--embedding-mode",
        default="sentence-transformers",
        help="Embedding mode (default: sentence-transformers)",
    )
    parser.add_argument("--port", type=int, default=5557, help="ZMQ port (default: 5557)")
    parser.add_argument(
        "--passages-file",
        default=None,
        help="Path to index meta.json (required for recompute mode)",
    )
    args = parser.parse_args()

    # Read provider options from environment
    provider_options = None
    raw = os.environ.get("LEANN_EMBEDDING_OPTIONS")
    if raw:
        try:
            provider_options = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _run_foreground(
        args.model_name, args.embedding_mode, args.port, provider_options, args.passages_file
    )


if __name__ == "__main__":
    _main()
