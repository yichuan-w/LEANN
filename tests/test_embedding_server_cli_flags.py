import os
import subprocess
import sys
from pathlib import Path


def _run_help(module_name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    extra_paths = [
        str(root / "packages" / "leann-core" / "src"),
        str(root / "packages" / "leann-backend-hnsw"),
        str(root / "packages" / "leann-backend-diskann"),
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))

    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return proc.stdout


def test_hnsw_server_help_has_daemon_and_warmup_flags():
    out = _run_help("leann_backend_hnsw.hnsw_embedding_server")
    assert "--enable-warmup" in out
    assert "--daemon-mode" in out
    assert "--daemon-ttl" in out


def test_diskann_server_help_has_daemon_and_warmup_flags():
    out = _run_help("leann_backend_diskann.diskann_embedding_server")
    assert "--enable-warmup" in out
    assert "--daemon-mode" in out
    assert "--daemon-ttl" in out
