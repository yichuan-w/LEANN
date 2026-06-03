from __future__ import annotations

import contextlib
import hashlib
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

_INDEX_LOCKS_GUARD = threading.Lock()
_INDEX_LOCKS: dict[str, threading.RLock] = {}
_WINDOWS_LOCK_TIMEOUT_SECONDS = 300


def _lock_key(index_dir: Path) -> str:
    try:
        resolved = index_dir.resolve()
    except OSError:
        resolved = index_dir
    return hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:24]


def _lock_path(index_dir: Path, key: str) -> Path:
    lock_root = index_dir.parent / ".leann-locks"
    return lock_root / f"{key}.write.lock"


def _flock_acquire(lock_file: TextIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        lock_file.seek(0, 2)
        if lock_file.tell() == 0:
            lock_file.write("\n")
            lock_file.flush()
        lock_file.seek(0)
        deadline = time.monotonic() + _WINDOWS_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring index write lock after "
                        f"{_WINDOWS_LOCK_TIMEOUT_SECONDS}s"
                    )
                time.sleep(0.25)

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _flock_release(lock_file: TextIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def index_write_lock(index_dir: Path) -> Iterator[None]:
    """Serialize writes to one LEANN index across threads and processes."""
    key = _lock_key(index_dir)
    with _INDEX_LOCKS_GUARD:
        thread_lock = _INDEX_LOCKS.setdefault(key, threading.RLock())

    with thread_lock:
        lock_path = _lock_path(index_dir, key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            _flock_acquire(lock_file)
            try:
                index_dir.mkdir(parents=True, exist_ok=True)
                yield
            finally:
                _flock_release(lock_file)
