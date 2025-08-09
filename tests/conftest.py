"""Global test configuration and cleanup fixtures."""

import faulthandler
import os
import signal
import time
from collections.abc import Generator

import pytest

# Enable faulthandler to dump stack traces
faulthandler.enable()


@pytest.fixture(scope="session", autouse=True)
def _ci_backtraces():
    """Dump stack traces before CI timeout to diagnose hanging."""
    if os.getenv("CI") == "true":
        # Dump stack traces 10s before the 180s timeout
        faulthandler.dump_traceback_later(170, repeat=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(scope="session", autouse=True)
def global_test_cleanup() -> Generator:
    """Global cleanup fixture that runs after all tests.

    This ensures all ZMQ connections and child processes are properly cleaned up,
    preventing the test runner from hanging on exit.
    """
    yield

    # Cleanup after all tests
    print("\n🧹 Running global test cleanup...")

    # 1. Force cleanup of any LeannSearcher instances
    try:
        import gc

        # Force garbage collection to trigger __del__ methods
        gc.collect()
        time.sleep(0.2)
    except Exception:
        pass

    # 2. Set ZMQ linger but DON'T term Context.instance()
    # Terminating the global instance can block if other code still has sockets
    try:
        import zmq

        # Just set linger on the global instance, don't terminate it
        ctx = zmq.Context.instance()
        ctx.linger = 0
        # Do NOT call ctx.term() or ctx.destroy() on the global instance!
        # That would block waiting for all sockets to close
    except Exception:
        pass

    # Kill any leftover child processes (including grandchildren)
    try:
        import psutil

        current_process = psutil.Process()
        # Get ALL descendants recursively
        children = current_process.children(recursive=True)

        if children:
            print(f"\n⚠️  Cleaning up {len(children)} leftover child processes...")

            # First try to terminate gracefully
            for child in children:
                try:
                    print(f"  Terminating {child.pid} ({child.name()})")
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Wait a bit for processes to terminate
            gone, alive = psutil.wait_procs(children, timeout=2)

            # Force kill any remaining processes
            for child in alive:
                try:
                    print(f"  Force killing process {child.pid} ({child.name()})")
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Final wait to ensure cleanup
            psutil.wait_procs(alive, timeout=1)
    except ImportError:
        # psutil not installed, try basic process cleanup
        try:
            # Send SIGTERM to all child processes
            os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: Error during process cleanup: {e}")

    # List any remaining threads (for debugging)
    try:
        import threading

        threads = [t for t in threading.enumerate() if t is not threading.main_thread()]
        if threads:
            print(f"\n⚠️  {len(threads)} non-main threads still running:")
            for t in threads:
                print(f"  - {t.name} (daemon={t.daemon})")
    except Exception:
        pass


@pytest.fixture
def auto_cleanup_searcher():
    """Fixture that automatically cleans up LeannSearcher instances."""
    searchers = []

    def register(searcher):
        """Register a searcher for cleanup."""
        searchers.append(searcher)
        return searcher

    yield register

    # Cleanup all registered searchers
    for searcher in searchers:
        try:
            searcher.cleanup()
        except Exception:
            pass

    # Force garbage collection
    import gc

    gc.collect()
    time.sleep(0.1)


@pytest.fixture(scope="session", autouse=True)
def _reap_children():
    """Reap all child processes at session end as a safety net."""
    yield

    # Final aggressive cleanup
    try:
        import psutil

        me = psutil.Process()
        kids = me.children(recursive=True)
        for p in kids:
            try:
                p.terminate()
            except Exception:
                pass

        _, alive = psutil.wait_procs(kids, timeout=2)
        for p in alive:
            try:
                p.kill()
            except Exception:
                pass
    except Exception:
        pass


@pytest.fixture(autouse=True)
def cleanup_after_each_test():
    """Cleanup after each test to prevent resource leaks."""
    yield

    # Force garbage collection to trigger any __del__ methods
    import gc

    gc.collect()

    # Give a moment for async cleanup
    time.sleep(0.1)


def pytest_configure(config):
    """Configure pytest with better timeout handling."""
    # Set default timeout method to thread if not specified
    if not config.getoption("--timeout-method", None):
        config.option.timeout_method = "thread"
