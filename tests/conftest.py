"""Global test configuration and cleanup fixtures."""

import os
import signal
import time
from collections.abc import Generator

import pytest


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

    # 2. Terminate ZMQ contexts more aggressively
    try:
        import zmq

        # Get the global instance and destroy it
        ctx = zmq.Context.instance()
        ctx.linger = 0

        # Force termination - this is aggressive but needed for CI
        try:
            ctx.destroy(linger=0)
        except Exception:
            pass

        # Also try to terminate the default context
        try:
            zmq.Context.term(zmq.Context.instance())
        except Exception:
            pass
    except Exception:
        pass

    # Kill any leftover child processes
    try:
        import psutil

        current_process = psutil.Process()
        children = current_process.children(recursive=True)

        if children:
            print(f"\n⚠️  Cleaning up {len(children)} leftover child processes...")

            # First try to terminate gracefully
            for child in children:
                try:
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
