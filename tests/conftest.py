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
    try:
        import zmq

        # Set a very short linger on any remaining contexts
        # This prevents blocking on context termination
        ctx = zmq.Context.instance()
        ctx.linger = 0
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
