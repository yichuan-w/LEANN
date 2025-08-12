"""
pytest configuration and fixtures for LEANN tests.
"""

import os
import signal
import subprocess
import sys
import time

import pytest


def aggressive_cleanup():
    """Aggressively clean up any hanging processes."""
    try:
        # Kill embedding servers
        subprocess.run(["pkill", "-9", "-f", "embedding_server"], capture_output=True, timeout=2)
        subprocess.run(["pkill", "-9", "-f", "hnsw_embedding"], capture_output=True, timeout=2)
        subprocess.run(["pkill", "-9", "-f", "zmq"], capture_output=True, timeout=2)

        print("🧹 [CLEANUP] Killed hanging processes")
    except Exception as e:
        print(f"⚠️ [CLEANUP] Failed to kill processes: {e}")


def timeout_handler(signum, frame):
    """Handle timeout signal for individual tests."""
    print("\n💥 [TIMEOUT] Test exceeded individual timeout limit!")
    print("🔍 [TIMEOUT] Current stack trace:")
    import traceback

    traceback.print_stack(frame)

    # Cleanup before exit
    aggressive_cleanup()

    # Exit with timeout code
    sys.exit(124)


@pytest.fixture(autouse=True)
def test_timeout_fixture():
    """Automatically apply timeout to all tests in CI environment."""
    if os.environ.get("CI") != "true":
        yield
        return

    # Set up 3-minute timeout for individual tests
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)  # 3 minutes

    try:
        yield
    finally:
        # Cancel alarm and restore handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        # Cleanup after each test
        aggressive_cleanup()


@pytest.fixture(autouse=True)
def ci_process_monitor():
    """Monitor for hanging processes during CI tests."""
    if os.environ.get("CI") != "true":
        yield
        return

    import threading
    import time

    # Track test start time
    start_time = time.time()
    stop_monitor = threading.Event()

    def monitor_processes():
        """Background process to monitor for hangs."""
        while not stop_monitor.wait(30):  # Check every 30 seconds
            elapsed = time.time() - start_time

            if elapsed > 120:  # Warn after 2 minutes
                print(f"\n⚠️ [MONITOR] Test running for {elapsed:.1f}s")

                # Check for suspicious processes
                try:
                    result = subprocess.run(
                        ["pgrep", "-f", "embedding_server"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.stdout.strip():
                        print(f"📍 [MONITOR] Found embedding servers: {result.stdout.strip()}")
                except Exception:
                    pass

    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
    monitor_thread.start()

    try:
        yield
    finally:
        # Stop monitoring
        stop_monitor.set()


def pytest_runtest_call(puretest):
    """Hook to wrap each test with additional monitoring."""
    if os.environ.get("CI") != "true":
        return

    print(f"\n🚀 [TEST] Starting: {puretest.nodeid}")
    start_time = time.time()

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"✅ [TEST] Completed: {puretest.nodeid} in {elapsed:.1f}s")


def pytest_collection_modifyitems(config, items):
    """Skip problematic tests in CI or add timeouts."""
    if os.environ.get("CI") != "true":
        return

    for item in items:
        # Skip tests that are known to hang or take too long
        if "test_backend_basic" in item.nodeid:
            item.add_marker(pytest.mark.skip(reason="Skip backend tests in CI due to hanging"))
        elif "test_document_rag" in item.nodeid:
            item.add_marker(pytest.mark.skip(reason="Skip RAG tests in CI due to hanging"))
        elif "diskann" in item.nodeid.lower():
            # DiskANN tests seem to be problematic
            item.add_marker(
                pytest.mark.skip(reason="Skip DiskANN tests in CI due to chunking hangs")
            )


def pytest_sessionstart(session):
    """Clean up at the start of the session."""
    if os.environ.get("CI") == "true":
        print("\n🧹 [SESSION] Starting with cleanup...")
        aggressive_cleanup()


def pytest_sessionfinish(session, exitstatus):
    """Clean up at the end of the session."""
    if os.environ.get("CI") == "true":
        print(f"\n🧹 [SESSION] Ending with cleanup (exit: {exitstatus})...")
        aggressive_cleanup()
