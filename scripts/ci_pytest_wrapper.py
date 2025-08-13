#!/usr/bin/env python3
"""
CI pytest wrapper with comprehensive hang detection and cleanup.
Designed to prevent CI hangs due to subprocess or cleanup issues.
"""

import os
import signal
import subprocess
import sys
import time


def cleanup_all_processes():
    """Aggressively cleanup all related processes."""
    print("🧹 [CLEANUP] Performing aggressive cleanup...")

    # Kill by pattern - use separate calls to avoid shell injection
    # Avoid killing ourselves by being more specific
    patterns = [
        "embedding_server",
        "hnsw_embedding",
        "zmq",
    ]

    for pattern in patterns:
        try:
            subprocess.run(["pkill", "-9", "-f", pattern], timeout=5, capture_output=True)
        except Exception:
            pass

    # Clean up specific pytest processes but NOT the wrapper itself
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
        lines = result.stdout.split("\n")
        current_pid = str(os.getpid())

        for line in lines:
            # Skip our own process
            if current_pid in line:
                continue
            # Only kill actual pytest processes, not wrapper processes
            if (
                "python" in line
                and "pytest" in line
                and "ci_pytest_wrapper.py" not in line
                and "ci_debug_pytest.py" not in line
            ):
                try:
                    pid = line.split()[1]
                    subprocess.run(["kill", "-9", pid], timeout=2)
                except Exception:
                    pass
    except Exception:
        pass

    print("🧹 [CLEANUP] Cleanup completed")


def run_pytest_with_monitoring(pytest_args):
    """Run pytest with comprehensive monitoring and timeout handling."""

    # Pre-test cleanup
    print("🧹 [WRAPPER] Pre-test cleanup...")
    cleanup_all_processes()
    time.sleep(2)

    # Show pre-test state
    print("📊 [WRAPPER] Pre-test process state:")
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
        relevant_lines = [
            line
            for line in result.stdout.split("\n")
            if "python" in line or "embedding" in line or "zmq" in line
        ]
        if relevant_lines:
            for line in relevant_lines[:5]:  # Show first 5 matches
                print(f"  {line}")
        else:
            print("  No relevant processes found")
    except Exception:
        print("  Process check failed")

    # Setup signal handlers for cleanup
    def signal_handler(signum, frame):
        print(f"\n💥 [WRAPPER] Received signal {signum}, cleaning up...")
        cleanup_all_processes()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run pytest with monitoring
    print(f"🚀 [WRAPPER] Starting pytest with args: {pytest_args}")

    try:
        # Use Popen for better control
        cmd = [sys.executable, "scripts/ci_debug_pytest.py", *pytest_args]
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        # Monitor with timeout
        start_time = time.time()
        timeout = 600  # 10 minutes
        monitor_interval = 10  # Check every 10 seconds

        while True:
            # Check if process completed
            return_code = process.poll()
            if return_code is not None:
                print(f"✅ [WRAPPER] Pytest completed with return code: {return_code}")
                break

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"💥 [WRAPPER] Pytest timed out after {elapsed:.1f}s")

                # Try graceful termination
                try:
                    print("🔄 [WRAPPER] Attempting graceful termination...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        print("💀 [WRAPPER] Graceful termination failed, force killing...")
                        process.kill()
                        process.wait()
                except Exception as e:
                    print(f"⚠️ [WRAPPER] Error during termination: {e}")

                return_code = 124  # timeout exit code
                break

            # Monitor progress
            if int(elapsed) % 30 == 0:  # Every 30 seconds
                print(f"📊 [WRAPPER] Monitor check: {elapsed:.0f}s elapsed, pytest still running")

            time.sleep(monitor_interval)

        # Post-test cleanup verification
        print("🔍 [WRAPPER] Post-test cleanup verification...")
        time.sleep(2)

        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
            remaining = [
                line
                for line in result.stdout.split("\n")
                if "python" in line and ("pytest" in line or "embedding" in line)
            ]

            if remaining:
                print(f"⚠️ [WRAPPER] Found {len(remaining)} remaining processes:")
                for line in remaining[:3]:  # Show first 3
                    print(f"  {line}")
                print("💀 [WRAPPER] Performing final cleanup...")
                cleanup_all_processes()
            else:
                print("✅ [WRAPPER] No remaining processes found")
        except Exception:
            print("⚠️ [WRAPPER] Post-test verification failed, performing cleanup anyway")
            cleanup_all_processes()

        return return_code

    except Exception as e:
        print(f"💥 [WRAPPER] Error running pytest: {e}")
        cleanup_all_processes()
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: ci_pytest_wrapper.py <pytest_args...>")
        return 1

    pytest_args = sys.argv[1:]
    print(f"🎯 [WRAPPER] CI pytest wrapper starting with args: {pytest_args}")

    return run_pytest_with_monitoring(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
