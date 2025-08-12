import faulthandler
import os
import signal
import subprocess
import sys
import threading
import time
import traceback


def setup_hang_detection() -> None:
    """Setup signal handlers and periodic dumps to help debug hangs in CI.

    - Enables faulthandler to dump Python stack traces on fatal signals
    - Installs handlers for SIGUSR1/2 to dump all thread stacks on demand
    - Starts a background thread that periodically dumps stacks
    """
    # Enable faulthandler for automatic stack dumps
    faulthandler.enable()

    def dump_all_stacks(signum, frame):  # type: ignore[no-redef]
        print(f"\n🔥 [HANG DEBUG] SIGNAL {signum} - DUMPING ALL THREAD STACKS:")
        faulthandler.dump_traceback()
        # Also dump current frames manually for completeness
        for thread_id, thread_frame in sys._current_frames().items():
            print(f"\n📍 Thread {thread_id}:")
            traceback.print_stack(thread_frame)

    def periodic_stack_dump() -> None:
        """Periodically dump stacks to catch where the process is stuck."""
        start_time = time.time()

        while True:
            time.sleep(120)  # Check every 2 minutes
            elapsed = time.time() - start_time

            print(f"\n⏰ [HANG DEBUG] Periodic check at {elapsed:.1f}s elapsed:")

            # Check for hanging processes and dump stacks
            try:
                import subprocess

                # Check for embedding servers that might be hanging
                result = subprocess.run(
                    ["pgrep", "-f", "embedding_server"], capture_output=True, text=True, timeout=5
                )
                if result.stdout.strip():
                    print(
                        f"📍 [HANG DEBUG] Found embedding server processes: {result.stdout.strip()}"
                    )

                # Check for zmq processes
                result = subprocess.run(
                    ["pgrep", "-f", "zmq"], capture_output=True, text=True, timeout=5
                )
                if result.stdout.strip():
                    print(f"📍 [HANG DEBUG] Found zmq processes: {result.stdout.strip()}")

            except Exception as e:
                print(f"📍 [HANG DEBUG] Process check failed: {e}")

            # Dump thread stacks every 4 minutes
            if elapsed > 240 and int(elapsed) % 240 < 120:
                print(f"\n⚠️ [HANG DEBUG] Stack dump at {elapsed:.1f}s:")
                for thread_id, thread_frame in sys._current_frames().items():
                    print(f"\n📍 Thread {thread_id}:")
                    traceback.print_stack(thread_frame)

            # Emergency exit after 8 minutes (should be handled by wrapper timeout)
            if elapsed > 480:
                print(
                    f"\n💥 [HANG DEBUG] Emergency exit after {elapsed:.1f}s - pytest taking too long!"
                )
                faulthandler.dump_traceback()
                # Try to cleanup before exit
                try:
                    import subprocess

                    subprocess.run(["pkill", "-9", "-f", "embedding_server"], timeout=2)
                    subprocess.run(["pkill", "-9", "-f", "zmq"], timeout=2)
                except Exception:
                    pass
                import os

                os._exit(124)  # Force exit with timeout code

    # Register signal handlers for external debugging
    signal.signal(signal.SIGUSR1, dump_all_stacks)
    signal.signal(signal.SIGUSR2, dump_all_stacks)

    # Start periodic dumping thread
    dump_thread = threading.Thread(target=periodic_stack_dump, daemon=True)
    dump_thread.start()


def main(argv: list[str]) -> int:
    setup_hang_detection()
    # Re-exec pytest with debugging enabled
    # Use Popen for better control over the subprocess
    print(f"🚀 [DEBUG] Starting pytest with args: {argv}")

    try:
        # Use Popen for non-blocking execution
        process = subprocess.Popen(
            [sys.executable, "-m", "pytest", *argv],
            stdout=sys.stdout,
            stderr=sys.stderr,
            # Use separate process group to avoid signal inheritance issues
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        # Monitor the process with a reasonable timeout
        start_time = time.time()
        timeout = 600  # 10 minutes
        poll_interval = 5  # seconds

        while True:
            # Check if process has completed
            return_code = process.poll()
            if return_code is not None:
                print(f"✅ [DEBUG] Pytest completed with return code: {return_code}")
                return return_code

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"💥 [DEBUG] Pytest timed out after {elapsed:.1f}s, terminating...")
                try:
                    # Try graceful termination first
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        # Force kill if still running
                        process.kill()
                        process.wait()

                    # Cleanup any remaining processes
                    subprocess.run(["pkill", "-9", "-f", "pytest"], timeout=5)
                    subprocess.run(["pkill", "-9", "-f", "embedding_server"], timeout=5)
                except Exception:
                    pass
                return 124  # timeout exit code

            # Wait before next check
            time.sleep(poll_interval)

    except Exception as e:
        print(f"💥 [DEBUG] Error running pytest: {e}")
        # Cleanup on error
        try:
            subprocess.run(["pkill", "-9", "-f", "pytest"], timeout=5)
            subprocess.run(["pkill", "-9", "-f", "embedding_server"], timeout=5)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
