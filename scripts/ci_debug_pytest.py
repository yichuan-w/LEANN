import faulthandler
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
        time.sleep(300)  # Wait 5 minutes
        print(f"\n⏰ [HANG DEBUG] Periodic stack dump at {time.time()}:")
        for thread_id, thread_frame in sys._current_frames().items():
            print(f"\n📍 Thread {thread_id}:")
            traceback.print_stack(thread_frame)
        time.sleep(300)  # Wait another 5 minutes if still running
        print(f"\n⚠️ [HANG DEBUG] Final stack dump at {time.time()} (likely hanging):")
        faulthandler.dump_traceback()

    # Register signal handlers for external debugging
    signal.signal(signal.SIGUSR1, dump_all_stacks)
    signal.signal(signal.SIGUSR2, dump_all_stacks)

    # Start periodic dumping thread
    dump_thread = threading.Thread(target=periodic_stack_dump, daemon=True)
    dump_thread.start()


def main(argv: list[str]) -> int:
    setup_hang_detection()
    # Re-exec pytest with debugging enabled
    result = subprocess.run([sys.executable, "-m", "pytest", *argv])
    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
