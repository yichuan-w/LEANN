"""
Test timeout utilities for CI environments.
"""

import functools
import os
import signal
import sys
from typing import Any, Callable


def timeout_test(seconds: int = 30):
    """
    Decorator to add timeout to test functions, especially useful in CI environments.

    Args:
        seconds: Timeout in seconds (default: 30)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Only apply timeout in CI environment
            if os.environ.get("CI") != "true":
                return func(*args, **kwargs)

            # Set up timeout handler
            def timeout_handler(signum, frame):
                print(f"\n❌ Test {func.__name__} timed out after {seconds} seconds in CI!")
                print("This usually indicates a hanging process or infinite loop.")
                # Try to cleanup any hanging processes
                try:
                    import subprocess

                    subprocess.run(
                        ["pkill", "-f", "embedding_server"], capture_output=True, timeout=2
                    )
                    subprocess.run(
                        ["pkill", "-f", "hnsw_embedding"], capture_output=True, timeout=2
                    )
                except Exception:
                    pass
                # Exit with timeout code
                sys.exit(124)  # Standard timeout exit code

            # Set signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except Exception:
                signal.alarm(0)  # Cancel alarm on exception
                raise
            finally:
                # Restore original handler
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


def ci_timeout(seconds: int = 60):
    """
    Timeout decorator specifically for CI environments.
    Uses threading for more reliable timeout handling.

    Args:
        seconds: Timeout in seconds (default: 60)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Only apply in CI
            if os.environ.get("CI") != "true":
                return func(*args, **kwargs)

            import threading

            result = [None]
            exception = [None]
            finished = threading.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
                finally:
                    finished.set()

            # Start function in thread
            thread = threading.Thread(target=target, daemon=True)
            thread.start()

            # Wait for completion or timeout
            if not finished.wait(timeout=seconds):
                print(f"\n💥 CI TIMEOUT: Test {func.__name__} exceeded {seconds}s limit!")
                print("This usually indicates hanging embedding servers or infinite loops.")

                # Try to cleanup embedding servers
                try:
                    import subprocess

                    subprocess.run(
                        ["pkill", "-9", "-f", "embedding_server"], capture_output=True, timeout=2
                    )
                    subprocess.run(
                        ["pkill", "-9", "-f", "hnsw_embedding"], capture_output=True, timeout=2
                    )
                    print("Attempted to kill hanging embedding servers.")
                except Exception as e:
                    print(f"Cleanup failed: {e}")

                # Raise TimeoutError instead of sys.exit for better pytest integration
                raise TimeoutError(f"Test {func.__name__} timed out after {seconds} seconds")

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator
