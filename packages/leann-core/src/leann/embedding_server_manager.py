import atexit
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import psutil

# Set up logging based on environment variable
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _is_colab_environment() -> bool:
    """Check if we're running in Google Colab environment."""
    return "COLAB_GPU" in os.environ or "COLAB_TPU" in os.environ


def _get_available_port(start_port: int = 5557) -> int:
    """Get an available port starting from start_port."""
    port = start_port
    while port < start_port + 100:  # Try up to 100 ports
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + 100}")


def _check_port(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _check_process_matches_config(
    port: int, expected_model: str, expected_passages_file: str
) -> bool:
    """
    Check if the process using the port matches our expected model and passages file.
    Returns True if matches, False otherwise.
    """
    try:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            if not _is_process_listening_on_port(proc, port):
                continue

            cmdline = proc.info["cmdline"]
            if not cmdline:
                continue

            return _check_cmdline_matches_config(
                cmdline, port, expected_model, expected_passages_file
            )

        logger.debug(f"No process found listening on port {port}")
        return False

    except Exception as e:
        logger.warning(f"Could not check process on port {port}: {e}")
        return False


def _is_process_listening_on_port(proc, port: int) -> bool:
    """Check if a process is listening on the given port."""
    try:
        connections = proc.net_connections()
        for conn in connections:
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                return True
        return False
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def _check_cmdline_matches_config(
    cmdline: list, port: int, expected_model: str, expected_passages_file: str
) -> bool:
    """Check if command line matches our expected configuration."""
    cmdline_str = " ".join(cmdline)
    logger.debug(f"Found process on port {port}: {cmdline_str}")

    # Check if it's our embedding server
    is_embedding_server = any(
        server_type in cmdline_str
        for server_type in [
            "embedding_server",
            "leann_backend_diskann.embedding_server",
            "leann_backend_hnsw.hnsw_embedding_server",
        ]
    )

    if not is_embedding_server:
        logger.debug(f"Process on port {port} is not our embedding server")
        return False

    # Check model name
    model_matches = _check_model_in_cmdline(cmdline, expected_model)

    # Check passages file if provided
    passages_matches = _check_passages_in_cmdline(cmdline, expected_passages_file)

    result = model_matches and passages_matches
    logger.debug(
        f"model_matches: {model_matches}, passages_matches: {passages_matches}, overall: {result}"
    )
    return result


def _check_model_in_cmdline(cmdline: list, expected_model: str) -> bool:
    """Check if the command line contains the expected model."""
    if "--model-name" not in cmdline:
        return False

    model_idx = cmdline.index("--model-name")
    if model_idx + 1 >= len(cmdline):
        return False

    actual_model = cmdline[model_idx + 1]
    return actual_model == expected_model


def _check_passages_in_cmdline(cmdline: list, expected_passages_file: str) -> bool:
    """Check if the command line contains the expected passages file."""
    if "--passages-file" not in cmdline:
        return False  # Expected but not found

    passages_idx = cmdline.index("--passages-file")
    if passages_idx + 1 >= len(cmdline):
        return False

    actual_passages = cmdline[passages_idx + 1]
    expected_path = Path(expected_passages_file).resolve()
    actual_path = Path(actual_passages).resolve()
    return actual_path == expected_path


def _find_compatible_port_or_next_available(
    start_port: int, model_name: str, passages_file: str, max_attempts: int = 100
) -> tuple[int, bool]:
    """
    Find a port that either has a compatible server or is available.
    Returns (port, is_compatible) where is_compatible indicates if we found a matching server.
    """
    for port in range(start_port, start_port + max_attempts):
        if not _check_port(port):
            # Port is available
            return port, False

        # Port is in use, check if it's compatible
        if _check_process_matches_config(port, model_name, passages_file):
            logger.info(f"Found compatible server on port {port}")
            return port, True
        else:
            logger.info(f"Port {port} has incompatible server, trying next port...")

    raise RuntimeError(
        f"Could not find compatible or available port in range {start_port}-{start_port + max_attempts}"
    )


class EmbeddingServerManager:
    """
    A simplified manager for embedding server processes that avoids complex update mechanisms.
    """

    def __init__(self, backend_module_name: str):
        """
        Initializes the manager for a specific backend.

        Args:
            backend_module_name (str): The full module name of the backend's server script.
                                       e.g., "leann_backend_diskann.embedding_server"
        """
        self.backend_module_name = backend_module_name
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port: Optional[int] = None
        self._server_pid: Optional[int] = None  # Adopted server PID when not launched here
        self._atexit_registered = False
        # Also register a weakref finalizer to ensure cleanup when manager is GC'ed
        try:
            import weakref

            self._finalizer = weakref.finalize(self, self._finalize_process)
        except Exception:
            self._finalizer = None

    def start_server(
        self,
        port: int,
        model_name: str,
        embedding_mode: str = "sentence-transformers",
        **kwargs,
    ) -> tuple[bool, int]:
        """Start the embedding server."""
        passages_file = kwargs.get("passages_file")

        # Check if we have a compatible server already running
        if self._has_compatible_running_server(model_name, passages_file):
            logger.info("Found compatible running server!")
            return True, port

        # For Colab environment, use a different strategy
        if _is_colab_environment():
            logger.info("Detected Colab environment, using alternative startup strategy")
            return self._start_server_colab(port, model_name, embedding_mode, **kwargs)

        # In CI or when explicitly requested, skip compatibility scanning to avoid slow/hanging proc scans
        skip_compat = (
            os.environ.get("LEANN_SKIP_COMPAT", "").lower() in {"1", "true", "yes"}
            or os.environ.get("CI") == "true"
        )
        if skip_compat:
            try:
                actual_port = _get_available_port(port)
                is_compatible = False
            except RuntimeError:
                logger.error("No available ports found")
                return False, port
        else:
            # Find a compatible port or next available (may scan processes)
            actual_port, is_compatible = _find_compatible_port_or_next_available(
                port, model_name, passages_file
            )

        if is_compatible:
            logger.info(f"Found compatible server on port {actual_port}")
            # Adopt the existing server so we can clean it up on exit
            self._adopt_existing_server(actual_port, model_name, passages_file)
            return True, actual_port

        # Start a new server
        return self._start_new_server(actual_port, model_name, embedding_mode, **kwargs)

    def _start_server_colab(
        self,
        port: int,
        model_name: str,
        embedding_mode: str = "sentence-transformers",
        **kwargs,
    ) -> tuple[bool, int]:
        """Start server with Colab-specific configuration."""
        # Try to find an available port
        try:
            actual_port = _get_available_port(port)
        except RuntimeError:
            logger.error("No available ports found")
            return False, port

        logger.info(f"Starting server on port {actual_port} for Colab environment")

        # Use a simpler startup strategy for Colab
        command = self._build_server_command(actual_port, model_name, embedding_mode, **kwargs)

        try:
            # In Colab, we'll use a more direct approach
            self._launch_server_process_colab(command, actual_port)
            return self._wait_for_server_ready_colab(actual_port)
        except Exception as e:
            logger.error(f"Failed to start embedding server in Colab: {e}")
            return False, actual_port

    def _has_compatible_running_server(self, model_name: str, passages_file: str) -> bool:
        """Check if we have a compatible running server."""
        if not (self.server_process and self.server_process.poll() is None and self.server_port):
            return False

        if _check_process_matches_config(self.server_port, model_name, passages_file):
            logger.info(f"Existing server process (PID {self.server_process.pid}) is compatible")
            return True

        logger.info("Existing server process is incompatible. Should start a new server.")
        return False

    def _start_new_server(
        self, port: int, model_name: str, embedding_mode: str, **kwargs
    ) -> tuple[bool, int]:
        """Start a new embedding server on the given port."""
        logger.info(f"Starting embedding server on port {port}...")

        command = self._build_server_command(port, model_name, embedding_mode, **kwargs)

        try:
            self._launch_server_process(command, port)
            return self._wait_for_server_ready(port)
        except Exception as e:
            logger.error(f"Failed to start embedding server: {e}")
            return False, port

    def _build_server_command(
        self, port: int, model_name: str, embedding_mode: str, **kwargs
    ) -> list:
        """Build the command to start the embedding server."""
        command = [
            sys.executable,
            "-m",
            self.backend_module_name,
            "--zmq-port",
            str(port),
            "--model-name",
            model_name,
        ]

        if kwargs.get("passages_file"):
            # Convert to absolute path to ensure subprocess can find the file
            passages_file = Path(kwargs["passages_file"]).resolve()
            command.extend(["--passages-file", str(passages_file)])
        if embedding_mode != "sentence-transformers":
            command.extend(["--embedding-mode", embedding_mode])
        if kwargs.get("distance_metric"):
            command.extend(["--distance-metric", kwargs["distance_metric"]])

        return command

    def _launch_server_process(self, command: list, port: int) -> None:
        """Launch the server process."""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        logger.info(f"Command: {' '.join(command)}")

        # In CI environment, redirect stdout to avoid buffer deadlock but keep stderr for debugging
        # Embedding servers use many print statements that can fill stdout buffers
        is_ci = os.environ.get("CI") == "true"
        if is_ci:
            stdout_target = subprocess.DEVNULL
            stderr_target = None  # Keep stderr for error debugging in CI
            logger.info(
                "CI environment detected, redirecting embedding server stdout to DEVNULL, keeping stderr"
            )
        else:
            stdout_target = None  # Direct to console for visible logs
            stderr_target = None  # Direct to console for visible logs

        # Start embedding server subprocess
        self.server_process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=stdout_target,
            stderr=stderr_target,
        )
        self.server_port = port
        logger.info(f"Server process started with PID: {self.server_process.pid}")

        # Register atexit callback only when we actually start a process
        if not self._atexit_registered:
            # Use a lambda to avoid issues with bound methods
            atexit.register(lambda: self.stop_server() if self.server_process else None)
            self._atexit_registered = True
        # Touch finalizer so it knows there is a live process
        if getattr(self, "_finalizer", None) is not None and not self._finalizer.alive:
            try:
                import weakref

                self._finalizer = weakref.finalize(self, self._finalize_process)
            except Exception:
                pass

    def _wait_for_server_ready(self, port: int) -> tuple[bool, int]:
        """Wait for the server to be ready."""
        max_wait, wait_interval = 120, 0.5
        for _ in range(int(max_wait / wait_interval)):
            if _check_port(port):
                logger.info("Embedding server is ready!")
                return True, port

            if self.server_process and self.server_process.poll() is not None:
                logger.error("Server terminated during startup.")
                return False, port

            time.sleep(wait_interval)

        logger.error(f"Server failed to start within {max_wait} seconds.")
        self.stop_server()
        return False, port

    def stop_server(self):
        """Stops the embedding server process if it's running."""
        if not self.server_process and not self._server_pid:
            return

        # If we have an adopted PID and no Popen handle, try PID-based stop
        if not self.server_process and self._server_pid:
            try:
                proc = psutil.Process(self._server_pid)
                logger.info(f"Terminating adopted server process (PID: {self._server_pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    logger.info(f"Adopted server process {self._server_pid} terminated gracefully.")
                except psutil.TimeoutExpired:
                    logger.warning(
                        f"Adopted server process {self._server_pid} did not terminate within 5 seconds, force killing..."
                    )
                    proc.kill()
                finally:
                    self._server_pid = None
                    self.server_port = None
            except Exception as e:
                logger.warning(f"Failed to terminate adopted server PID {self._server_pid}: {e}")
                self._server_pid = None
                self.server_port = None
            return

        if self.server_process and self.server_process.poll() is not None:
            # Process already terminated
            self.server_process = None
            self.server_port = None
            return

        logger.info(
            f"Terminating server process (PID: {self.server_process.pid}) for backend {self.backend_module_name}..."
        )

        # Use simple termination - our improved server shutdown should handle this properly
        self.server_process.terminate()

        try:
            self.server_process.wait(timeout=5)  # Give more time for graceful shutdown
            logger.info(f"Server process {self.server_process.pid} terminated gracefully.")
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Server process {self.server_process.pid} did not terminate within 5 seconds, force killing..."
            )
            self.server_process.kill()
            try:
                self.server_process.wait(timeout=2)
                logger.info(f"Server process {self.server_process.pid} killed successfully.")
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Failed to kill server process {self.server_process.pid} - it may be hung"
                )

        # Clean up process resources with timeout to avoid CI hang
        try:
            # Use shorter timeout in CI environments
            is_ci = os.environ.get("CI") == "true"
            timeout = 3 if is_ci else 10
            self.server_process.wait(timeout=timeout)
            logger.info(f"Server process {self.server_process.pid} cleanup completed")
        except subprocess.TimeoutExpired:
            logger.warning(f"Process cleanup timeout after {timeout}s, proceeding anyway")
        except Exception as e:
            logger.warning(f"Error during process cleanup: {e}")
        finally:
            self.server_process = None
            self.server_port = None

    def _finalize_process(self) -> None:
        """Best-effort cleanup used by weakref.finalize/atexit."""
        try:
            self.stop_server()
        except Exception:
            pass

    def _adopt_existing_server(self, port: int, expected_model: str, passages_file: str) -> None:
        """Adopt an existing compatible server so that we can stop it on exit.

        This scans processes to find the PID listening on the given port that matches
        our expected config, then records its PID and registers atexit cleanup.
        """
        try:
            adopted_pid: Optional[int] = None
            for proc in psutil.process_iter(["pid", "cmdline"]):
                if not _is_process_listening_on_port(proc, port):
                    continue
                cmdline = proc.info.get("cmdline") or []
                if _check_cmdline_matches_config(cmdline, port, expected_model, passages_file):
                    adopted_pid = proc.info["pid"]
                    break

            if adopted_pid:
                self._server_pid = adopted_pid
                self.server_port = port
                logger.info(f"Adopted existing embedding server PID {adopted_pid} on port {port}")
                if not self._atexit_registered:
                    atexit.register(self._finalize_process)
                    self._atexit_registered = True
            else:
                logger.warning(f"Could not find PID to adopt for server on port {port}")
        except Exception as e:
            logger.warning(f"Failed to adopt existing server on port {port}: {e}")

    def _launch_server_process_colab(self, command: list, port: int) -> None:
        """Launch the server process with Colab-specific settings."""
        logger.info(f"Colab Command: {' '.join(command)}")

        # In Colab, we need to be more careful about process management
        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.server_port = port
        logger.info(f"Colab server process started with PID: {self.server_process.pid}")

        # Register atexit callback
        if not self._atexit_registered:
            atexit.register(lambda: self.stop_server() if self.server_process else None)
            self._atexit_registered = True

    def _wait_for_server_ready_colab(self, port: int) -> tuple[bool, int]:
        """Wait for the server to be ready with Colab-specific timeout."""
        max_wait, wait_interval = 30, 0.5  # Shorter timeout for Colab

        for _ in range(int(max_wait / wait_interval)):
            if _check_port(port):
                logger.info("Colab embedding server is ready!")
                return True, port

            if self.server_process and self.server_process.poll() is not None:
                # Check for error output
                stdout, stderr = self.server_process.communicate()
                logger.error("Colab server terminated during startup.")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                return False, port

            time.sleep(wait_interval)

        logger.error(f"Colab server failed to start within {max_wait} seconds.")
        self.stop_server()
        return False, port
