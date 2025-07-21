import threading
import time
import atexit
import socket
import subprocess
import sys
import zmq
import msgpack
from pathlib import Path
from typing import Optional, Dict
import select
import psutil


def _check_port(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _check_process_matches_config(
    port: int, expected_model: str, expected_passages_file: str = None
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
            
            return _check_cmdline_matches_config(cmdline, port, expected_model, expected_passages_file)
        
        print(f"DEBUG: No process found listening on port {port}")
        return False
    
    except Exception as e:
        print(f"WARNING: Could not check process on port {port}: {e}")
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
    cmdline: list, port: int, expected_model: str, expected_passages_file: str = None
) -> bool:
    """Check if command line matches our expected configuration."""
    cmdline_str = " ".join(cmdline)
    print(f"DEBUG: Found process on port {port}: {cmdline_str}")
    
    # Check if it's our embedding server
    is_embedding_server = any(server_type in cmdline_str for server_type in [
        "embedding_server",
        "leann_backend_diskann.embedding_server", 
        "leann_backend_hnsw.hnsw_embedding_server"
    ])
    
    if not is_embedding_server:
        print(f"DEBUG: Process on port {port} is not our embedding server")
        return False
    
    # Check model name
    model_matches = _check_model_in_cmdline(cmdline, expected_model)
    
    # Check passages file if provided
    passages_matches = _check_passages_in_cmdline(cmdline, expected_passages_file)
    
    result = model_matches and passages_matches
    print(f"DEBUG: model_matches: {model_matches}, passages_matches: {passages_matches}, overall: {result}")
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


def _check_passages_in_cmdline(cmdline: list, expected_passages_file: str = None) -> bool:
    """Check if the command line contains the expected passages file."""
    if not expected_passages_file:
        return True  # No passages file expected
    
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
    start_port: int, model_name: str, passages_file: str = None, max_attempts: int = 100
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
            print(f"✅ Found compatible server on port {port}")
            return port, True
        else:
            print(f"⚠️  Port {port} has incompatible server, trying next port...")

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
        self._atexit_registered = False

    def start_server(
        self,
        port: int,
        model_name: str,
        embedding_mode: str = "sentence-transformers",
        **kwargs,
    ) -> tuple[bool, int]:
        """
        Starts the embedding server process.

        Args:
            port (int): The preferred ZMQ port for the server.
            model_name (str): The name of the embedding model to use.
            **kwargs: Additional arguments for the server.

        Returns:
            tuple[bool, int]: (success, actual_port_used)
        """
        passages_file = kwargs.get("passages_file")

        # Check if we have a compatible running server
        if self._has_compatible_running_server(model_name, passages_file):
            return True, self.server_port

        # Find available port (compatible or free)
        try:
            actual_port, is_compatible = _find_compatible_port_or_next_available(
                port, model_name, passages_file
            )
        except RuntimeError as e:
            print(f"❌ {e}")
            return False, port

        if is_compatible:
            print(f"✅ Using existing compatible server on port {actual_port}")
            self.server_port = actual_port
            self.server_process = None  # We don't own this process
            return True, actual_port

        if actual_port != port:
            print(f"⚠️  Using port {actual_port} instead of {port}")

        # Start new server
        return self._start_new_server(actual_port, model_name, embedding_mode, **kwargs)

    def _has_compatible_running_server(self, model_name: str, passages_file: str) -> bool:
        """Check if we have a compatible running server."""
        if not (self.server_process and self.server_process.poll() is None and self.server_port):
            return False

        if _check_process_matches_config(self.server_port, model_name, passages_file):
            print(f"✅ Existing server process (PID {self.server_process.pid}) is compatible")
            return True
        
        print("⚠️  Existing server process is incompatible. Stopping it...")
        self.stop_server()
        return False

    def _start_new_server(self, port: int, model_name: str, embedding_mode: str, **kwargs) -> tuple[bool, int]:
        """Start a new embedding server on the given port."""
        print(f"INFO: Starting embedding server on port {port}...")

        command = self._build_server_command(port, model_name, embedding_mode, **kwargs)
        
        try:
            self._launch_server_process(command, port)
            return self._wait_for_server_ready(port)
        except Exception as e:
            print(f"❌ ERROR: Failed to start embedding server: {e}")
            return False, port

    def _build_server_command(self, port: int, model_name: str, embedding_mode: str, **kwargs) -> list:
        """Build the command to start the embedding server."""
        command = [
            sys.executable, "-m", self.backend_module_name,
            "--zmq-port", str(port),
            "--model-name", model_name,
        ]

        if kwargs.get("passages_file"):
            command.extend(["--passages-file", str(kwargs["passages_file"])])
        if embedding_mode != "sentence-transformers":
            command.extend(["--embedding-mode", embedding_mode])
        if kwargs.get("enable_warmup") is False:
            command.extend(["--disable-warmup"])

        return command

    def _launch_server_process(self, command: list, port: int) -> None:
        """Launch the server process."""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        print(f"INFO: Command: {' '.join(command)}")

        self.server_process = subprocess.Popen(
            command, cwd=project_root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", bufsize=1, universal_newlines=True,
        )
        self.server_port = port
        print(f"INFO: Server process started with PID: {self.server_process.pid}")
        
        # Register atexit callback only when we actually start a process
        if not self._atexit_registered:
            # Use a lambda to avoid issues with bound methods
            atexit.register(lambda: self.stop_server() if self.server_process else None)
            self._atexit_registered = True

    def _wait_for_server_ready(self, port: int) -> tuple[bool, int]:
        """Wait for the server to be ready."""
        max_wait, wait_interval = 120, 0.5
        for _ in range(int(max_wait / wait_interval)):
            if _check_port(port):
                print("✅ Embedding server is ready!")
                threading.Thread(target=self._log_monitor, daemon=True).start()
                return True, port
            
            if self.server_process.poll() is not None:
                print("❌ ERROR: Server terminated during startup.")
                self._print_recent_output()
                return False, port
            
            time.sleep(wait_interval)

        print(f"❌ ERROR: Server failed to start within {max_wait} seconds.")
        self.stop_server()
        return False, port

    def _print_recent_output(self):
        """Print any recent output from the server process."""
        if not self.server_process or not self.server_process.stdout:
            return
        try:
            if select.select([self.server_process.stdout], [], [], 0)[0]:
                output = self.server_process.stdout.read()
                if output:
                    print(f"[{self.backend_module_name} OUTPUT]: {output}")
        except Exception as e:
            print(f"Error reading server output: {e}")

    def _log_monitor(self):
        """Monitors and prints the server's stdout and stderr."""
        if not self.server_process:
            return
        try:
            if self.server_process.stdout:
                while True:
                    line = self.server_process.stdout.readline()
                    if not line:
                        break
                    print(
                        f"[{self.backend_module_name} LOG]: {line.strip()}", flush=True
                    )
        except Exception as e:
            print(f"Log monitor error: {e}")

    def stop_server(self):
        """Stops the embedding server process if it's running."""
        if not self.server_process:
            return
            
        if self.server_process.poll() is not None:
            # Process already terminated
            self.server_process = None
            return
            
        print(f"INFO: Terminating server process (PID: {self.server_process.pid}) for backend {self.backend_module_name}...")
        self.server_process.terminate()
        
        try:
            self.server_process.wait(timeout=5)
            print(f"INFO: Server process {self.server_process.pid} terminated.")
        except subprocess.TimeoutExpired:
            print(f"WARNING: Server process {self.server_process.pid} did not terminate gracefully, killing it.")
            self.server_process.kill()
        
        self.server_process = None
