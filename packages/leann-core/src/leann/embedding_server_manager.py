
import os
import threading
import time
import atexit
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

def _check_port(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class EmbeddingServerManager:
    """
    A generic manager for handling the lifecycle of a backend-specific embedding server process.
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
        atexit.register(self.stop_server)

    def start_server(self, port: int, model_name: str, **kwargs) -> bool:
        """
        Starts the embedding server process.

        Args:
            port (int): The ZMQ port for the server.
            model_name (str): The name of the embedding model to use.
            **kwargs: Additional arguments for the server (e.g., passages_file, distance_metric).

        Returns:
            bool: True if the server is started successfully or already running, False otherwise.
        """
        if self.server_process and self.server_process.poll() is None:
            print(f"INFO: Reusing existing server process for this session (PID {self.server_process.pid})")
            return True

        if _check_port(port):
            print(f"WARNING: Port {port} is already in use. Assuming an external server is running.")
            return True

        print(f"INFO: Starting session-level embedding server for '{self.backend_module_name}'...")

        try:
            command = [
                sys.executable,
                "-m", self.backend_module_name,
                "--zmq-port", str(port),
                "--model-name", model_name
            ]

            # Add extra arguments for specific backends
            if "passages_file" in kwargs and kwargs["passages_file"]:
                command.extend(["--passages-file", str(kwargs["passages_file"])])
            # if "distance_metric" in kwargs and kwargs["distance_metric"]:
            #     command.extend(["--distance-metric", kwargs["distance_metric"]])

            project_root = Path(__file__).parent.parent.parent.parent.parent
            print(f"INFO: Running command from project root: {project_root}")

            self.server_process = subprocess.Popen(
                command,
                cwd=project_root,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            self.server_port = port
            print(f"INFO: Server process started with PID: {self.server_process.pid}")

            max_wait, wait_interval = 30, 0.5
            for _ in range(int(max_wait / wait_interval)):
                if _check_port(port):
                    print(f"✅ Embedding server is up and ready for this session.")
                    log_thread = threading.Thread(target=self._log_monitor, daemon=True)
                    log_thread.start()
                    return True
                if self.server_process.poll() is not None:
                    print("❌ ERROR: Server process terminated unexpectedly during startup.")
                    self._log_monitor()
                    return False
                time.sleep(wait_interval)

            print(f"❌ ERROR: Server process failed to start listening within {max_wait} seconds.")
            self.stop_server()
            return False

        except Exception as e:
            print(f"❌ ERROR: Failed to start embedding server process: {e}")
            return False

    def _log_monitor(self):
        """Monitors and prints the server's stdout and stderr."""
        if not self.server_process:
            return
        try:
            if self.server_process.stdout:
                for line in iter(self.server_process.stdout.readline, ''):
                    print(f"[{self.backend_module_name} LOG]: {line.strip()}")
                self.server_process.stdout.close()
            if self.server_process.stderr:
                for line in iter(self.server_process.stderr.readline, ''):
                    print(f"[{self.backend_module_name} ERROR]: {line.strip()}")
                self.server_process.stderr.close()
        except Exception as e:
            print(f"Log monitor error: {e}")

    def stop_server(self):
        """Stops the embedding server process if it's running."""
        if self.server_process and self.server_process.poll() is None:
            print(f"INFO: Terminating session server process (PID: {self.server_process.pid})...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("INFO: Server process terminated.")
            except subprocess.TimeoutExpired:
                print("WARNING: Server process did not terminate gracefully, killing it.")
                self.server_process.kill()
        self.server_process = None
