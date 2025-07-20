import threading
import time
import atexit
import socket
import subprocess
import sys
import zmq
import msgpack
from pathlib import Path
from typing import Optional
import select


def _check_port(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _check_server_meta_path(port: int, expected_meta_path: str) -> bool:
    """
    Check if the existing server on the port is using the correct meta file.
    Returns True if the server has the right meta path, False otherwise.
    """
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 second timeout
        socket.connect(f"tcp://localhost:{port}")

        # Send a special control message to query the server's meta path
        control_request = ["__QUERY_META_PATH__"]
        request_bytes = msgpack.packb(control_request)
        socket.send(request_bytes)

        # Wait for response
        response_bytes = socket.recv()
        response = msgpack.unpackb(response_bytes)

        socket.close()
        context.term()

        # Check if the response contains the meta path and if it matches
        if isinstance(response, list) and len(response) > 0:
            server_meta_path = response[0]
            # Normalize paths for comparison
            expected_path = Path(expected_meta_path).resolve()
            server_path = Path(server_meta_path).resolve() if server_meta_path else None
            return server_path == expected_path

        return False

    except Exception as e:
        print(f"WARNING: Could not query server meta path on port {port}: {e}")
        return False


def _update_server_meta_path(port: int, new_meta_path: str) -> bool:
    """
    Send a control message to update the server's meta path.
    Returns True if successful, False otherwise.
    """
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        socket.connect(f"tcp://localhost:{port}")

        # Send a control message to update the meta path
        control_request = ["__UPDATE_META_PATH__", new_meta_path]
        request_bytes = msgpack.packb(control_request)
        socket.send(request_bytes)

        # Wait for response
        response_bytes = socket.recv()
        response = msgpack.unpackb(response_bytes)

        socket.close()
        context.term()

        # Check if the update was successful
        if isinstance(response, list) and len(response) > 0:
            return response[0] == "SUCCESS"

        return False

    except Exception as e:
        print(f"ERROR: Could not update server meta path on port {port}: {e}")
        return False


def _check_server_model(port: int, expected_model: str) -> bool:
    """
    Check if the existing server on the port is using the correct embedding model.
    Returns True if the server has the right model, False otherwise.
    """
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 second timeout
        socket.connect(f"tcp://localhost:{port}")

        # Send a special control message to query the server's model
        control_request = ["__QUERY_MODEL__"]
        request_bytes = msgpack.packb(control_request)
        socket.send(request_bytes)

        # Wait for response
        response_bytes = socket.recv()
        response = msgpack.unpackb(response_bytes)

        socket.close()
        context.term()

        # Check if the response contains the model name and if it matches
        if isinstance(response, list) and len(response) > 0:
            server_model = response[0]
            return server_model == expected_model

        return False

    except Exception as e:
        print(f"WARNING: Could not query server model on port {port}: {e}")
        return False


def _update_server_model(port: int, new_model: str) -> bool:
    """
    Send a control message to update the server's embedding model.
    Returns True if successful, False otherwise.
    """
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout for model loading
        socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout for sending
        socket.connect(f"tcp://localhost:{port}")

        # Send a control message to update the model
        control_request = ["__UPDATE_MODEL__", new_model]
        request_bytes = msgpack.packb(control_request)
        socket.send(request_bytes)

        # Wait for response
        response_bytes = socket.recv()
        response = msgpack.unpackb(response_bytes)

        socket.close()
        context.term()

        # Check if the update was successful
        if isinstance(response, list) and len(response) > 0:
            return response[0] == "SUCCESS"

        return False

    except Exception as e:
        print(f"ERROR: Could not update server model on port {port}: {e}")
        return False


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

    def start_server(self, port: int, model_name: str, embedding_mode: str = "sentence-transformers", **kwargs) -> bool:
        """
        Starts the embedding server process.

        Args:
            port (int): The ZMQ port for the server.
            model_name (str): The name of the embedding model to use.
            **kwargs: Additional arguments for the server (e.g., passages_file, distance_metric, enable_warmup).

        Returns:
            bool: True if the server is started successfully or already running, False otherwise.
        """
        if self.server_process and self.server_process.poll() is None:
            # Even if we have a running process, check if model/meta path match
            if self.server_port is not None:
                port_in_use = _check_port(self.server_port)
                if port_in_use:
                    print(
                        f"INFO: Checking compatibility of existing server process (PID {self.server_process.pid})"
                    )

                    # Check model compatibility
                    model_matches = _check_server_model(self.server_port, model_name)
                    if model_matches:
                        print(
                            f"✅ Existing server already using correct model: {model_name}"
                        )
                        
                        # Still check meta path if provided
                        passages_file = kwargs.get("passages_file")
                        if passages_file and str(passages_file).endswith(
                            ".meta.json"
                        ):
                            meta_matches = _check_server_meta_path(
                                self.server_port, str(passages_file)
                            )
                            if not meta_matches:
                                print("⚠️  Updating meta path to: {passages_file}")
                                _update_server_meta_path(
                                    self.server_port, str(passages_file)
                                )
                        
                        return True
                    else:
                        print(
                            f"⚠️  Existing server has different model. Attempting to update to: {model_name}"
                        )
                        if not _update_server_model(self.server_port, model_name):
                            print(
                                "❌ Failed to update existing server model. Restarting server..."
                            )
                            self.stop_server()
                            # Continue to start new server below
                        else:
                            print(
                                f"✅ Successfully updated existing server model to: {model_name}"
                            )

                            # Also check meta path if provided
                            passages_file = kwargs.get("passages_file")
                            if passages_file and str(passages_file).endswith(
                                ".meta.json"
                            ):
                                meta_matches = _check_server_meta_path(
                                    self.server_port, str(passages_file)
                                )
                                if not meta_matches:
                                    print("⚠️  Updating meta path to: {passages_file}")
                                    _update_server_meta_path(
                                        self.server_port, str(passages_file)
                                    )

                            return True
                else:
                    # Server process exists but port not responding - restart
                    print("⚠️  Server process exists but not responding. Restarting...")
                    self.stop_server()
                    # Continue to start new server below
            else:
                # No port stored - restart
                print("⚠️  No port information stored. Restarting server...")
                self.stop_server()
                # Continue to start new server below

        if _check_port(port):
            # Port is in use, check if it's using the correct meta file and model
            passages_file = kwargs.get("passages_file")

            print(f"INFO: Port {port} is in use. Checking server compatibility...")

            # Check model compatibility first
            model_matches = _check_server_model(port, model_name)
            if model_matches:
                print(
                    f"✅ Existing server on port {port} is using correct model: {model_name}"
                )
            else:
                print(
                    f"⚠️  Existing server on port {port} has different model. Attempting to update to: {model_name}"
                )
                if not _update_server_model(port, model_name):
                    raise RuntimeError(
                        f"❌ Failed to update server model to {model_name}. Consider using a different port."
                    )
                print(f"✅ Successfully updated server model to: {model_name}")

            # Check meta path compatibility if provided
            if passages_file and str(passages_file).endswith(".meta.json"):
                meta_matches = _check_server_meta_path(port, str(passages_file))
                if not meta_matches:
                    print(
                        f"⚠️  Existing server on port {port} has different meta path. Attempting to update..."
                    )
                    if not _update_server_meta_path(port, str(passages_file)):
                        raise RuntimeError(
                            "❌ Failed to update server meta path. This may cause data synchronization issues."
                        )
                    print(
                        f"✅ Successfully updated server meta path to: {passages_file}"
                    )
                else:
                    print(
                        f"✅ Existing server on port {port} is using correct meta path: {passages_file}"
                    )

            print(f"✅ Server on port {port} is compatible and ready to use.")
            return True

        print(
            f"INFO: Starting session-level embedding server for '{self.backend_module_name}'..."
        )

        try:
            command = [
                sys.executable,
                "-m",
                self.backend_module_name,
                "--zmq-port",
                str(port),
                "--model-name",
                model_name,
            ]

            # Add extra arguments for specific backends
            if "passages_file" in kwargs and kwargs["passages_file"]:
                command.extend(["--passages-file", str(kwargs["passages_file"])])
            # if "distance_metric" in kwargs and kwargs["distance_metric"]:
            #     command.extend(["--distance-metric", kwargs["distance_metric"]])
            if embedding_mode != "sentence-transformers":
                command.extend(["--embedding-mode", embedding_mode])
            if "enable_warmup" in kwargs and not kwargs["enable_warmup"]:
                command.extend(["--disable-warmup"])

            project_root = Path(__file__).parent.parent.parent.parent.parent
            print(f"INFO: Running command from project root: {project_root}")
            print(f"INFO: Command: {' '.join(command)}")  # Debug: show actual command

            self.server_process = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout for easier monitoring
                text=True,
                encoding="utf-8",
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )
            self.server_port = port
            print(f"INFO: Server process started with PID: {self.server_process.pid}")

            max_wait, wait_interval = 120, 0.5
            for _ in range(int(max_wait / wait_interval)):
                if _check_port(port):
                    print("✅ Embedding server is up and ready for this session.")
                    log_thread = threading.Thread(target=self._log_monitor, daemon=True)
                    log_thread.start()
                    return True
                if self.server_process.poll() is not None:
                    print(
                        "❌ ERROR: Server process terminated unexpectedly during startup."
                    )
                    self._print_recent_output()
                    return False
                time.sleep(wait_interval)

            print(
                f"❌ ERROR: Server process failed to start listening within {max_wait} seconds."
            )
            self.stop_server()
            return False

        except Exception as e:
            print(f"❌ ERROR: Failed to start embedding server process: {e}")
            return False

    def _print_recent_output(self):
        """Print any recent output from the server process."""
        if not self.server_process or not self.server_process.stdout:
            return
        try:
            # Read any available output

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
        if self.server_process and self.server_process.poll() is None:
            print(
                f"INFO: Terminating session server process (PID: {self.server_process.pid})..."
            )
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("INFO: Server process terminated.")
            except subprocess.TimeoutExpired:
                print(
                    "WARNING: Server process did not terminate gracefully, killing it."
                )
                self.server_process.kill()
        self.server_process = None
