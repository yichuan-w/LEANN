"""
FAISS-specific embedding server.

"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import msgpack
import numpy as np
import zmq

# Set up logging based on environment variable
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
logger = logging.getLogger(__name__)

# Force set logger level (don't rely on basicConfig in subprocess)
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logger.setLevel(log_level)

# Ensure we have handlers if none exist
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

log_path = os.getenv("LEANN_FAISS_LOG_PATH")
if log_path:
    try:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [pid=%(process)d] %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as exc:
        logger.warning(f"Failed to attach file handler for log path {log_path}: {exc}")

logger.propagate = False

# Parse provider options from environment
_RAW_PROVIDER_OPTIONS = os.getenv("LEANN_EMBEDDING_OPTIONS")
try:
    PROVIDER_OPTIONS: dict[str, Any] = (
        json.loads(_RAW_PROVIDER_OPTIONS) if _RAW_PROVIDER_OPTIONS else {}
    )
except json.JSONDecodeError:
    logger.warning("Failed to parse LEANN_EMBEDDING_OPTIONS; ignoring provider options")
    PROVIDER_OPTIONS = {}


def create_faiss_embedding_server(
    passages_file: Optional[str] = None,
    zmq_port: int = 5557,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    distance_metric: str = "mips",
    embedding_mode: str = "sentence-transformers",
) -> None:
    """
    Create and start a ZMQ-based embedding server for FAISS backend.
    Simplified version using unified embedding computation module.
    """
    logger.info(f"Starting FAISS server on port {zmq_port} with model {model_name}")
    logger.info(f"Using embedding mode: {embedding_mode}")

    # Add leann-core to path for unified embedding computation
    current_dir = Path(__file__).parent
    leann_core_path = current_dir.parent.parent / "leann-core" / "src"
    sys.path.insert(0, str(leann_core_path))

    try:
        from leann.api import PassageManager
        from leann.embedding_compute import compute_embeddings

        logger.info("Successfully imported unified embedding computation module")
    except ImportError as e:
        logger.error(f"Failed to import embedding computation module: {e}")
        return
    finally:
        sys.path.pop(0)

    # Check port availability
    import socket

    def check_port(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    if check_port(zmq_port):
        logger.error(f"Port {zmq_port} is already in use")
        return

    # Only support metadata file, fail fast for everything else
    if not passages_file or not passages_file.endswith(".meta.json"):
        raise ValueError("Only metadata files (.meta.json) are supported")

    # Load metadata to get passage sources
    with open(passages_file) as f:
        meta = json.load(f)

    # Let PassageManager handle path resolution uniformly
    passages = PassageManager(meta["passage_sources"], metadata_file_path=passages_file)

    # Dimension from metadata for shaping responses
    try:
        embedding_dim: int = int(meta.get("dimensions", 0))
    except Exception:
        embedding_dim = 0
    logger.info(f"Loaded PassageManager with {len(passages)} passages from metadata")

    # Attempt to load ID map (maps FAISS integer labels -> passage IDs)
    id_map: list[str] = []
    try:
        meta_path = Path(passages_file)
        base = meta_path.name
        if base.endswith(".meta.json"):
            base = base[: -len(".meta.json")]
        if base.endswith(".leann"):
            base = base[: -len(".leann")]
        idmap_file = meta_path.parent / f"{base}.ids.txt"
        if idmap_file.exists():
            with open(idmap_file, encoding="utf-8") as f:
                id_map = [line.rstrip("\n") for line in f]
            logger.info(f"Loaded ID map with {len(id_map)} entries from {idmap_file}")
        else:
            logger.warning(f"ID map file not found at {idmap_file}; will use raw labels")
    except Exception as e:
        logger.warning(f"Failed to load ID map: {e}")

    def _map_node_id(nid: Any) -> str:
        try:
            if id_map and isinstance(nid, (int, np.integer)):
                idx = int(nid)
                if 0 <= idx < len(id_map):
                    return id_map[idx]
        except Exception:
            pass
        return str(nid)

    # Server state
    shutdown_event = threading.Event()

    def zmq_server_thread_with_shutdown(shutdown_evt: threading.Event) -> None:
        """ZMQ server thread that respects shutdown signal."""
        logger.info("ZMQ server thread started with shutdown support")

        context = zmq.Context()
        rep_socket = context.socket(zmq.REP)
        rep_socket.bind(f"tcp://*:{zmq_port}")
        logger.info(f"FAISS ZMQ REP server listening on port {zmq_port}")
        rep_socket.setsockopt(zmq.RCVTIMEO, 1000)
        rep_socket.setsockopt(zmq.SNDTIMEO, 1000)
        rep_socket.setsockopt(zmq.LINGER, 0)

        try:
            while not shutdown_evt.is_set():
                try:
                    e2e_start = time.time()
                    logger.debug("Waiting for ZMQ message...")
                    request_bytes = rep_socket.recv()

                    request = msgpack.unpackb(request_bytes)

                    # Handle model query
                    if len(request) == 1 and request[0] == "__QUERY_MODEL__":
                        response_bytes = msgpack.packb([model_name])
                        rep_socket.send(response_bytes)
                        continue

                    # Handle direct text embedding request
                    if (
                        isinstance(request, list)
                        and request
                        and all(isinstance(item, str) for item in request)
                    ):
                        embeddings = compute_embeddings(
                            request,
                            model_name,
                            mode=embedding_mode,
                            provider_options=PROVIDER_OPTIONS,
                        )
                        rep_socket.send(msgpack.packb(embeddings.tolist()))
                        e2e_end = time.time()
                        logger.info(f"Text embedding E2E time: {e2e_end - e2e_start:.6f}s")
                        continue

                    # Handle distance calculation request: [[ids], [query_vector]]
                    if (
                        isinstance(request, list)
                        and len(request) == 2
                        and isinstance(request[0], list)
                        and isinstance(request[1], list)
                    ):
                        node_ids = request[0]
                        if len(node_ids) == 1 and isinstance(node_ids[0], list):
                            node_ids = node_ids[0]
                        query_vector = np.array(request[1], dtype=np.float32)

                        logger.debug(f"Distance calculation for {len(node_ids)} nodes")

                        # Gather texts for found ids
                        texts: list[str] = []
                        found_indices: list[int] = []
                        for idx, nid in enumerate(node_ids):
                            try:
                                passage_id = _map_node_id(nid)
                                passage_data = passages.get_passage(passage_id)
                                txt = passage_data.get("text", "")
                                if isinstance(txt, str) and len(txt) > 0:
                                    texts.append(txt)
                                    found_indices.append(idx)
                            except KeyError:
                                logger.error(f"Passage ID {nid} not found")
                            except Exception as e:
                                logger.error(f"Exception looking up passage ID {nid}: {e}")

                        # Prepare full-length response with large sentinel values
                        large_distance = 1e9
                        response_distances = [large_distance] * len(node_ids)

                        if texts:
                            try:
                                embeddings = compute_embeddings(
                                    texts,
                                    model_name,
                                    mode=embedding_mode,
                                    provider_options=PROVIDER_OPTIONS,
                                )
                                if distance_metric == "l2":
                                    partial = np.sum(
                                        np.square(embeddings - query_vector.reshape(1, -1)), axis=1
                                    )
                                else:  # mips or cosine
                                    partial = -np.dot(embeddings, query_vector)

                                for pos, dval in zip(found_indices, partial.flatten().tolist()):
                                    response_distances[pos] = float(dval)
                            except Exception as e:
                                logger.error(f"Distance computation error: {e}")

                        rep_socket.send(msgpack.packb([response_distances], use_single_float=True))
                        e2e_end = time.time()
                        logger.info(f"Distance calculation E2E time: {e2e_end - e2e_start:.6f}s")
                        continue

                    # Fallback: treat as embedding-by-id request
                    if (
                        isinstance(request, list)
                        and len(request) == 1
                        and isinstance(request[0], list)
                    ):
                        node_ids = request[0]
                    elif isinstance(request, list):
                        node_ids = request
                    else:
                        node_ids = []

                    logger.info(f"ZMQ received {len(node_ids)} node IDs for embedding fetch")

                    # Preallocate zero-filled flat data
                    if embedding_dim <= 0:
                        dims = [0, 0]
                        flat_data: list[float] = []
                    else:
                        dims = [len(node_ids), embedding_dim]
                        flat_data = [0.0] * (dims[0] * dims[1])

                    # Collect texts for found ids
                    texts = []
                    found_indices = []
                    for idx, nid in enumerate(node_ids):
                        try:
                            passage_id = _map_node_id(nid)
                            passage_data = passages.get_passage(passage_id)
                            txt = passage_data.get("text", "")
                            if isinstance(txt, str) and len(txt) > 0:
                                texts.append(txt)
                                found_indices.append(idx)
                        except KeyError:
                            logger.error(f"Passage with ID {nid} not found")
                        except Exception as e:
                            logger.error(f"Exception looking up passage ID {nid}: {e}")

                    if texts:
                        try:
                            embeddings = compute_embeddings(
                                texts,
                                model_name,
                                mode=embedding_mode,
                                provider_options=PROVIDER_OPTIONS,
                            )
                            emb_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
                            flat = emb_f32.flatten().tolist()
                            for j, pos in enumerate(found_indices):
                                start = pos * embedding_dim
                                end = start + embedding_dim
                                if end <= len(flat_data):
                                    flat_data[start:end] = flat[
                                        j * embedding_dim : (j + 1) * embedding_dim
                                    ]
                        except Exception as e:
                            logger.error(f"Embedding computation error: {e}")

                    response_payload = [dims, flat_data]
                    response_bytes = msgpack.packb(response_payload, use_single_float=True)
                    rep_socket.send(response_bytes)
                    e2e_end = time.time()
                    logger.info(f"ZMQ E2E time: {e2e_end - e2e_start:.6f}s")

                except zmq.Again:
                    continue
                except Exception as e:
                    if not shutdown_evt.is_set():
                        logger.error(f"Error in ZMQ server loop: {e}")
                        try:
                            rep_socket.send(msgpack.packb([[0, 0], []], use_single_float=True))
                        except Exception:
                            pass
                    else:
                        break

        finally:
            try:
                rep_socket.close(0)
            except Exception:
                pass
            try:
                context.term()
            except Exception:
                pass

        logger.info("ZMQ server thread exiting gracefully")

    def shutdown_zmq_server() -> None:
        """Gracefully shutdown ZMQ server."""
        logger.info("Initiating graceful shutdown...")
        shutdown_event.set()

        if zmq_thread.is_alive():
            logger.info("Waiting for ZMQ thread to finish...")
            zmq_thread.join(timeout=5)

        logger.info("Graceful shutdown completed")
        sys.exit(0)

    def signal_handler(sig: int, frame: Any) -> None:
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        shutdown_zmq_server()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start ZMQ thread
    zmq_thread = threading.Thread(
        target=lambda: zmq_server_thread_with_shutdown(shutdown_event),
        daemon=False,
    )
    zmq_thread.start()
    logger.info(f"Started FAISS ZMQ server thread on port {zmq_port}")

    # Keep the main thread alive
    try:
        while not shutdown_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("FAISS Server shutting down...")
        shutdown_zmq_server()
        return

    logger.info("Main loop exited, process should be shutting down")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS Embedding service")
    parser.add_argument("--zmq-port", type=int, default=5557, help="ZMQ port to run on")
    parser.add_argument(
        "--passages-file",
        type=str,
        help="JSON file containing passage ID to text mapping",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1.5",
        help="Embedding model name",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="mips",
        help="Distance metric to use",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "mlx", "ollama"],
        help="Embedding backend mode",
    )

    args = parser.parse_args()

    create_faiss_embedding_server(
        passages_file=args.passages_file,
        zmq_port=args.zmq_port,
        model_name=args.model_name,
        distance_metric=args.distance_metric,
        embedding_mode=args.embedding_mode,
    )
