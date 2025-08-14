"""
HNSW-specific embedding server
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import msgpack
import numpy as np
import zmq

# Set up logging based on environment variable
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
logger = logging.getLogger(__name__)

# Force set logger level (don't rely on basicConfig in subprocess)
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logger.setLevel(log_level)

# Ensure we have a handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def create_hnsw_embedding_server(
    passages_file: Optional[str] = None,
    zmq_port: int = 5555,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    distance_metric: str = "mips",
    embedding_mode: str = "sentence-transformers",
):
    """
    Create and start a ZMQ-based embedding server for HNSW backend.
    Simplified version using unified embedding computation module.
    """
    logger.info(f"Starting HNSW server on port {zmq_port} with model {model_name}")
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

    def check_port(port):
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

    # Let PassageManager handle path resolution uniformly. It supports fallback order:
    # 1) path/index_path; 2) *_relative; 3) standard siblings next to meta
    passages = PassageManager(meta["passage_sources"], metadata_file_path=passages_file)
    # Dimension from metadata for shaping responses
    try:
        embedding_dim: int = int(meta.get("dimensions", 0))
    except Exception:
        embedding_dim = 0
    logger.info(
        f"Loaded PassageManager with {len(passages.global_offset_map)} passages from metadata"
    )

    # (legacy ZMQ thread removed; using shutdown-capable server only)

    def zmq_server_thread_with_shutdown(shutdown_event):
        """ZMQ server thread that respects shutdown signal.

        Creates its own REP socket bound to zmq_port and polls with timeouts
        to allow graceful shutdown.
        """
        logger.info("ZMQ server thread started with shutdown support")

        context = zmq.Context()
        rep_socket = context.socket(zmq.REP)
        rep_socket.bind(f"tcp://*:{zmq_port}")
        logger.info(f"HNSW ZMQ REP server listening on port {zmq_port}")
        rep_socket.setsockopt(zmq.RCVTIMEO, 1000)
        # Keep sends from blocking during shutdown; fail fast and drop on close
        rep_socket.setsockopt(zmq.SNDTIMEO, 1000)
        rep_socket.setsockopt(zmq.LINGER, 0)

        # Track last request type/length for shape-correct fallbacks
        last_request_type = "unknown"  # 'text' | 'distance' | 'embedding' | 'unknown'
        last_request_length = 0

        try:
            while not shutdown_event.is_set():
                try:
                    e2e_start = time.time()
                    logger.debug("🔍 Waiting for ZMQ message...")
                    request_bytes = rep_socket.recv()

                    # Rest of the processing logic (same as original)
                    request = msgpack.unpackb(request_bytes)

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
                        last_request_type = "text"
                        last_request_length = len(request)
                        embeddings = compute_embeddings(request, model_name, mode=embedding_mode)
                        rep_socket.send(msgpack.packb(embeddings.tolist()))
                        e2e_end = time.time()
                        logger.info(f"⏱️  Text embedding E2E time: {e2e_end - e2e_start:.6f}s")
                        continue

                    # Handle distance calculation request: [[ids], [query_vector]]
                    if (
                        isinstance(request, list)
                        and len(request) == 2
                        and isinstance(request[0], list)
                        and isinstance(request[1], list)
                    ):
                        node_ids = request[0]
                        # Handle nested [[ids]] shape defensively
                        if len(node_ids) == 1 and isinstance(node_ids[0], list):
                            node_ids = node_ids[0]
                        query_vector = np.array(request[1], dtype=np.float32)
                        last_request_type = "distance"
                        last_request_length = len(node_ids)

                        logger.debug("Distance calculation request received")
                        logger.debug(f"    Node IDs: {node_ids}")
                        logger.debug(f"    Query vector dim: {len(query_vector)}")

                        # Gather texts for found ids
                        texts: list[str] = []
                        found_indices: list[int] = []
                        for idx, nid in enumerate(node_ids):
                            try:
                                passage_data = passages.get_passage(str(nid))
                                txt = passage_data.get("text", "")
                                if isinstance(txt, str) and len(txt) > 0:
                                    texts.append(txt)
                                    found_indices.append(idx)
                                else:
                                    logger.error(f"Empty text for passage ID {nid}")
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
                                    texts, model_name, mode=embedding_mode
                                )
                                logger.info(
                                    f"Computed embeddings for {len(texts)} texts, shape: {embeddings.shape}"
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
                                logger.error(f"Distance computation error, using sentinels: {e}")

                        # Send response in expected shape [[distances]]
                        rep_socket.send(msgpack.packb([response_distances], use_single_float=True))
                        e2e_end = time.time()
                        logger.info(f"⏱️  Distance calculation E2E time: {e2e_end - e2e_start:.6f}s")
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
                    last_request_type = "embedding"
                    last_request_length = len(node_ids)
                    logger.info(f"ZMQ received {len(node_ids)} node IDs for embedding fetch")

                    # Preallocate zero-filled flat data for robustness
                    if embedding_dim <= 0:
                        dims = [0, 0]
                        flat_data: list[float] = []
                    else:
                        dims = [len(node_ids), embedding_dim]
                        flat_data = [0.0] * (dims[0] * dims[1])

                    # Collect texts for found ids
                    texts: list[str] = []
                    found_indices: list[int] = []
                    for idx, nid in enumerate(node_ids):
                        try:
                            passage_data = passages.get_passage(str(nid))
                            txt = passage_data.get("text", "")
                            if isinstance(txt, str) and len(txt) > 0:
                                texts.append(txt)
                                found_indices.append(idx)
                            else:
                                logger.error(f"Empty text for passage ID {nid}")
                        except KeyError:
                            logger.error(f"Passage with ID {nid} not found")
                        except Exception as e:
                            logger.error(f"Exception looking up passage ID {nid}: {e}")

                    if texts:
                        try:
                            embeddings = compute_embeddings(texts, model_name, mode=embedding_mode)
                            logger.info(
                                f"Computed embeddings for {len(texts)} texts, shape: {embeddings.shape}"
                            )

                            if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                                logger.error(
                                    f"NaN or Inf detected in embeddings! Requested IDs: {node_ids[:5]}..."
                                )
                                dims = [0, embedding_dim]
                                flat_data = []
                            else:
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
                            logger.error(f"Embedding computation error, returning zeros: {e}")

                    response_payload = [dims, flat_data]
                    response_bytes = msgpack.packb(response_payload, use_single_float=True)

                    rep_socket.send(response_bytes)
                    e2e_end = time.time()
                    logger.info(f"⏱️  ZMQ E2E time: {e2e_end - e2e_start:.6f}s")

                except zmq.Again:
                    # Timeout - check shutdown_event and continue
                    continue
                except Exception as e:
                    if not shutdown_event.is_set():
                        logger.error(f"Error in ZMQ server loop: {e}")
                        # Shape-correct fallback
                        try:
                            if last_request_type == "distance":
                                large_distance = 1e9
                                fallback_len = max(0, int(last_request_length))
                                safe = [[large_distance] * fallback_len]
                            elif last_request_type == "embedding":
                                bsz = max(0, int(last_request_length))
                                dim = max(0, int(embedding_dim))
                                safe = (
                                    [[bsz, dim], [0.0] * (bsz * dim)] if dim > 0 else [[0, 0], []]
                                )
                            elif last_request_type == "text":
                                safe = []  # direct text embeddings expectation is a flat list
                            else:
                                safe = [[0, int(embedding_dim) if embedding_dim > 0 else 0], []]
                            rep_socket.send(msgpack.packb(safe, use_single_float=True))
                        except Exception:
                            pass
                    else:
                        logger.info("Shutdown in progress, ignoring ZMQ error")
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

    # Add shutdown coordination
    shutdown_event = threading.Event()

    def shutdown_zmq_server():
        """Gracefully shutdown ZMQ server."""
        logger.info("Initiating graceful shutdown...")
        shutdown_event.set()

        if zmq_thread.is_alive():
            logger.info("Waiting for ZMQ thread to finish...")
            zmq_thread.join(timeout=5)
            if zmq_thread.is_alive():
                logger.warning("ZMQ thread did not finish in time")

        # Clean up ZMQ resources
        try:
            # Note: socket and context are cleaned up by thread exit
            logger.info("ZMQ resources cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning ZMQ resources: {e}")

        # Clean up other resources
        try:
            import gc

            gc.collect()
            logger.info("Additional resources cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning additional resources: {e}")

        logger.info("Graceful shutdown completed")
        sys.exit(0)

    # Register signal handlers within this function scope
    import signal

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        shutdown_zmq_server()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Pass shutdown_event to ZMQ thread
    zmq_thread = threading.Thread(
        target=lambda: zmq_server_thread_with_shutdown(shutdown_event),
        daemon=False,  # Not daemon - we want to wait for it
    )
    zmq_thread.start()
    logger.info(f"Started HNSW ZMQ server thread on port {zmq_port}")

    # Keep the main thread alive
    try:
        while not shutdown_event.is_set():
            time.sleep(0.1)  # Check shutdown more frequently
    except KeyboardInterrupt:
        logger.info("HNSW Server shutting down...")
        shutdown_zmq_server()
        return

    # If we reach here, shutdown was triggered by signal
    logger.info("Main loop exited, process should be shutting down")


if __name__ == "__main__":
    import sys

    # Signal handlers are now registered within create_hnsw_embedding_server

    parser = argparse.ArgumentParser(description="HNSW Embedding service")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ port to run on")
    parser.add_argument(
        "--passages-file",
        type=str,
        help="JSON file containing passage ID to text mapping",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--distance-metric", type=str, default="mips", help="Distance metric to use"
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "mlx", "ollama"],
        help="Embedding backend mode",
    )

    args = parser.parse_args()

    # Create and start the HNSW embedding server
    create_hnsw_embedding_server(
        passages_file=args.passages_file,
        zmq_port=args.zmq_port,
        model_name=args.model_name,
        distance_metric=args.distance_metric,
        embedding_mode=args.embedding_mode,
    )
