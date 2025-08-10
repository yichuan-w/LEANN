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
from typing import Union

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
    passages_file: Union[str, None] = None,
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

    # Convert relative paths to absolute paths based on metadata file location
    metadata_dir = Path(passages_file).parent.parent  # Go up one level from the metadata file
    passage_sources = []
    for source in meta["passage_sources"]:
        source_copy = source.copy()
        # Convert relative paths to absolute paths
        if not Path(source_copy["path"]).is_absolute():
            source_copy["path"] = str(metadata_dir / source_copy["path"])
        if not Path(source_copy["index_path"]).is_absolute():
            source_copy["index_path"] = str(metadata_dir / source_copy["index_path"])
        passage_sources.append(source_copy)

    passages = PassageManager(passage_sources)
    logger.info(
        f"Loaded PassageManager with {len(passages.global_offset_map)} passages from metadata"
    )

    def zmq_server_thread():
        """ZMQ server thread"""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{zmq_port}")
        logger.info(f"HNSW ZMQ server listening on port {zmq_port}")

        socket.setsockopt(zmq.RCVTIMEO, 300000)
        socket.setsockopt(zmq.SNDTIMEO, 300000)

        while True:
            try:
                message_bytes = socket.recv()
                logger.debug(f"Received ZMQ request of size {len(message_bytes)} bytes")

                e2e_start = time.time()
                request_payload = msgpack.unpackb(message_bytes)

                # Handle direct text embedding request
                if isinstance(request_payload, list) and len(request_payload) > 0:
                    # Check if this is a direct text request (list of strings)
                    if all(isinstance(item, str) for item in request_payload):
                        logger.info(
                            f"Processing direct text embedding request for {len(request_payload)} texts in {embedding_mode} mode"
                        )

                        # Use unified embedding computation (now with model caching)
                        embeddings = compute_embeddings(
                            request_payload, model_name, mode=embedding_mode
                        )

                        response = embeddings.tolist()
                        socket.send(msgpack.packb(response))
                        e2e_end = time.time()
                        logger.info(f"⏱️  Text embedding E2E time: {e2e_end - e2e_start:.6f}s")
                        continue

                # Handle distance calculation requests
                if (
                    isinstance(request_payload, list)
                    and len(request_payload) == 2
                    and isinstance(request_payload[0], list)
                    and isinstance(request_payload[1], list)
                ):
                    node_ids = request_payload[0]
                    query_vector = np.array(request_payload[1], dtype=np.float32)

                    logger.debug("Distance calculation request received")
                    logger.debug(f"    Node IDs: {node_ids}")
                    logger.debug(f"    Query vector dim: {len(query_vector)}")

                    # Get embeddings for node IDs
                    texts = []
                    for nid in node_ids:
                        try:
                            passage_data = passages.get_passage(str(nid))
                            txt = passage_data["text"]
                            texts.append(txt)
                        except KeyError:
                            logger.error(f"Passage ID {nid} not found")
                            raise RuntimeError(f"FATAL: Passage with ID {nid} not found")
                        except Exception as e:
                            logger.error(f"Exception looking up passage ID {nid}: {e}")
                            raise

                    # Process embeddings
                    embeddings = compute_embeddings(texts, model_name, mode=embedding_mode)
                    logger.info(
                        f"Computed embeddings for {len(texts)} texts, shape: {embeddings.shape}"
                    )

                    # Calculate distances
                    if distance_metric == "l2":
                        distances = np.sum(
                            np.square(embeddings - query_vector.reshape(1, -1)), axis=1
                        )
                    else:  # mips or cosine
                        distances = -np.dot(embeddings, query_vector)

                    response_payload = distances.flatten().tolist()
                    response_bytes = msgpack.packb([response_payload], use_single_float=True)
                    logger.debug(f"Sending distance response with {len(distances)} distances")

                    socket.send(response_bytes)
                    e2e_end = time.time()
                    logger.info(f"⏱️  Distance calculation E2E time: {e2e_end - e2e_start:.6f}s")
                    continue

                # Standard embedding request (passage ID lookup)
                if (
                    not isinstance(request_payload, list)
                    or len(request_payload) != 1
                    or not isinstance(request_payload[0], list)
                ):
                    logger.error(
                        f"Invalid MessagePack request format. Expected [[ids...]] or [texts...], got: {type(request_payload)}"
                    )
                    socket.send(msgpack.packb([[], []]))
                    continue

                node_ids = request_payload[0]
                logger.debug(f"Request for {len(node_ids)} node embeddings")

                # Look up texts by node IDs
                texts = []
                for nid in node_ids:
                    try:
                        passage_data = passages.get_passage(str(nid))
                        txt = passage_data["text"]
                        if not txt:
                            raise RuntimeError(f"FATAL: Empty text for passage ID {nid}")
                        texts.append(txt)
                    except KeyError:
                        raise RuntimeError(f"FATAL: Passage with ID {nid} not found")
                    except Exception as e:
                        logger.error(f"Exception looking up passage ID {nid}: {e}")
                        raise

                # Process embeddings
                embeddings = compute_embeddings(texts, model_name, mode=embedding_mode)
                logger.info(
                    f"Computed embeddings for {len(texts)} texts, shape: {embeddings.shape}"
                )

                # Serialization and response
                if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                    logger.error(
                        f"NaN or Inf detected in embeddings! Requested IDs: {node_ids[:5]}..."
                    )
                    raise AssertionError()

                hidden_contiguous_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
                response_payload = [
                    list(hidden_contiguous_f32.shape),
                    hidden_contiguous_f32.flatten().tolist(),
                ]
                response_bytes = msgpack.packb(response_payload, use_single_float=True)

                socket.send(response_bytes)
                e2e_end = time.time()
                logger.info(f"⏱️  ZMQ E2E time: {e2e_end - e2e_start:.6f}s")

            except zmq.Again:
                logger.debug("ZMQ socket timeout, continuing to listen")
                continue
            except Exception as e:
                logger.error(f"Error in ZMQ server loop: {e}")
                import traceback

                traceback.print_exc()
                socket.send(msgpack.packb([[], []]))

    zmq_thread = threading.Thread(target=zmq_server_thread, daemon=True)
    zmq_thread.start()
    logger.info(f"Started HNSW ZMQ server thread on port {zmq_port}")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("HNSW Server shutting down...")
        return


if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

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
