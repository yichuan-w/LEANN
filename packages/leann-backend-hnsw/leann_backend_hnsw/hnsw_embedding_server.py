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
    # Use index dimensions from metadata for shaping fallback responses
    embedding_dim: int = int(meta.get("dimensions", 0))
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

        # Track last request type for safe fallback responses on exceptions
        last_request_type = "unknown"  # one of: 'text', 'distance', 'embedding', 'unknown'
        last_request_length = 0
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
                        last_request_type = "text"
                        last_request_length = len(request_payload)
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
                    last_request_type = "distance"
                    last_request_length = len(node_ids)

                    logger.debug("Distance calculation request received")
                    logger.debug(f"    Node IDs: {node_ids}")
                    logger.debug(f"    Query vector dim: {len(query_vector)}")

                    # Get embeddings for node IDs, tolerate missing IDs
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

                    # Prepare full-length response distances with safe fallbacks
                    large_distance = 1e9
                    response_distances = [large_distance] * len(node_ids)

                    if texts:
                        try:
                            # Process embeddings only for found indices
                            embeddings = compute_embeddings(texts, model_name, mode=embedding_mode)
                            logger.info(
                                f"Computed embeddings for {len(texts)} texts, shape: {embeddings.shape}"
                            )

                            # Calculate distances for found embeddings only
                            if distance_metric == "l2":
                                partial_distances = np.sum(
                                    np.square(embeddings - query_vector.reshape(1, -1)), axis=1
                                )
                            else:  # mips or cosine
                                partial_distances = -np.dot(embeddings, query_vector)

                            # Place computed distances back into the full response array
                            for pos, dval in zip(
                                found_indices, partial_distances.flatten().tolist()
                            ):
                                response_distances[pos] = float(dval)
                        except Exception as e:
                            logger.error(
                                f"Distance computation error, falling back to large distances: {e}"
                            )

                    # Always reply with exactly len(node_ids) distances
                    response_bytes = msgpack.packb([response_distances], use_single_float=True)
                    logger.debug(
                        f"Sending distance response with {len(response_distances)} distances (found={len(found_indices)})"
                    )

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
                last_request_type = "embedding"
                last_request_length = len(node_ids)

                # Allocate output buffer (B, D) and fill with zeros for robustness
                if embedding_dim <= 0:
                    logger.error("Embedding dimension unknown; cannot serve embedding request")
                    dims = [0, 0]
                    data = []
                else:
                    dims = [len(node_ids), embedding_dim]
                    data = [0.0] * (dims[0] * dims[1])

                # Look up texts by node IDs; compute embeddings where available
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
                        # Process embeddings for found texts only
                        embeddings = compute_embeddings(texts, model_name, mode=embedding_mode)
                        logger.info(
                            f"Computed embeddings for {len(texts)} texts, shape: {embeddings.shape}"
                        )

                        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                            logger.error(
                                f"NaN or Inf detected in embeddings! Requested IDs: {node_ids[:5]}..."
                            )
                            dims = [0, embedding_dim]
                            data = []
                        else:
                            # Copy computed embeddings into the correct positions
                            emb_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
                            flat = emb_f32.flatten().tolist()
                            for j, pos in enumerate(found_indices):
                                start = pos * embedding_dim
                                end = start + embedding_dim
                                data[start:end] = flat[j * embedding_dim : (j + 1) * embedding_dim]
                    except Exception as e:
                        logger.error(f"Embedding computation error, returning zeros: {e}")

                response_payload = [dims, data]
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
                # Fallback to a safe, minimal-structure response to avoid client crashes
                if last_request_type == "distance":
                    # Return a vector of large distances with the expected length
                    fallback_len = max(0, int(last_request_length))
                    large_distance = 1e9
                    safe_response = [[large_distance] * fallback_len]
                elif last_request_type == "embedding":
                    # Return an empty embedding block with known dimension if available
                    if embedding_dim > 0:
                        safe_response = [[0, embedding_dim], []]
                    else:
                        safe_response = [[0, 0], []]
                else:
                    # Unknown request type: default to empty embedding structure
                    safe_response = [[0, int(embedding_dim) if embedding_dim > 0 else 0], []]
                socket.send(msgpack.packb(safe_response, use_single_float=True))

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
