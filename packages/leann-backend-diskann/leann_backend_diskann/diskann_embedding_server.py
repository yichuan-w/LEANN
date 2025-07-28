"""
DiskANN-specific embedding server
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

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


def create_diskann_embedding_server(
    passages_file: str | None = None,
    zmq_port: int = 5555,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    embedding_mode: str = "sentence-transformers",
    distance_metric: str = "l2",
):
    """
    Create and start a ZMQ-based embedding server for DiskANN backend.
    Uses ROUTER socket and protobuf communication as required by DiskANN C++ implementation.
    """
    logger.info(f"Starting DiskANN server on port {zmq_port} with model {model_name}")
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

    passages = PassageManager(meta["passage_sources"])
    logger.info(
        f"Loaded PassageManager with {len(passages.global_offset_map)} passages from metadata"
    )

    # Import protobuf after ensuring the path is correct
    try:
        from . import embedding_pb2
    except ImportError as e:
        logger.error(f"Failed to import protobuf module: {e}")
        return

    def zmq_server_thread():
        """ZMQ server thread using REP socket for universal compatibility"""
        context = zmq.Context()
        socket = context.socket(
            zmq.REP
        )  # REP socket for both BaseSearcher and DiskANN C++ REQ clients
        socket.bind(f"tcp://*:{zmq_port}")
        logger.info(f"DiskANN ZMQ REP server listening on port {zmq_port}")

        socket.setsockopt(zmq.RCVTIMEO, 300000)
        socket.setsockopt(zmq.SNDTIMEO, 300000)

        while True:
            try:
                # REP socket receives single-part messages
                message = socket.recv()

                # Check for empty messages - REP socket requires response to every request
                if len(message) == 0:
                    logger.debug("Received empty message, sending empty response")
                    socket.send(b"")  # REP socket must respond to every request
                    continue

                logger.debug(f"Received ZMQ request of size {len(message)} bytes")
                logger.debug(f"Message preview: {message[:50]}")  # Show first 50 bytes

                e2e_start = time.time()

                # Try protobuf first (for DiskANN C++ node_ids requests - primary use case)
                texts = []
                node_ids = []
                is_text_request = False

                try:
                    req_proto = embedding_pb2.NodeEmbeddingRequest()
                    req_proto.ParseFromString(message)
                    node_ids = list(req_proto.node_ids)

                    if not node_ids:
                        raise RuntimeError(
                            f"PROTOBUF: Received empty node_ids! Message size: {len(message)}"
                        )

                    logger.info(
                        f"✅ PROTOBUF: Node ID request for {len(node_ids)} node embeddings: {node_ids[:10]}"
                    )
                except Exception as protobuf_error:
                    logger.debug(f"Protobuf parsing failed: {protobuf_error}")
                    # Fallback to msgpack (for BaseSearcher direct text requests)
                    try:
                        import msgpack

                        request = msgpack.unpackb(message)
                        # For BaseSearcher compatibility, request is a list of texts directly
                        if isinstance(request, list) and all(
                            isinstance(item, str) for item in request
                        ):
                            texts = request
                            is_text_request = True
                            logger.info(f"✅ MSGPACK: Direct text request for {len(texts)} texts")
                        else:
                            raise ValueError("Not a valid msgpack text request")
                    except Exception as msgpack_error:
                        raise RuntimeError(
                            f"Both protobuf and msgpack parsing failed! Protobuf: {protobuf_error}, Msgpack: {msgpack_error}"
                        )

                # Look up texts by node IDs (only if not direct text request)
                if not is_text_request:
                    for nid in node_ids:
                        try:
                            passage_data = passages.get_passage(str(nid))
                            txt = passage_data["text"]
                            if not txt:
                                raise RuntimeError(f"FATAL: Empty text for passage ID {nid}")
                            texts.append(txt)
                        except KeyError as e:
                            logger.error(f"Passage ID {nid} not found: {e}")
                            raise e
                        except Exception as e:
                            logger.error(f"Exception looking up passage ID {nid}: {e}")
                            raise

                    # Debug logging
                    logger.debug(f"Processing {len(texts)} texts")
                    logger.debug(f"Text lengths: {[len(t) for t in texts[:5]]}")  # Show first 5

                # Process embeddings using unified computation
                embeddings = compute_embeddings(texts, model_name, mode=embedding_mode)
                logger.info(
                    f"Computed embeddings for {len(texts)} texts, shape: {embeddings.shape}"
                )

                # Prepare response based on request type
                if is_text_request:
                    # For BaseSearcher compatibility: return msgpack format
                    import msgpack

                    response_data = msgpack.packb(embeddings.tolist())
                else:
                    # For DiskANN C++ compatibility: return protobuf format
                    resp_proto = embedding_pb2.NodeEmbeddingResponse()
                    hidden_contiguous = np.ascontiguousarray(embeddings, dtype=np.float32)

                    # Serialize embeddings data
                    resp_proto.embeddings_data = hidden_contiguous.tobytes()
                    resp_proto.dimensions.append(hidden_contiguous.shape[0])
                    resp_proto.dimensions.append(hidden_contiguous.shape[1])

                    response_data = resp_proto.SerializeToString()

                # Send response back to the client
                socket.send(response_data)

                e2e_end = time.time()
                logger.info(f"⏱️  ZMQ E2E time: {e2e_end - e2e_start:.6f}s")

            except zmq.Again:
                logger.debug("ZMQ socket timeout, continuing to listen")
                continue
            except Exception as e:
                logger.error(f"Error in ZMQ server loop: {e}")
                import traceback

                traceback.print_exc()
                raise

    zmq_thread = threading.Thread(target=zmq_server_thread, daemon=True)
    zmq_thread.start()
    logger.info(f"Started DiskANN ZMQ server thread on port {zmq_port}")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("DiskANN Server shutting down...")
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

    parser = argparse.ArgumentParser(description="DiskANN Embedding service")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ port to run on")
    parser.add_argument(
        "--passages-file",
        type=str,
        help="Metadata JSON file containing passage sources",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "mlx"],
        help="Embedding backend mode",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="l2",
        choices=["l2", "mips", "cosine"],
        help="Distance metric for similarity computation",
    )

    args = parser.parse_args()

    # Create and start the DiskANN embedding server
    create_diskann_embedding_server(
        passages_file=args.passages_file,
        zmq_port=args.zmq_port,
        model_name=args.model_name,
        embedding_mode=args.embedding_mode,
        distance_metric=args.distance_metric,
    )
