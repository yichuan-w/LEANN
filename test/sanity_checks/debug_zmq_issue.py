#!/usr/bin/env python3
"""
Debug script to test ZMQ communication with the exact same setup as main_cli_example.py
"""

import sys
import time

import zmq

sys.path.append("packages/leann-backend-diskann")
from leann_backend_diskann import embedding_pb2


def test_zmq_with_same_model():
    print("=== Testing ZMQ with same model as main_cli_example.py ===")

    # Test the exact same model that main_cli_example.py uses
    model_name = "sentence-transformers/all-mpnet-base-v2"

    # Start server with the same model
    import subprocess

    server_cmd = [
        sys.executable,
        "-m",
        "packages.leann-backend-diskann.leann_backend_diskann.embedding_server",
        "--zmq-port",
        "5556",  # Use different port to avoid conflicts
        "--model-name",
        model_name,
    ]

    print(f"Starting server with command: {' '.join(server_cmd)}")
    server_process = subprocess.Popen(
        server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(10)

    # Check if server is running
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print(f"Server failed to start. stdout: {stdout}")
        print(f"Server failed to start. stderr: {stderr}")
        return False

    print(f"Server started with PID: {server_process.pid}")

    try:
        # Test client
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://127.0.0.1:5556")
        socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout like C++
        socket.setsockopt(zmq.SNDTIMEO, 30000)

        # Create request with same format as C++
        request = embedding_pb2.NodeEmbeddingRequest()
        request.node_ids.extend([0, 1, 2, 3, 4])  # Test with some node IDs

        print(f"Sending request with {len(request.node_ids)} node IDs...")
        start_time = time.time()

        # Send request
        socket.send(request.SerializeToString())

        # Receive response
        response_data = socket.recv()
        end_time = time.time()

        print(f"Received response in {end_time - start_time:.3f} seconds")
        print(f"Response size: {len(response_data)} bytes")

        # Parse response
        response = embedding_pb2.NodeEmbeddingResponse()
        response.ParseFromString(response_data)

        print(f"Response dimensions: {list(response.dimensions)}")
        print(f"Embeddings data size: {len(response.embeddings_data)} bytes")
        print(f"Missing IDs: {list(response.missing_ids)}")

        # Calculate expected size
        if len(response.dimensions) == 2:
            batch_size = response.dimensions[0]
            embedding_dim = response.dimensions[1]
            expected_bytes = batch_size * embedding_dim * 4  # 4 bytes per float
            print(f"Expected bytes: {expected_bytes}, Actual: {len(response.embeddings_data)}")

            if len(response.embeddings_data) == expected_bytes:
                print("✅ Response format is correct!")
                return True
            else:
                print("❌ Response format mismatch!")
                return False
        else:
            print("❌ Invalid response dimensions!")
            return False

    except Exception as e:
        print(f"❌ Error during ZMQ test: {e}")
        return False
    finally:
        # Clean up
        server_process.terminate()
        server_process.wait()
        print("Server terminated")


if __name__ == "__main__":
    success = test_zmq_with_same_model()
    if success:
        print("\n✅ ZMQ communication test passed!")
    else:
        print("\n❌ ZMQ communication test failed!")
