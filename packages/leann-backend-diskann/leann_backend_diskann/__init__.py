print("Initializing leann-backend-diskann...")

try:
    from .diskann_backend import DiskannBackend
    print("INFO: DiskANN backend loaded successfully")
except ImportError as e:
    print(f"WARNING: Could not import DiskANN backend: {e}")