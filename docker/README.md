# Docker Setup

This folder provides reference Docker images for LEANN.

## Build (default)

Default image (no forced CPU index):

```bash
docker build -f docker/Dockerfile docker -t leann:latest
```

## Build (CPU)

CPU-optimized image (forces PyTorch CPU index):

```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg LEANN_VERSION=0.3.6 \
  -f docker/Dockerfile.cpu docker -t leann:cpu
```

## Build (development)

Development image (source tree + build/test toolchain + `uv sync`):

```bash
docker build -f docker/Dockerfile.dev . -t leann:dev
```

## Run

```bash
docker run --rm leann:latest
docker run --rm leann:cpu
docker run --rm -it leann:dev
```

Expected output:

```text
LEANN installed and importable.
```

## Notes

- Both images keep LEANN package semantics unchanged (`pip install leann` with both backends).
- `Dockerfile.cpu` uses the PyTorch CPU index to avoid downloading CUDA wheels.
- `Dockerfile.dev` is for local development, not production deployment.
