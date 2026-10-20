ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG LEANN_VERSION=0.3.6

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Keep runtime image minimal while ensuring common C++ runtime libs are present.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && pip install --prefer-binary "leann==${LEANN_VERSION}" \
    && python -c "import leann; import leann_backend_hnsw; import leann_backend_diskann"

WORKDIR /workspace

CMD ["python", "-c", "import leann; print('LEANN installed and importable.')"]
