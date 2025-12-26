# Omnilingual-ASR Model Server
# GPU-enabled container for high-performance ASR inference

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY server.py .
COPY schemas.py .

# Install dependencies with uv
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    litserve>=0.2.6 \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.32.0" \
    python-multipart>=0.0.12 \
    pydantic>=2.9.0 \
    soundfile>=0.12.1 \
    librosa>=0.10.2 \
    "torch>=2.4.0" \
    "omnilingual-asr @ git+https://github.com/facebookresearch/omnilingual-asr.git"

# Environment variables for configuration
ENV OMNILINGUAL_MODEL="omniASR_CTC_1B_v2" \
    OMNILINGUAL_BATCH_SIZE="4" \
    OMNILINGUAL_BATCH_TIMEOUT="0.05" \
    OMNILINGUAL_WORKERS="1" \
    OMNILINGUAL_PORT="8000" \
    # Optimize CUDA memory allocation
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" \
    # Disable tokenizers parallelism warning
    TOKENIZERS_PARALLELISM="false"

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["python3", "server.py"]
