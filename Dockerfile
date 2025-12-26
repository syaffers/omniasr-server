FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libsndfile1 \
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

# Install dependencies with uv
RUN uv sync --extra-index-url https://download.pytorch.org/whl/cu126

COPY app/ .
COPY main.py .

# Expose the API port
EXPOSE 8080

# Run the server
CMD ["uv", "run", "main.py"]
