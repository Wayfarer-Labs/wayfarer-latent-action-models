#######################################################################
#  Dockerfile  –  CUDA 12.5 runtime + uv + system libs + project code #
#######################################################################

# ───────────────────────── base image ────────────────────────────────
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Expose NVDEC (video) libraries into the container at runtime.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# ───────────────────────── system packages ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

# ───────────────────────── install uv ────────────────────────────────
# 1. tini adds proper signal handling       2. uv is the fast Rust-based installer
RUN pip install --no-cache-dir "tini>=0.19" "uv>=0.2.8"

# so sigterm is forwarded
ENTRYPOINT ["/usr/local/bin/tini", "--"]

# ───────────────────────── project deps ──────────────────────────────
# Copy only the requirement files first to leverage Docker layer cache
COPY requirements.txt /tmp/req.txt

# Use uv as a drop-in replacement for pip.
# The --extra-index-url flag works exactly the same.
RUN uv pip install -r /tmp/req.txt \
       --extra-index-url=https://pypi.nvidia.com \
       --no-cache-dir

# ───────────────────────── project code ──────────────────────────────
WORKDIR /workspace/wayfarer-latent-action-models
COPY . .
RUN uv pip install -e .

# ───────────────────────── default cmd ───────────────────────────────
CMD ["bash"]
