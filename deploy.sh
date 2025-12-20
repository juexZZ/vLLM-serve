#!/bin/bash

# Singularity container settings (override via env if needed)
OVERLAY_PATH="${OVERLAY_PATH:-/scratch/jz4725/singularity_envs/vlmvla.ext3:ro}"
CONTAINER_SIF="${CONTAINER_SIF:-/scratch/jz4725/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"

# Run everything inside the container
singularity exec --nv \
  --overlay "${OVERLAY_PATH}" \
  "${CONTAINER_SIF}" \
  bash -lc '
# set up environment variables inside container
export HF_HOME="/scratch/$USER/hf-cache"
export SSL_CERT_FILE=/scratch/jz4725/cacert-2025-11-04.pem
export PATH="/ext3/uv:$PATH"

# work from this script directory
echo "Current directory: $(pwd)"

source .venv/bin/activate

# for testing purposes only, 1 gpu
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

vllm serve "$MODEL_NAME" \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling

# for deployment, 8 gpu (example)
# MODEL_NAME="Qwen/Qwen3-VL-235B-A22B-Instruct" # for real deployment
# vllm serve "$MODEL_NAME" \
#   --tensor-parallel-size 8 \
#   --mm-encoder-tp-mode data \
#   --limit-mm-per-prompt.video 0 \
#   --async-scheduling
'