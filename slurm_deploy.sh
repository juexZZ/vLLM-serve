#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --output=logs/serve_%j.out
#SBATCH --error=logs/serve_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --account=torch_pr_54_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jz4725@nyu.edu

set -euo pipefail

# Configurable paths (override via env when submitting)
OVERLAY_PATH="${OVERLAY_PATH:-/scratch/jz4725/singularity_envs/vlmvla.ext3:ro}"
CONTAINER_SIF="${CONTAINER_SIF:-/scratch/jz4725/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"

echo "Using overlay: ${OVERLAY_PATH}"
echo "Using container: ${CONTAINER_SIF}"

echo "Launching vLLM serve inside Singularity..."

# Use srun to attach resources to the containerized process
singularity exec --nv \
  --overlay "${OVERLAY_PATH}" /bin/bash <<'SINGULARITY_SCRIPT'

# Environment inside the container
export HF_HOME="/scratch/$USER/hf-cache"
export SSL_CERT_FILE=/scratch/jz4725/cacert-2025-11-04.pem
export PATH="/ext3/uv:$PATH"

# Confirm working directory
echo "Container working directory: $(pwd)"

source .venv/bin/activate

# Model and tensor-parallel config (override via env)
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
# MODEL_NAME="Qwen/Qwen3-VL-235B-A22B-Instruct"  # for larger deployments

# check CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

TP_SIZE="${TP_SIZE:-$NUM_GPU}"

# Start vLLM server
vllm serve "$MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling
SINGULARITY_SCRIPT
