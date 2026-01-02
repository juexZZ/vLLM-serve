#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --output=logs/serve_%j.out
#SBATCH --error=logs/serve_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --account=torch_pr_54_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jz4725@nyu.edu

echo "Using overlay: ${OVERLAY_PATH}"
echo "Using container: ${CONTAINER_SIF}"

# Model and tensor-parallel config (override via env)
# MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen/Qwen3-VL-235B-A22B-Instruct"  # for larger deployments

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
  --max-model-len 32768 \
  --async-scheduling \
  --host 0.0.0.0 \
  --api-key "ai4ce"
SINGULARITY_SCRIPT
