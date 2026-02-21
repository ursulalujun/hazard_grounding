#!/bin/bash
set -ex
# rj -n vllm-qwen3-8b -i "registry.h.pjlab.org.cn/ailab-ai4good1-ai4good1_gpu/vllm:py311_torch-2.8.0-cu128_vllm-0.11.0rc2" -g 1 bash -exc /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/scripts/serve_vllm_8b.sh

export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding

MODEL=checkpoints/Qwen3-VL-8B-Thinking
SERVED_NAME=Qwen/Qwen3-VL-8B-Thinking
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=64
NUM_GPUS=1

python scripts/launch_vllm.py \
    --model $MODEL \
    --served_model_name $SERVED_NAME \
    --tp $NUM_GPUS \
    --max_model_len $MAX_MODEL_LEN \
    --max_num_seqs $MAX_NUM_SEQS > /dev/null 2>&1 &
