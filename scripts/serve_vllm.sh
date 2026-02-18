#!/bin/bash
# rj -n vllm-qwen3 -i "registry.h.pjlab.org.cn/ailab-ai4good1-ai4good1_gpu/vllm:py311_torch-2.8.0-cu128_vllm-0.11.0rc2" -g 4 bash -exc /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/scripts/serve_vllm.sh
set -ex

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# source /mnt/shared-storage-user/luxiaoya/.bashrc
cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding

MODEL=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-VL-235B-A22B-Thinking/snapshots/6664affde68449468deb7527186455c7450c13c0
SERVED_NAME=Qwen/Qwen3-VL-235B-A22B-Thinking
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=64
NUM_GPUS=4

# python scripts/launch_vllm.py \
#     --model $MODEL \
#     --served_model_name $SERVED_NAME \
#     --tp $NUM_GPUS \
#     --max_model_len $MAX_MODEL_LEN \
#     --max_num_seqs $MAX_NUM_SEQS

nohup python scripts/launch_vllm.py \
    --model $MODEL \
    --served_model_name $SERVED_NAME \
    --tp $NUM_GPUS \
    --max_model_len $MAX_MODEL_LEN \
    --max_num_seqs $MAX_NUM_SEQS > /dev/null 2>&1 &
