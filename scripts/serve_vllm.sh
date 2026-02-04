#!/bin/bash
set -ex

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# source /mnt/shared-storage-user/luxiaoya/.bashrc
cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding

MODEL=/mnt/shared-storage-user/ai4good1-share/models/Qwen3-VL-235B-A22B-Thinking
SERVED_NAME=Qwen/Qwen3-VL-235B-A22B-Thinking
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=64
NUM_GPUS=4

python scripts/launch_vllm.py \
    --model $MODEL \
    --served_model_name $SERVED_NAME \
    --tp $NUM_GPUS \
    --max_model_len $MAX_MODEL_LEN \
    --max_num_seqs $MAX_NUM_SEQS

# nohup python scripts/launch_vllm.py \
#     --model $MODEL \
#     --served_model_name $SERVED_NAME \
#     --tp $NUM_GPUS \
#     --max_model_len $MAX_MODEL_LEN \
#     --max_num_seqs $MAX_NUM_SEQS > /dev/null 2>&1 &
