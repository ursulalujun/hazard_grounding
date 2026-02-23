#!/bin/bash
# RFT Training Script for Risk Grounding - Action Triggered Hazard Detection
# Adapted from Visual-RFT for safety hazard detection and localization
# source /mnt/shared-storage-user/luxiaoya/.bashrc
# conda activate rft
cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding
# ==============================================================================
# Configuration
# ==============================================================================

# Set W&B to offline mode (for nodes without internet access)
export WANDB_MODE=offline
export WANDB_PROJECT="hazard_grounding"

# Paths
export PROJECT_ROOT="/mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner"
export DATA_PATH="${PROJECT_ROOT}/risk_grounding/data_pipeline/data/rft_training_list.json"
export EMBEDDING_MODEL_PATH="${PROJECT_ROOT}/risk_grounding/checkpoints/all-MiniLM-L6-v2"

# Model checkpoint (update this path to your Qwen3-VL-8B-Instruct checkpoint)
# If the model is in shared storage, use the actual path
export CKPT_PATH="${PROJECT_ROOT}/risk_grounding/alignment/LlamaFactory/saves/Qwen3-VL-8B-Thinking_sft_merged"

# Output directory
# export SAVE_PATH="${PROJECT_ROOT}/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct-RFT-${HAZARD_TYPE}"

# DeepSpeed config
export DEEPSPEED_CONFIG="${PROJECT_ROOT}/risk_grounding/alignment/Visual-RFT/src/virft/local_scripts/zero3.json"

# ==============================================================================
# Reward Weights Configuration
# ==============================================================================
#
# For action_triggered hazard:
# - safe_accuracy: whether safety judgment is correct
# - safety_hazard_match: semantic similarity of safety hazard description
# - principle_accuracy: whether safety principle classification is correct
# - iou_target_object: localization of the object user needs to interact with
# - iou_constraint_object: localization of hazard-causing background objects
# - format: JSON format validity
#
# NOTE: target_object should ALWAYS be localized, regardless of scene safety.
#       The safety judgment depends on constraint_object (hazards), not target_object.
#
# ==============================================================================

REWARD_WEIGHT_SAFE_ACCURACY=1.0
REWARD_WEIGHT_SAFETY_HAZARD_MATCH=0.5
REWARD_WEIGHT_PRINCIPLE_ACCURACY=2.0
REWARD_WEIGHT_IOU_TARGET_OBJECT=2.0
REWARD_WEIGHT_IOU_CONSTRAINT_OBJECT=2.0
REWARD_WEIGHT_FORMAT=0.1
EPOCH_NUM=2

export RUN_NAME="Qwen3-VL-8B-Thinking-RFT-epoch${EPOCH_NUM}-wsh${REWARD_WEIGHT_SAFETY_HAZARD_MATCH}-wp${REWARD_WEIGHT_PRINCIPLE_ACCURACY}-wit${REWARD_WEIGHT_IOU_TARGET_OBJECT}-wic${REWARD_WEIGHT_IOU_CONSTRAINT_OBJECT}"
export SAVE_PATH="${PROJECT_ROOT}/risk_grounding/checkpoints/${RUN_NAME}"

# ==============================================================================
# Launch Training
# ==============================================================================

echo "=========================================="
echo "RFT Training for Risk Grounding"
echo "Data Path: ${DATA_PATH}"
echo "Model: ${CKPT_PATH}"
echo "Output: ${SAVE_PATH}"
echo ""
echo "Reward Weights:"
echo "  safe_accuracy: ${REWARD_WEIGHT_SAFE_ACCURACY}"
echo "  safety_hazard_match: ${REWARD_WEIGHT_SAFETY_HAZARD_MATCH}"
echo "  principle_accuracy: ${REWARD_WEIGHT_PRINCIPLE_ACCURACY}"
echo "  iou_target_object: ${REWARD_WEIGHT_IOU_TARGET_OBJECT}"
echo "  iou_constraint_object: ${REWARD_WEIGHT_IOU_CONSTRAINT_OBJECT}"
echo "  format: ${REWARD_WEIGHT_FORMAT}"
echo "=========================================="

# Check if DeepSpeed config exists
if [ ! -f "${DEEPSPEED_CONFIG}" ]; then
    echo "Warning: DeepSpeed config not found at ${DEEPSPEED_CONFIG}"
    echo "Creating default DeepSpeed zero3 config..."
    mkdir -p $(dirname ${DEEPSPEED_CONFIG})
    cat > ${DEEPSPEED_CONFIG} << 'EOF'
{
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}
EOF
fi

# Check if data file exists
if [ ! -f "${DATA_PATH}" ]; then
    echo "Error: Data file not found at ${DATA_PATH}"
    exit 1
fi

# Create output directory
mkdir -p ${SAVE_PATH}

# Set number of GPUs
# NUM_GPUS=${NUM_GPUS:-8}
NUM_GPUS=8

# Run training with torchrun
torchrun --nproc_per_node="${NUM_GPUS}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    ${PROJECT_ROOT}/risk_grounding/alignment/rft/train_rft.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_path ${DATA_PATH} \
    --embedding_model_path ${EMBEDDING_MODEL_PATH} \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --max_prompt_length 2048 \
    --max_completion_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 16 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --min_pixels 3136 \
    --num_train_epochs ${EPOCH_NUM} \
    --run_name ${RUN_NAME} \
    --save_steps 500 \
    --save_only_model false \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --reward_weight_safe_accuracy ${REWARD_WEIGHT_SAFE_ACCURACY} \
    --reward_weight_safety_hazard_match ${REWARD_WEIGHT_SAFETY_HAZARD_MATCH} \
    --reward_weight_principle_accuracy ${REWARD_WEIGHT_PRINCIPLE_ACCURACY} \
    --reward_weight_iou_target_object ${REWARD_WEIGHT_IOU_TARGET_OBJECT} \
    --reward_weight_iou_constraint_object ${REWARD_WEIGHT_IOU_CONSTRAINT_OBJECT} \
    --reward_weight_format ${REWARD_WEIGHT_FORMAT}

echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${SAVE_PATH}"
echo "=========================================="
