#!/bin/bash
# RFT Training Script with LoRA - Risk Grounding Action Triggered Hazard Detection
# Adapted from Visual-RFT for safety hazard detection and localization
source /mnt/shared-storage-user/luxiaoya/.bashrc
conda activate rft
cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding
# ==============================================================================
# Configuration
# ==============================================================================

# Set W&B to offline mode (for nodes without internet access)
export WANDB_MODE=offline
export WANDB_PROJECT="hazard_grounding"

# Paths
export PROJECT_ROOT="/mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner"
export DATA_PATH="${PROJECT_ROOT}/risk_grounding/data_pipeline/data/success_list_with_cot.json"
export EMBEDDING_MODEL_PATH="${PROJECT_ROOT}/risk_grounding/checkpoints/all-MiniLM-L6-v2"

# Model checkpoint (update this path to your Qwen3-VL-8B-Instruct checkpoint)
# If the model is in shared storage, use the actual path
export CKPT_PATH="${PROJECT_ROOT}/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct"

# DeepSpeed config (use zero2 for more stable training with LoRA)
# ZeRO Stage 3 has compatibility issues with LoRA + Vision models
export DEEPSPEED_CONFIG="${PROJECT_ROOT}/Visual-RFT/src/virft/local_scripts/zero2.json"
# export DEEPSPEED_CONFIG="${PROJECT_ROOT}/Visual-RFT/src/virft/local_scripts/zero3.json"

# ==============================================================================
# LoRA Configuration
# ==============================================================================

# LoRA rank (higher = more parameters, better performance but more memory)
LORA_R=64

# LoRA alpha (scaling factor, typically r/2 or r/4)
LORA_ALPHA=16

# LoRA dropout (typically 0.05-0.1)
LORA_DROPOUT=0.05

# Target modules (which layers to apply LoRA to)
# Options: "all-linear" (recommended for best performance) or specific modules like "q_proj,v_proj"
LORA_TARGET_MODULES="all-linear"

# ==============================================================================
# Reward Weights Configuration
# ==============================================================================
#
# For action_triggered hazard:
# - safe_accuracy: whether safety judgment is correct
# - risk_match: semantic similarity of risk description
# - iou_target_object: localization of the object user needs to interact with
# - iou_constraint_object: localization of hazard-causing background objects
# - format: JSON format validity
#
# NOTE: target_object should ALWAYS be localized, regardless of scene safety.
#       The safety judgment depends on constraint_object (hazards), not target_object.
#
# ==============================================================================

REWARD_WEIGHT_SAFE_ACCURACY=0.5
REWARD_WEIGHT_RISK_MATCH=1.0
REWARD_WEIGHT_IOU_TARGET_OBJECT=1.0
REWARD_WEIGHT_PRINCIPLE_ACCURACY=1.0
REWARD_WEIGHT_IOU_CONSTRAINT_OBJECT=1.5
REWARD_WEIGHT_FORMAT=0.1

# Output directory (add -LoRA suffix to distinguish from full fine-tuning)
export RUN_NAME="Qwen3-VL-8B-Instruct-RFT-${HAZARD_TYPE}-LoRA-wr${REWARD_WEIGHT_RISK_MATCH}-wit${REWARD_WEIGHT_IOU_TARGET_OBJECT}-wic${REWARD_WEIGHT_IOU_CONSTRAINT_OBJECT}"
export SAVE_PATH="${PROJECT_ROOT}/risk_grounding/checkpoints/${RUN_NAME}"

# ==============================================================================
# Launch Training
# ==============================================================================

echo "=========================================="
echo "RFT Training with LoRA for Risk Grounding"
echo "Hazard Type: ${HAZARD_TYPE}"
echo "Data Path: ${DATA_PATH}"
echo "Model: ${CKPT_PATH}"
echo "Output: ${SAVE_PATH}"
echo ""
echo "LoRA Config:"
echo "  Rank: ${LORA_R}"
echo "  Alpha: ${LORA_ALPHA}"
echo "  Dropout: ${LORA_DROPOUT}"
echo "  Target Modules: ${LORA_TARGET_MODULES}"
echo ""
echo "Reward Weights:"
echo "  safe_accuracy: ${REWARD_WEIGHT_SAFE_ACCURACY}"
echo "  risk_match: ${REWARD_WEIGHT_RISK_MATCH}"
echo "  iou_target_object: ${REWARD_WEIGHT_IOU_TARGET_OBJECT}"
echo "  iou_constraint_object: ${REWARD_WEIGHT_IOU_CONSTRAINT_OBJECT}"
echo "  format: ${REWARD_WEIGHT_FORMAT}"
echo "=========================================="

# Check if DeepSpeed config exists
if [ ! -f "${DEEPSPEED_CONFIG}" ]; then
    echo "Warning: DeepSpeed config not found at ${DEEPSPEED_CONFIG}"
    echo "Creating default DeepSpeed zero2 config..."
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
        "stage": 2,
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
    "train_micro_batch_size_per_gpu": "auto"
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
NUM_GPUS=4

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
    --hazard_type ${HAZARD_TYPE} \
    --embedding_model_path ${EMBEDDING_MODEL_PATH} \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_generations 16 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --use_peft true \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    --reward_weight_safe_accuracy ${REWARD_WEIGHT_SAFE_ACCURACY} \
    --reward_weight_risk_match ${REWARD_WEIGHT_RISK_MATCH} \
    --reward_weight_iou_target_object ${REWARD_WEIGHT_IOU_TARGET_OBJECT} \
    --reward_weight_principle_accuracy ${REWARD_WEIGHT_PRINCIPLE_ACCURACY} \
    --reward_weight_iou_constraint_object ${REWARD_WEIGHT_IOU_CONSTRAINT_OBJECT} \
    --reward_weight_format ${REWARD_WEIGHT_FORMAT}

echo "=========================================="
echo "Training completed!"
echo "LoRA adapter saved to: ${SAVE_PATH}"
echo ""
echo "To merge LoRA adapter with base model, use:"
echo "  python scripts/merge_lora.py --base_model ${CKPT_PATH} --adapter ${SAVE_PATH} --output_dir ${SAVE_PATH}-merged"
echo "=========================================="
