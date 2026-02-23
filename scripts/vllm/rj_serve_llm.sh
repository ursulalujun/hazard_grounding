rjob submit --charged-group="ai4good1_gpu" \
    --name=vllm-qwen3  \
    --gpu=4 \
    --memory=100000 \
    --cpu=128 \
    --private-machine=group \
    --mount=gpfs://gpfs1/zhouyijin:/mnt/shared-storage-user/zhouyijin \
    --mount=gpfs://gpfs1/ai4good1-share:/mnt/shared-storage-user/ai4good1-share \
    --image=registry.h.pjlab.org.cn/ailab-ai4good1-ai4good1_gpu/vllm:py311_torch-2.8.0-cu128_vllm-0.11.0rc2  \
    -- bash /mnt/shared-storage-user/zhouyijin/workspace/MyProj/hazard_grounding/scripts/serve_vllm.sh