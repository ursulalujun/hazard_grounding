set -ex
source /mnt/shared-storage-user/luxiaoya/miniconda3/bin/activate 

# source /mnt/shared-storage-user/luxiaoya/.bashrc
conda activate agent

cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding

export TARGET_API_URL="https://api.boyuerichdata.opensphereai.com/v1"
export TARGET_API_KEY="sk-jZMbdRTTbzZdabBYS74dOgvV6FWs1tkwZY6K8iCCi4aSjqdN"

export EVALUATION_API_URL="http://100.99.199.196:40239/v1"
export EVALUATION_API_KEY="bearer"

# python -m evaluation.eval_mssbench --target_model checkpoints/Qwen3-VL-4B-Thinking-RFT-mixed-epoch2-wsh0.5-wp2.0-wit2.0-wic2.0 --version v1

python -m evaluation.eval_mssbench --target_model checkpoints/Qwen3-VL-4B-Thinking --version v3

# rj -n ear_eval -i "registry.h.pjlab.org.cn/ailab-ai4good1-ai4good1_gpu/vllm:py311_torch-2.8.0-cu128_vllm-0.11.0rc2" -g 1 bash -exc /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/scripts/eval/mssbench_eval.sh