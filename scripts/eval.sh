set -ex
source /mnt/shared-storage-user/luxiaoya/miniconda3/bin/activate 

# source /mnt/shared-storage-user/luxiaoya/.bashrc
conda activate agent

cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding

export TARGET_API_URL="https://api.boyuerichdata.opensphereai.com/v1"
export TARGET_API_KEY="sk-jZMbdRTTbzZdabBYS74dOgvV6FWs1tkwZY6K8iCCi4aSjqdN"

export EVALUATION_API_URL="http://100.96.191.3:50376/v1"
export EVALUATION_API_KEY="bearer"

# python -m evaluation.evaluation --target_model checkpoints/Qwen3-VL-8B-Thinking --version v3 --scenario_type safe

# python -m evaluation.eval_earbench --target_model checkpoints/Qwen3-VL-8B-Instruct --version v2

python -m evaluation.pas_earbench --target_model checkpoints/Qwen3-VL-4B-Thinking-RFT-mixed-epoch2-wsh0.5-wp2.0-wit2.0-wic2.0 --version v1

# python -m evaluation.pas_earbench --target_model checkpoints/Qwen3-VL-4B-Thinking --version v3

# rj -n pas_eval -i "registry.h.pjlab.org.cn/ailab-ai4good1-ai4good1_gpu/vllm:py311_torch-2.8.0-cu128_vllm-0.11.0rc2" -g 1 bash -exc /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/scripts/eval.sh