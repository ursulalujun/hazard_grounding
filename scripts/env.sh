export no_proxy="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1/,100.99.245.215/,localhost,.pjlab.org.cn,.h.pjlab.org.cn"

export REPLACE_API_URL="http://100.96.191.3:50376/v1"
export REPLACE_API_KEY="bearer"

export AUG_API_URL="https://api.boyuerichdata.opensphereai.com/v1"
export AUG_API_KEY="sk-jZMbdRTTbzZdabBYS74dOgvV6FWs1tkwZY6K8iCCi4aSjqdN"

export PLAN_API_URL="http://100.96.191.3:50376/v1"
export PLAN_API_KEY="bearer"

export EDIT_API_URL="https://api.boyuerichdata.opensphereai.com/v1"
export EDIT_API_KEY="sk-jZMbdRTTbzZdabBYS74dOgvV6FWs1tkwZY6K8iCCi4aSjqdN"

export ANNOTATION_API_URL="http://100.96.191.3:50376/v1"
export ANNOTATION_API_KEY="bearer"

export VERIFY_API_URL="http://100.96.191.3:50376/v1"
export VERIFY_API_KEY="bearer"

# For evaluation 
export TARGET_API_URL="https://api.boyuerichdata.opensphereai.com/v1"
export TARGET_API_KEY="sk-jZMbdRTTbzZdabBYS74dOgvV6FWs1tkwZY6K8iCCi4aSjqdN"

export EVALUATION_API_URL="http://100.96.191.3:50376/v1"
export EVALUATION_API_KEY="bearer"

conda activate agent

python -m evaluation.evaluation --target_model checkpoints/Qwen3-VL-8B-Thinking --version v3

python -m evaluation.evaluation --target_model checkpoints/Qwen3-VL-8B-Thinking --adapter alignment/LlamaFactory/saves/qwen3-vl-8b-ins/lora/sft --version v1

python -m evaluation.oversafety_evaluation --target_model gemini-2.5-pro --version v2_cot