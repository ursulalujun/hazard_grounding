# ssh -f -N -4 -L 23128:proxy.h.pjlab.org.cn:23128 luxiaoya@100.98.161.47
# export http_proxy="http://luxiaoya:U8z9i4bL10OCVplAEbVDbdP8t4EYnmJNFmRNQ0AK3cZeJjOjUDwhfcHf4fFz@127.0.0.1:23128"
# export https_proxy="$http_proxy"
# export HTTP_PROXY="$http_proxy"
# export HTTPS_PROXY="$http_proxy"

conda activate agent

cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding

export TARGET_API_URL="https://api.boyuerichdata.opensphereai.com/v1"
export TARGET_API_KEY="sk-jZMbdRTTbzZdabBYS74dOgvV6FWs1tkwZY6K8iCCi4aSjqdN"

export EVALUATION_API_URL="https://api.boyuerichdata.opensphereai.com/v1"
export EVALUATION_API_KEY="sk-jZMbdRTTbzZdabBYS74dOgvV6FWs1tkwZY6K8iCCi4aSjqdN"

python oversafety_evaluation.py --hazard_type action_triggered --target_model /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct-RFT-action_triggered --data_type train

sleep 20s

python oversafety_evaluation.py --hazard_type action_triggered --target_model /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct-RFT-action_triggered

sleep 20s

python evaluation.py --hazard_type action_triggered --target_model checkpoints/Qwen3-VL-8B-Instruct-RFT-action_triggered-mixed --data_type train

sleep 20s

python evaluation.py --hazard_type action_triggered --target_model checkpoints/Qwen3-VL-8B-Instruct-RFT-action_triggered-mixed