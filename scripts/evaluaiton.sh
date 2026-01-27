conda activate agent

cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding

export TARGET_API_URL="https://api.boyuerichdata.opensphereai.com/v1"
export TARGET_API_KEY="sk-jZMbdRTTbzZdabBYS74dOgvV6FWs1tkwZY6K8iCCi4aSjqdN"

export EVALUATION_API_URL="https://api.boyuerichdata.opensphereai.com/v1"
export EVALUATION_API_KEY="sk-jZMbdRTTbzZdabBYS74dOgvV6FWs1tkwZY6K8iCCi4aSjqdN"

python oversafety_evaluation.py --hazard_type action_triggered --target_model /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct-RFT-action_triggered

sleep 20s

python oversafety_evaluation.py --hazard_type action_triggered --target_model /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct

sleep 20s

python oversafety_evaluation.py --hazard_type action_triggered --target_model /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct-RFT-action_triggered-iou

sleep 20s

python evaluation.py --hazard_type action_triggered --target_model /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct-RFT-action_triggered-iou/checkpoint-600 --iou_with_label

sleep 20s

python evaluation.py --hazard_type action_triggered --target_model /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct-RFT-action_triggered --iou_with_label

sleep 20s

python evaluation.py --hazard_type action_triggered --target_model /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/checkpoints/Qwen3-VL-8B-Instruct --iou_with_label