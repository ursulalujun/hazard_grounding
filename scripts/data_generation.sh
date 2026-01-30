cd /mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/data_pipeline

conda activate agent

# 用rjob起一个Qwen3-vl-thinking vllm，image：registry.h.pjlab.org.cn/ailab-ai4good1/luxiaoya-workspace:hazard_grounding，需要4*H200
rj -n vllm-qwen3 -i "registry.h.pjlab.org.cn/ailab-ai4good1-ai4good1_gpu/vllm:py311_torch-2.8.0-cu128_vllm-0.11.0rc2" -g 4 sleep inf
which nvcc
bash serve_vllm.sh

# 设置环境变量，更改一下url，换成vllm的ip和port
bash env.sh

# 生成planning前先统计已经生成的样本分布，更新principle_checkpoint.json
TYPE=environmental # action_triggered
NUMP=$2
FOLDER=/mnt/shared-storage-user/zhouyijin/workspace/MyProj/hazard_grounding/data_pipeline/data

python -m nodes.editing_planner --hazard_type $TYPE --max_per_principle $NUMP --root_folder $FOLDER

# 加了一个阶段 object augmentation
python -m nodes.item_replacement --mode replace --hazard_type $TYPE --replace_model Qwen/Qwen3-VL-235B-A22B-Thinking

# Qwen-Image-Edit生成数据的速度很慢，需要数据分片并行，需要1*H200
START=$4
END=$5

python -m nodes.scene_editor --hazard_type $TYPE --editor_model /mnt/shared-storage-user/ai4good1-share/models/Qwen-Image-Edit-2511 --min_index $START --max_index $END

python -m nodes.fidelity_verifier --hazard_type $TYPE --root_folder $FOLDER

python -m nodes.hazard_verifier --hazard_type $TYPE --root_folder $FOLDER