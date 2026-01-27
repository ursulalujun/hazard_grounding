INPUT_LEN=$1
SEG_LEN=$2

SEG_COUNT=$(( (INPUT_LEN + SEG_LEN - 1) / SEG_LEN ))
echo "Input length: $INPUT_LEN, segment length: $SEG_LEN, Running in $SEG_COUNT segments..."

for ((i=0; i<SEG_COUNT; i++)); do
    START=$((i * SEG_LEN))
    END=$((START + SEG_LEN < INPUT_LEN ? START + SEG_LEN : INPUT_LEN))
    rj -g 1 -n scene-editing -i registry.h.pjlab.org.cn/ailab-ai4good1/luxiaoya-workspace:hazard_grounding bash -exc scripts/scene_editing.sh $START $END
done