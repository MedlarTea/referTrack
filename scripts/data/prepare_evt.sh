#!/bin/bash
# ============================================================
# Raw EVT-Bench episodes -> ReferTrack training data.
#
#   RAW_ROOT/<split>/seed_*/<scene>/<k>{.mp4,_info.json,.json}
#   -> DATA_ROOT/<split>/{frames,tracks,jsonl,vision_cache}
#
# Step 1: frames + YOLO/ByteTrack tracks + target matching -> *_withTrack.jsonl
# Step 2: DINOv3 + SigLIP tokens -> vision_cache/
# Both steps skip finished outputs, so the script can be re-run after interruption.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

RAW_ROOT="${RAW_ROOT:-$(pwd)/data/evt_bench}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/data/evt_bench_train}"
SPLITS=(${SPLITS:-stt_singleview_train dt_singleview_train at_singleview_train})
NUM_GPUS="${NUM_GPUS:-8}"
YOLO_MODEL="${YOLO_MODEL:-yolo11x.pt}"
BATCH_SIZE="${BATCH_SIZE:-256}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

for SPLIT in "${SPLITS[@]}"; do
    echo "==== [$SPLIT] build jsonl ===="
    for ((i=0; i<NUM_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i python referTrack/tool/build_refer_jsonl.py \
            --input_root "$RAW_ROOT/$SPLIT" --output_root "$DATA_ROOT/$SPLIT" \
            --yolo_model "$YOLO_MODEL" --device cuda:0 --rank $i --world_size $NUM_GPUS &
    done
    wait

    echo "==== [$SPLIT] precache features ===="
    for ((i=0; i<NUM_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i python referTrack/tool/precache_features.py \
            --data_root "$DATA_ROOT/$SPLIT" --batch_size $BATCH_SIZE --rank $i --world_size $NUM_GPUS &
    done
    wait
done
echo "Done: $DATA_ROOT"
