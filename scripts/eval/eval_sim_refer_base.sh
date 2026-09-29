#!/bin/bash
# ============================================================
# ReferTrack Habitat simulation eval (DT / STT / AT)
#
# Closed-loop eval in Habitat Track:
#   - online YOLO + ByteTrack (default yolo11x + bytetrack)
#   - ReferAgent: catalog + per-step CoT + planner
#   - each episode writes info.json; SAVE_VIDEO=1 also writes mp4
#
# Layout:
#   ckpt:  $LOG_ROOT/$MODEL/$CKPT
#   out:   $LOG_ROOT/$MODEL/eval_sim_refer_${CKPT_TAG}/$SPLIT/
#
# Requirements:
#   - habitat-lab installed (`pip install -e habitat-lab`)
#   - habitat-sim 0.3.1 withbullet
#   - EVT-Bench scenes / humanoids laid out as in README
#   - ultralytics + yolo11x.pt (downloaded on first run)
#   - ckpt directory contains model_config.json
#     (bash scripts/eval/download_ckpt.sh)
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

CHUNKS="${CHUNKS:-30}"
NUM_PARALLEL="${NUM_PARALLEL:-1}"
MAX_NUMS="${MAX_NUMS:--1}"

SPLITS=("dt")
# SPLITS=("stt" "at")

MODEL="${MODEL:-ReferTrack-Qwen3-4B}"
CKPT="${CKPT:-refertrack_qwen3_4b.pt}"
YOLO_MODEL="${YOLO_MODEL:-yolo11x.pt}"

# CoT predicts NO_EXIST: 0 = trust planner (default); 1 = force stop
FALLBACK_STOP="${FALLBACK_STOP:-0}"
# 1 = write episode mp4 (slow); 0 = json/summary only
SAVE_VIDEO="${SAVE_VIDEO:-0}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

LOG_ROOT="${LOG_ROOT:-$(pwd)/data/logs}"
LOG_PATH="$LOG_ROOT/$MODEL"
CKPT_PATH="$LOG_PATH/$CKPT"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Checkpoint not found: $CKPT_PATH"
    echo "Download it with: bash scripts/eval/download_ckpt.sh"
    exit 1
fi

CKPT_TAG="${CKPT%.pt}"

for SPLIT in "${SPLITS[@]}"; do
    SAVE_PATH="$LOG_PATH/eval_sim_refer_${CKPT_TAG}/${SPLIT}"

    echo "========================================"
    echo "MODEL=$MODEL"
    echo "CKPT=$CKPT"
    echo "YOLO_MODEL=$YOLO_MODEL"
    echo "SPLIT=$SPLIT"
    echo "SAVE_PATH=$SAVE_PATH"
    echo "CHUNKS=$CHUNKS  NUM_PARALLEL=$NUM_PARALLEL  MAX_NUMS=$MAX_NUMS"
    echo "FALLBACK_STOP=$FALLBACK_STOP  SAVE_VIDEO=$SAVE_VIDEO"
    echo "========================================"

    IDX=0
    while [ $IDX -lt $CHUNKS ]; do
        for ((i=0; i<NUM_PARALLEL && IDX<CHUNKS; i++)); do
            GPU=$((i))
            echo "Launching IDX=$IDX on GPU=$GPU"

            EXTRA_ARGS=""
            if [ "$FALLBACK_STOP" = "1" ]; then
                EXTRA_ARGS="--fallback-stop"
            fi

            CUDA_VISIBLE_DEVICES=$GPU SAVE_VIDEO=$SAVE_VIDEO PYTHONPATH="habitat-lab:${PYTHONPATH:-}" \
            python referTrack/eval/run_eval_refer_sim.py \
                --split-num "$CHUNKS" \
                --split-id "$IDX" \
                --max-nums "$MAX_NUMS" \
                --exp-config "habitat-lab/habitat/config/benchmark/nav/track/track_infer_${SPLIT}.yaml" \
                --run-type eval \
                --save-path "$SAVE_PATH" \
                --ckpt-path "$CKPT_PATH" \
                --yolo-model "$YOLO_MODEL" \
                $EXTRA_ARGS &

            IDX=$((IDX + 1))
        done
        wait
    done
done

echo "========================================"
echo "ReferTrack simulation eval finished."
echo "Summarize with:"
echo "  python -m referTrack.tool.print_eval_result --result-dir $LOG_PATH/eval_sim_refer_${CKPT_TAG} --splits ${SPLITS[*]}"
echo "========================================"
