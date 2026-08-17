#!/bin/bash
# ============================================================
# ReferTrack arbitrary video inference (no Habitat)
#
# Given an arbitrary forward-view video (mp4 / image folder) + a natural-language
# instruction, run a trained ReferTrack ckpt:
#   - Online YOLO+ByteTrack tracker extracts person bboxes
#   - DINOv3 + SigLIP extract vcoarse / vfine
#   - inference_refer_navigation: two forwards + KV cache
#   - Render gray candidates + red Pred box + trajectory + text panel → mp4
#
# Output layout (OUT_DIR/<basename>.*):
#   <basename>.mp4               main rendered video
#   <basename>__raw.mp4          raw forward-video copy (when COPY_RAW=1)
#   <basename>_frames/*.jpg      per-frame jpgs (when SAVE_FRAMES=1)
#   <basename>_preds.jsonl       per-frame pred (on by default)
#   <basename>_summary.json      clip-level summary (on by default)
#
# This release ships a single-view ckpt (view_list=['forward']): only VIDEO_FORWARD is required.
#
# Prerequisites:
#   - ckpt: data/logs/ReferTrack-Qwen3-4B/refertrack_qwen3_4b.pt
#     (bash scripts/eval/download_ckpt.sh)
#   - ultralytics + yolo11x.pt (auto-downloaded on first run)
#   - imageio[ffmpeg] to decode mp4; ffmpeg CLI to write the output mp4
# ============================================================

set -e
cd "$(dirname "$0")/../.."

# ========== Required: ckpt + forward video + instruction ==========
CKPT_PATH="${CKPT_PATH:-data/logs/ReferTrack-Qwen3-4B/refertrack_qwen3_4b.pt}"
VIDEO_FORWARD="${VIDEO_FORWARD:-}"
INSTRUCTION="${INSTRUCTION:-Follow the person.}"

# ========== Optional multi-view inputs ==========
VIDEO_LEFT="${VIDEO_LEFT:-}"
VIDEO_RIGHT="${VIDEO_RIGHT:-}"
VIDEO_BACK="${VIDEO_BACK:-}"

# ========== Output ==========
# Default: infer_video_<step>/ next to the ckpt
OUT_DIR="${OUT_DIR:-}"

# ========== Inference params ==========
FPS_OUT="${FPS_OUT:-8}"               # Playback fps of the output mp4
FRAME_STRIDE="${FRAME_STRIDE:-1}"     # Source-video frame stride; 4 is common for 30fps → ~8fps
MAX_FRAMES="${MAX_FRAMES:-0}"         # 0=entire clip
DEVICE="${DEVICE:-cuda}"

# ========== Artifact toggles ==========
SAVE_FRAMES="${SAVE_FRAMES:-0}"       # 1=save per-frame jpgs
SAVE_PREDS="${SAVE_PREDS:-1}"         # 1=save per-frame jsonl (on by default)
SAVE_SUMMARY="${SAVE_SUMMARY:-1}"     # 1=save summary.json (on by default)
COPY_RAW="${COPY_RAW:-0}"             # 1=copy/transcode a raw forward-video copy

# ========== Validate ==========
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: ckpt not found: $CKPT_PATH"
    echo "Download it with: bash scripts/eval/download_ckpt.sh"
    exit 1
fi
if [ -z "$VIDEO_FORWARD" ]; then
    echo "ERROR: VIDEO_FORWARD is required (mp4 file or image folder)"
    exit 1
fi
if [ ! -e "$VIDEO_FORWARD" ]; then
    echo "ERROR: VIDEO_FORWARD not found: $VIDEO_FORWARD"
    exit 1
fi

# ========== Auto OUT_DIR ==========
if [ -z "$OUT_DIR" ]; then
    CKPT_DIR="$(dirname "$CKPT_PATH")"
    if [ "$(basename "$CKPT_DIR")" = "model_weights" ] || [ "$(basename "$CKPT_DIR")" = "ckpts" ]; then
        CKPT_DIR="$(dirname "$CKPT_DIR")"
    fi
    CKPT_BASENAME="$(basename "$CKPT_PATH" .pt)"
    STEP_TAG="$(echo "$CKPT_BASENAME" | grep -oE 'step[0-9_]+' | head -n1)"
    [ -z "$STEP_TAG" ] && STEP_TAG="$CKPT_BASENAME"
    VIDEO_TAG="$(basename "${VIDEO_FORWARD%.*}")"
    OUT_DIR="$CKPT_DIR/infer_video_${STEP_TAG}/${VIDEO_TAG}"
fi
mkdir -p "$OUT_DIR"

# ========== Environment ==========
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========== Assemble CLI args ==========
ARGS=(
    --ckpt-path "$CKPT_PATH"
    --video-forward "$VIDEO_FORWARD"
    --instruction "$INSTRUCTION"
    --out-dir "$OUT_DIR"
    --fps-out "$FPS_OUT"
    --frame-stride "$FRAME_STRIDE"
    --max-frames "$MAX_FRAMES"
    --device "$DEVICE"
)
[ -n "$VIDEO_LEFT" ]  && ARGS+=(--video-left  "$VIDEO_LEFT")
[ -n "$VIDEO_RIGHT" ] && ARGS+=(--video-right "$VIDEO_RIGHT")
[ -n "$VIDEO_BACK" ]  && ARGS+=(--video-back  "$VIDEO_BACK")

[ "$SAVE_FRAMES" = "1" ] && ARGS+=(--save-frames)
[ "$SAVE_PREDS" = "0" ]  && ARGS+=(--no-save-preds)
[ "$SAVE_SUMMARY" = "0" ] && ARGS+=(--no-save-summary)
[ "$COPY_RAW" = "1" ]    && ARGS+=(--copy-raw-video)

# ========== Print config ==========
echo "============================================================"
echo "ReferTrack  Arbitrary Video Inference"
echo "============================================================"
echo "CKPT:        $CKPT_PATH"
echo "VIDEO[fwd]:  $VIDEO_FORWARD"
[ -n "$VIDEO_LEFT" ]  && echo "VIDEO[left]: $VIDEO_LEFT"
[ -n "$VIDEO_RIGHT" ] && echo "VIDEO[right]:$VIDEO_RIGHT"
[ -n "$VIDEO_BACK" ]  && echo "VIDEO[back]: $VIDEO_BACK"
echo "INSTRUCTION: $INSTRUCTION"
echo "OUT_DIR:     $OUT_DIR"
echo "FPS_OUT:     $FPS_OUT  FRAME_STRIDE: $FRAME_STRIDE  MAX_FRAMES: $MAX_FRAMES"
echo "SAVE_FRAMES: $SAVE_FRAMES  SAVE_PREDS: $SAVE_PREDS  SAVE_SUMMARY: $SAVE_SUMMARY  COPY_RAW: $COPY_RAW"
echo "DEVICE:      $DEVICE"
echo "============================================================"

# ========== Launch ==========
python -m referTrack.eval.run_eval_refer_video "${ARGS[@]}"

echo ""
echo "============================================================"
echo "Inference completed!"
echo "OUT_DIR: $OUT_DIR"
echo "============================================================"
