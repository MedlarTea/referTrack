#!/bin/bash
# ============================================================
# ReferTrack training (stage 2): EVT-Bench refer-navigation + SYNTH-PEDES refer-QA.
#
# Defaults reproduce the released ReferTrack-Qwen3-4B: 16 GPUs x batch 16,
# 5 epochs, warm-started from the stage-1 checkpoint.
#
# Single node:  NUM_GPUS=8 bash scripts/train/train_refer.sh
# Multi node:   NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<ip> bash scripts/train/train_refer.sh
# Extra args are forwarded, e.g. `--max_steps 50 --save_every 25` for a smoke run.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_ROOT="${DATA_ROOT:-$(pwd)/data/evt_bench_train}"
REFER_QA_ROOT="${REFER_QA_ROOT:-$(pwd)/data/refer_vqa_dataset}"
LLM_NAME="${LLM_NAME:-$(pwd)/LLM_hf/qwen3-4b}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$(pwd)/data/logs/ReferTrack-Qwen3-4B/refertrack_qwen3_4b_stage1.pt}"
OUT_DIR="${OUT_DIR:-$(pwd)/data/logs/$(date +%y%m%d)-refertrack-qwen3-4b}"

NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_DIR"
torchrun --nnodes "$NNODES" --nproc_per_node "$NUM_GPUS" --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    referTrack/train/train_refer.py \
    --nav_roots "$DATA_ROOT/stt_singleview_train" "$DATA_ROOT/dt_singleview_train" "$DATA_ROOT/at_singleview_train" \
    --refer_qa_root "$REFER_QA_ROOT" \
    --llm_name "$LLM_NAME" \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
    --out_dir "$OUT_DIR" \
    "$@" 2>&1 | tee -a "$OUT_DIR/train.log"
