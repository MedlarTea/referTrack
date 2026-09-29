#!/bin/bash
# ============================================================
# Collect EVT-Bench expert episodes (forward RGB, 384x384, 10 fps) with the rule-based expert.
#
#   -> $RAW_ROOT/{stt,dt,at}_singleview_train/seed_101{,_failed}/<scene>/<k>{.mp4,_info.json,.json}
#
# Each split's train episodes are cut into CHUNKS pieces; one process per chunk.
# Finished episodes are skipped, so the script can be re-run after interruption.
# Needs the Habitat assets from README section 2 (HM3D / MP3D train scenes, humanoids).
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

RAW_ROOT="${RAW_ROOT:-$(pwd)/data/evt_bench}"
SPLITS=(${SPLITS:-stt dt at})
CHUNKS="${CHUNKS:-30}"
NUM_GPUS="${NUM_GPUS:-8}"
SEED=101

export PYTHONPATH="$(pwd):$(pwd)/habitat-lab:${PYTHONPATH:-}"
mkdir -p "$RAW_ROOT"

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    for ((gpu = 0; gpu < NUM_GPUS && IDX < CHUNKS; gpu++)); do
        for SPLIT in "${SPLITS[@]}"; do
            CUDA_VISIBLE_DEVICES=$gpu python referTrack/tool/collect_expert.py \
                --exp-config "habitat-lab/habitat/config/benchmark/nav/track/track_train_${SPLIT}.yaml" \
                --split-num "$CHUNKS" --split-id "$IDX" \
                --save-path "$RAW_ROOT/${SPLIT}_singleview_train/seed_${SEED}" \
                habitat.simulator.seed=$SEED \
                > "$RAW_ROOT/collect_${SPLIT}_${IDX}.log" 2>&1 &
        done
        IDX=$((IDX + 1))
    done
    wait
done
echo "Done: $RAW_ROOT"
