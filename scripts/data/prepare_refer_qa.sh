#!/bin/bash
# ============================================================
# SYNTH-PEDES -> refer-QA composites + vision cache.
#
#   $SYNTH_ROOT/{synthpedes-dataset.json, Part*/, backgrounds/}
#   -> $OUT_DIR/{images/, info.json, val_indices.json, vision_cache/}
#
# backgrounds/ is not part of SYNTH-PEDES: put your own scene images there (any sub-folders).
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

SYNTH_ROOT="${SYNTH_ROOT:-$(pwd)/data/SYNTH-PEDES}"
OUT_DIR="${OUT_DIR:-$(pwd)/data/refer_vqa_dataset}"
TOTAL_IMAGES="${TOTAL_IMAGES:-1300000}"
NUM_GPUS="${NUM_GPUS:-8}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p "$(dirname "$OUT_DIR")"
python referTrack/tool/make_refer_qa.py \
    --data_root "$SYNTH_ROOT" --output_dir "$OUT_DIR" --total_images "$TOTAL_IMAGES" --seed 42

for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python referTrack/tool/precache_features.py \
        --data_root "$OUT_DIR" --image_dir images --view "" --rank $i --world_size $NUM_GPUS &
done
wait
echo "Done: $OUT_DIR"
