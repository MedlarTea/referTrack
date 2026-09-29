#!/usr/bin/env bash
# Download hjyeee/ReferTrack-Qwen3-4B into data/logs/ReferTrack-Qwen3-4B/.
#
# Needed next to the .pt:
#   model_config.json
#
# Usage:
#   bash scripts/eval/download_ckpt.sh
#   bash scripts/eval/download_ckpt.sh --stage1     # also the stage-1 warm-start weights for training
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/eval/download_ckpt.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT_HF_REPO="${CKPT_HF_REPO:-hjyeee/ReferTrack-Qwen3-4B}"
DEST="${LOG_ROOT:-$(pwd)/data/logs}/ReferTrack-Qwen3-4B"
# config.json is only fetched so the HF Hub counts downloads.
FILES="config.json refertrack_qwen3_4b.pt model_config.json"
[ "${1:-}" = "--stage1" ] && FILES="$FILES refertrack_qwen3_4b_stage1.pt"

mkdir -p "$DEST"

python - <<PY
import os
from huggingface_hub import hf_hub_download
repo = "${CKPT_HF_REPO}"
dest = "${DEST}"
for name in "${FILES}".split():
    local = os.path.join(dest, name)
    if os.path.isfile(local):
        print(f"skip {name} (exists)")
        continue
    path = hf_hub_download(repo_id=repo, filename=name, local_dir=dest)
    print(f"OK {name} -> {path}")
PY

for f in $FILES; do
    if [ ! -f "$DEST/$f" ]; then
        echo "Checkpoint not complete yet (missing $f)."
        exit 1
    fi
done
echo "Checkpoint ready: $DEST"
