#!/usr/bin/env bash
# Download hjyeee/ReferTrack-Qwen3-4B into data/logs/ReferTrack-Qwen3-4B/.
#
# Needed next to the .pt:
#   model_config.json
#
# Usage:
#   bash scripts/eval/download_ckpt.sh
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/eval/download_ckpt.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT_HF_REPO="${CKPT_HF_REPO:-hjyeee/ReferTrack-Qwen3-4B}"
DEST="${LOG_ROOT:-$(pwd)/data/logs}/ReferTrack-Qwen3-4B"

mkdir -p "$DEST"

python - <<PY
import os
from huggingface_hub import hf_hub_download
repo = "${CKPT_HF_REPO}"
dest = "${DEST}"
for name in ("refertrack_qwen3_4b.pt", "model_config.json"):
    local = os.path.join(dest, name)
    if os.path.isfile(local):
        print(f"skip {name} (exists)")
        continue
    path = hf_hub_download(repo_id=repo, filename=name, local_dir=dest)
    print(f"OK {name} -> {path}")
PY

if [ -f "$DEST/refertrack_qwen3_4b.pt" ] && [ -f "$DEST/model_config.json" ]; then
    echo "Checkpoint ready: $DEST/refertrack_qwen3_4b.pt"
else
    echo "Checkpoint not complete yet (missing .pt or model_config.json)."
    exit 1
fi
