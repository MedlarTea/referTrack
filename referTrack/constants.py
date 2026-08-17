"""Eval-only constants for ReferTrack."""
from __future__ import annotations

import math
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parents[1])

# Catalog / CoT
MAX_CANDIDATES = 20
NO_EXIST_TOKEN = "<NO_EXIST>"
OBJ_TOKEN_FMT = "<obj_{}>"

SEG_MARKER_TOKENS = {
    "cat_open": "<cat>",
    "cat_close": "</cat>",
    "vis_open": "<vis>",
    "vis_close": "</vis>",
    "reasoning_open": "<reasoning_refer>",
    "reasoning_close": "</reasoning_refer>",
}

# 20 obj tokens + 1 NO_EXIST + 6 segment markers
REFER_NEW_SPECIAL_TOKENS_COUNT = 27

# Online tracker (must match the released checkpoint's data-generation tracker)
REFER_TRACKER_CFG = {
    "yolo_model": "yolo11x.pt",
    "tracker_yaml": "bytetrack.yaml",
    "classes": [0],  # COCO person
    "conf": 0.1,
    "iou": 0.9,
    "imgsz": 384,
    "target_iou_thresh": 0.2,
}

VIEW_YAWS = {
    "forward": 0,
    "left": math.pi / 2,
    "right": 3 * math.pi / 2,
    "back": 2 * math.pi,
}
