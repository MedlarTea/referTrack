#!/usr/bin/env python3
"""Build ReferTrack training samples from raw EVT-Bench expert episodes.

Input layout (one split, e.g. ``stt_singleview_train``)::

    <input_root>/seed_*/<scene>/<k>.mp4          forward RGB, 10 fps
    <input_root>/seed_*/<scene>/<k>_info.json    per-step robot/human state + GT ``human_bbox``
    <input_root>/seed_*/<scene>/<k>.json         episode result + ``instruction``

Output layout::

    <output_root>/frames/<seed>/<scene>/<k>/forward/frame_%05d.jpg
    <output_root>/tracks/<seed>/<scene>/<k>/forward.json     YOLO tracks cache
    <output_root>/jsonl/<seed>/<scene>/<k>_withTrack.jsonl   one sample per line

Each frame is tracked with YOLO + ByteTrack (``REFER_TRACKER_CFG``, same as the
online tracker at inference). The target track id is the track with the best
IoU against the simulator GT box; ``-1`` = target not visible, ``-2`` = visible
but no track passes ``target_iou_thresh``. Both become NO_EXIST samples.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from referTrack.constants import REFER_TRACKER_CFG

GT_MISSING_BBOX = [-1, 0, 0, 0, 0]
NO_MATCH_BBOX = [-2, 0, 0, 0, 0]


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gt_is_valid(gt: Optional[List[float]]) -> bool:
    return gt is not None and len(gt) >= 4 and gt[2] > gt[0] and gt[3] > gt[1]


def match_target(gt: Optional[List[float]], tracks: List[List[float]], iou_thresh: float) -> List[float]:
    """``[tid, x1, y1, x2, y2]`` of the best-IoU track, or a -1 / -2 sentinel."""
    if not gt_is_valid(gt):
        return list(GT_MISSING_BBOX)
    best_iou, best = -1.0, None
    for det in tracks:
        cur = iou_xyxy(gt, det[1:5])
        if cur > best_iou:
            best_iou, best = cur, det
    if best is None or best_iou < iou_thresh:
        return list(NO_MATCH_BBOX)
    return [int(best[0])] + [float(v) for v in best[1:5]]


def extract_frames(mp4: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4), "-q:v", "2", str(out_dir / "frame_%05d.jpg")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sorted(out_dir.glob("*.jpg"))


def run_tracking(model, mp4: Path, num_frames: int, cache: Path, device: Optional[str]) -> List[List[List[float]]]:
    """Per-frame ``[[tid, x1, y1, x2, y2], ...]`` in pixels, padded/truncated to ``num_frames``."""
    if cache.exists():
        tracks = json.loads(cache.read_text())
    else:
        cfg = REFER_TRACKER_CFG
        kwargs = dict(
            source=str(mp4), persist=False, stream=True, tracker=cfg["tracker_yaml"],
            classes=cfg["classes"], conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"], verbose=False,
        )
        if device:
            kwargs["device"] = device
        tracks = []
        for r in model.track(**kwargs):
            boxes = r.boxes
            if boxes is None or boxes.id is None:
                tracks.append([])
                continue
            ids = boxes.id.int().cpu().tolist()
            tracks.append([[tid] + box for tid, box in zip(ids, boxes.xyxy.cpu().tolist())])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(tracks))
    return (tracks + [[]] * num_frames)[:num_frames]


def is_success(status: Optional[dict]) -> bool:
    if not status:
        return False
    success = status.get("success")
    return (
        bool(status.get("finish"))
        or (isinstance(success, (int, float)) and success > 0)
        or "success" in str(status.get("status", "")).lower()
    )


def build_episode(info_json: Path, seed: str, scene: str, stem: str, out_root: Path,
                  model, args) -> Optional[List[dict]]:
    run_dir = info_json.parent
    status_path = run_dir / f"{stem}.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    if not is_success(status):
        return None
    steps = json.loads(info_json.read_text())
    instruction = (status.get("instruction") or "").strip() or "Follow the target person without collision."

    rel = Path(seed) / scene / stem
    frames = extract_frames(run_dir / f"{stem}.mp4", out_root / "frames" / rel / "forward")
    if not frames:
        return None
    frame_paths = [f"frames/{rel.as_posix()}/forward/{p.name}" for p in frames]
    n = len(frame_paths)

    tracks = run_tracking(model, run_dir / f"{stem}.mp4", n, out_root / "tracks" / rel / "forward.json", args.device)
    matches = [
        match_target(steps[f].get("human_bbox") if f < len(steps) else None, tracks[f],
                     REFER_TRACKER_CFG["target_iou_thresh"])
        for f in range(n)
    ]
    actions = [list(map(float, (s.get("base_velocity") or [0.0, 0.0, 0.0])[:3])) for s in steps]

    samples = []
    for j in range(n):
        if j + args.horizon > len(actions) - 1:
            continue
        start = max(0, j - args.history)
        step = steps[j] if j < len(steps) else {}
        samples.append({
            "images": {"forward": frame_paths[start:j]},
            "current": {"forward": frame_paths[j]},
            "instruction": instruction,
            "actions": actions[j: j + args.horizon + 1],
            "human_pos": step.get("human_pos"),
            "robot_pos": step.get("robot_pos"),
            "track_bboxes": {"forward": tracks[j]},
            "current_track_target_bbox": {"forward": matches[j]},
            "track_target_bbox_history": {"forward": matches[start:j]},
        })
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_root", required=True, help="Raw split root, e.g. .../stt_singleview_train")
    ap.add_argument("--output_root", required=True, help="Processed split root")
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--yolo_model", default=REFER_TRACKER_CFG["yolo_model"])
    ap.add_argument("--device", default=None, help="YOLO device, e.g. cuda:0")
    ap.add_argument("--rank", type=int, default=0, help="Shard index; episodes[rank::world_size]")
    ap.add_argument("--world_size", type=int, default=1)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    from ultralytics import YOLO

    in_root, out_root = Path(args.input_root).resolve(), Path(args.output_root).resolve()
    episodes = sorted(
        p for seed in sorted(in_root.glob("seed_*")) if seed.is_dir() and "failed" not in seed.name
        for p in seed.glob("*/*_info.json")
    )[args.rank::args.world_size]

    model = YOLO(args.yolo_model)
    written = 0
    for i, info_json in enumerate(episodes):
        stem = info_json.name[: -len("_info.json")]
        scene, seed = info_json.parent.name, info_json.parent.parent.name
        out = out_root / "jsonl" / seed / scene / f"{stem}_withTrack.jsonl"
        if out.exists() or not (info_json.parent / f"{stem}.mp4").exists():
            continue
        samples = build_episode(info_json, seed, scene, stem, out_root, model, args)
        if not samples:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(s) + "\n" for s in samples))
        written += 1
        if i % 50 == 0:
            print(f"[rank {args.rank}] {i + 1}/{len(episodes)} episodes", flush=True)
    print(f"[rank {args.rank}] wrote {written} episode files under {out_root / 'jsonl'}")


if __name__ == "__main__":
    main()
