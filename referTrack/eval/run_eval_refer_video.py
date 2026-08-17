#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline ReferTrack inference and visualization on arbitrary video + instruction (no Habitat).

Ports `referTrack/baseline/trained_agent_refer.py::ReferAgent.act()` —
online tracker + visual encoding + history-queue maintenance +
`inference_refer_navigation` — into a **pure offline video runner**:

  1. Decode a user-provided video (mp4/avi/mov) or image folder
  2. Per frame: `expand2square + resize 384x384`, matching the training distribution
  3. Online YOLO+ByteTrack (`REFER_TRACKER_CFG`) extracts person bboxes →
     area top-N + stable shuffle → catalog
  4. DINOv3 + SigLIP extract vcoarse / vfine tokens; keep an H-length coarse_hist
  5. Maintain a CoT-driven target_bbox_hist queue (same as training-side D-6)
  6. Per frame call `model.inference_refer_navigation(...)` (two forwards + KV-cache reuse)
     → pred_slot + trajectory
  7. Render each frame (gray candidate boxes + red Pred box + Pred trajectory + top text panel)
  8. Mux `<basename>.mp4` plus optional jpg frames / per-frame jsonl / summary.json /
     a transcoded copy of the raw video

Dependencies:
  - Required: torch, transformers, PIL, numpy, ffmpeg (CLI)
  - Video decode: imageio (strongly recommended for mp4) or cv2 (fallback)
  - Tracker: ultralytics (yolo11x.pt), bytetrack.yaml

About resolution:
  - Model input and rendering both use **384x384** (same as training) so
    catalog/bbox normalization matches training exactly.
  - Output mp4 defaults to 384x384; upsample in ffmpeg if you need HD.

Usage (single-view forward):
  python -m referTrack.eval.run_eval_refer_video \
    --ckpt-path /abs/path/to/model_weights/step_XXX.pt \
    --video-forward /path/to/walk.mp4 \
    --instruction "Follow the person in the red jacket." \
    --out-dir ./infer_out

Usage (multi-view; routed automatically from the ckpt's view_list):
  python -m referTrack.eval.run_eval_refer_video \
    --ckpt-path /abs/path/to/model_weights/step_XXX.pt \
    --video-forward /path/to/forward.mp4 \
    --video-left    /path/to/left.mp4 \
    --video-right   /path/to/right.mp4 \
    --instruction "Follow the person in the red jacket." \
    --out-dir ./infer_out
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from referTrack.constants import REFER_TRACKER_CFG, VIEW_YAWS
from referTrack.eval.load_refer_ckpt import load_refer_model
from referTrack.eval.vis_utils import (
    _draw_bbox_rect,
    _draw_text_panel,
    _draw_traj,
    _slot_to_label,
    _write_mp4,
)
from referTrack.eval.cache_gridpool import (
    VisionCacheConfig,
    VisionFeatureCacher,
    grid_pool_tokens,
)


# Fixed input size used when building training data (bbox normalization denominator)
BBOX_IMAGE_SIZE = 384

# Supported extensions for video / image-folder inputs
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================================
# Frame source (unified iterator for mp4 / image folder)
# =========================================================================


class _FrameSource:
    """Unified frame iterator: mp4 / image folder / single image all yield (frame_idx, RGB ndarray).

    Iterates with `frame_stride` (default 1 = no skip). Each frame is expand2square + resize
    to (target_size, target_size) to match the training distribution.
    """

    def __init__(self, path: str, target_size: int = 384, frame_stride: int = 1):
        if not osp.exists(path):
            raise FileNotFoundError(f"video path not found: {path}")
        self.path = path
        self.target_size = target_size
        self.frame_stride = max(1, int(frame_stride))
        self.kind: str  # "video" | "folder" | "image"
        self._fps_src: float = 0.0
        self._n_total_src: int = 0
        self._image_paths: List[str] = []
        self._reader = None
        self._init()

    def _init(self):
        if osp.isdir(self.path):
            paths = []
            for ext in IMAGE_EXTS:
                paths.extend(Path(self.path).glob(f"*{ext}"))
                paths.extend(Path(self.path).glob(f"*{ext.upper()}"))
            paths = sorted([str(p) for p in paths])
            if not paths:
                raise ValueError(f"No image files found in folder: {self.path}")
            self._image_paths = paths
            self._n_total_src = len(paths)
            self.kind = "folder"
            return
        ext = osp.splitext(self.path)[1].lower()
        if ext in IMAGE_EXTS:
            self._image_paths = [self.path]
            self._n_total_src = 1
            self.kind = "image"
            return
        if ext in VIDEO_EXTS:
            try:
                import imageio.v2 as iio  # noqa: WPS433
            except ImportError as e:
                raise RuntimeError(
                    "imageio is required to decode video files. pip install imageio[ffmpeg]"
                ) from e
            self._reader = iio.get_reader(self.path)
            try:
                meta = self._reader.get_meta_data()
                self._fps_src = float(meta.get("fps", 0.0))
                # nframes can be inf in newer imageio; count_frames is slower but exact
                n = meta.get("nframes", 0)
                if n is None or n == float("inf") or n <= 0:
                    try:
                        n = self._reader.count_frames()
                    except Exception:
                        n = 0
                self._n_total_src = int(n) if n else 0
            except Exception:
                self._fps_src = 0.0
                self._n_total_src = 0
            self.kind = "video"
            return
        raise ValueError(f"Unsupported video/image input: {self.path} (ext={ext})")

    @property
    def fps(self) -> float:
        return self._fps_src

    @property
    def n_total(self) -> int:
        return self._n_total_src

    @staticmethod
    def _expand2square_resize(rgb: np.ndarray, target_size: int) -> np.ndarray:
        """expand2square (black-border padding) + resize to (target_size, target_size).

        Matches `VisionFeatureCacher._expand2square`; done in one PIL pass here.
        """
        pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        w, h = pil.size
        if w != h:
            side = max(w, h)
            canvas = Image.new("RGB", (side, side), (0, 0, 0))
            canvas.paste(pil, ((side - w) // 2, (side - h) // 2))
            pil = canvas
        if pil.size != (target_size, target_size):
            pil = pil.resize((target_size, target_size), Image.BICUBIC)
        return np.asarray(pil, dtype=np.uint8)

    def __iter__(self):
        idx_out = 0
        if self.kind in ("folder", "image"):
            for src_idx, p in enumerate(self._image_paths):
                if src_idx % self.frame_stride != 0:
                    continue
                try:
                    pil = Image.open(p).convert("RGB")
                except Exception as e:
                    print(f"[FRAME] skip unreadable {p}: {e}")
                    continue
                rgb = np.asarray(pil, dtype=np.uint8)
                yield idx_out, src_idx, self._expand2square_resize(rgb, self.target_size)
                idx_out += 1
        elif self.kind == "video":
            assert self._reader is not None
            for src_idx, frame in enumerate(self._reader):
                if src_idx % self.frame_stride != 0:
                    continue
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                yield idx_out, src_idx, self._expand2square_resize(
                    np.ascontiguousarray(frame), self.target_size
                )
                idx_out += 1

    def close(self):
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass


# =========================================================================
# Tracker (same logic as baseline/trained_agent_refer.py; inlined to reduce coupling)
# =========================================================================


class _OnlineTracker:
    """Same as `trained_agent_refer.py::_OnlineTracker`."""

    def __init__(self):
        self._model = None
        self._track_kwargs = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed. pip install ultralytics."
            ) from e
        cfg = REFER_TRACKER_CFG
        self._model = YOLO(cfg["yolo_model"])
        self._track_kwargs = dict(
            classes=cfg["classes"],
            conf=cfg["conf"],
            iou=cfg["iou"],
            imgsz=cfg["imgsz"],
            tracker=cfg["tracker_yaml"],
            persist=True,
            verbose=False,
        )

    def reset(self):
        self._model = None
        self._track_kwargs = None

    def track_one(self, frame_rgb: np.ndarray) -> List[List[float]]:
        self._ensure_model()
        results = self._model.track(source=frame_rgb, **self._track_kwargs)
        out: List[List[float]] = []
        if not results:
            return out
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        ids = boxes.id
        ids_list = ids.cpu().numpy().tolist() if ids is not None else [-1] * len(xyxy)
        for tid, xy in zip(ids_list, xyxy):
            out.append([
                int(tid) if tid is not None else -1,
                float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3]),
            ])
        return out


# =========================================================================
# Catalog construction (same shuffle / normalize rules as ReferAgent._build_catalog)
# =========================================================================


def _bbox_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def build_catalog(
    track_bboxes: List[List[float]],
    n_max: int,
    seed_key: Tuple[Any, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int, List[int]]:
    """Same structure as `ReferAgent._build_catalog`:

      1. Clean [tid, x1,y1,x2,y2]
      2. Keep area top-N_MAX
      3. seed=hash((episode_key, step_idx)) stable shuffle
      4. Keep a virtual NO_EXIST slot at the end
    """
    cleaned: List[Tuple[int, List[float]]] = []
    for row in track_bboxes:
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            continue
        cleaned.append((int(row[0]), [float(v) for v in row[1:5]]))

    if len(cleaned) > n_max:
        cleaned = sorted(cleaned, key=lambda it: _bbox_area(it[1]), reverse=True)[:n_max]

    K = len(cleaned)
    if K > 0:
        seed = int((hash(seed_key) & 0xFFFFFFFF))
        rng = np.random.default_rng(seed=seed)
        order = rng.permutation(K)
        cleaned = [cleaned[i] for i in order]

    cand_bbox = torch.zeros(n_max + 1, 4, dtype=torch.float32)
    cand_slot_valid = torch.zeros(n_max + 1, dtype=torch.bool)
    tid_per_slot: List[int] = [-1] * n_max

    for k, (tid, bbox) in enumerate(cleaned):
        cand_bbox[k] = torch.tensor(bbox, dtype=torch.float32) / float(BBOX_IMAGE_SIZE)
        cand_slot_valid[k] = True
        tid_per_slot[k] = tid

    cand_slot_valid[n_max] = True
    return (
        cand_bbox.unsqueeze(0).to(device),
        cand_slot_valid.unsqueeze(0).to(device),
        K,
        tid_per_slot,
    )


# =========================================================================
# Vision-encoding helper (one frame → vcoarse(4,C) / vfine(64,C))
# =========================================================================


@torch.inference_mode()
def encode_frame_tokens(
    cacher: VisionFeatureCacher, rgb_np: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    pil = Image.fromarray(rgb_np.astype(np.uint8), mode="RGB")
    tok_dino, Hp, Wp = cacher._encode_dino([pil])
    tok_sigl = cacher._encode_siglip([pil], out_hw=(Hp, Wp))
    Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)
    Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)[0].float()
    Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)[0].float()
    return Vcoarse, Vfine


# =========================================================================
# Sequence-tensor assembly (strictly aligned with ReferAgent.act 5a-5e)
# =========================================================================


def assemble_sequence_tensors(
    coarse_hist: Dict[str, deque],
    fine_dict: Dict[str, torch.Tensor],
    target_bbox_hist: deque,
    view_list: List[str],
    history: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Assemble tensors required by `inference_refer_navigation`.

    Matches `trained_agent_refer.py::ReferAgent.act` §§5a-5e exactly:
      - coarse_tokens left-padded with the earliest frame;
      - fine_tokens: current-frame V views concatenated, tidx all H;
      - yaw from `VIEW_YAWS`;
      - bbox_hist from the CoT-driven `target_bbox_hist`, left-padded with the earliest non-zero.
    """
    H = history
    V = len(view_list)
    fwd_idx = view_list.index("forward")

    # 5a) coarse_tokens / coarse_tidx
    T = len(coarse_hist[view_list[0]])
    trim_len = min(H, T)
    missing = H - trim_len

    coarse_list: List[torch.Tensor] = []
    coarse_tidx_list: List[torch.Tensor] = []
    first_tok: Optional[torch.Tensor] = None
    pending_pad = missing
    for t in range(H):
        if t < missing:
            continue
        tok_per_view = [
            coarse_hist[v][t - missing].to(device) for v in view_list
        ]
        tok_views = torch.cat(tok_per_view, dim=0)  # (V*4, C)
        if first_tok is None:
            first_tok = tok_views
            for pt in range(pending_pad):
                coarse_list.append(first_tok)
                coarse_tidx_list.append(
                    torch.full((tok_views.size(0),), pt, dtype=torch.long, device=device)
                )
            pending_pad = 0
        coarse_list.append(tok_views)
        coarse_tidx_list.append(
            torch.full((tok_views.size(0),), t, dtype=torch.long, device=device)
        )
    coarse_tokens = torch.cat(coarse_list, dim=0).unsqueeze(0)
    coarse_tidx = torch.cat(coarse_tidx_list, dim=0).unsqueeze(0)

    # 5b) fine_tokens / fine_tidx
    fine_tokens = torch.cat(
        [fine_dict[v] for v in view_list], dim=0
    ).to(device).unsqueeze(0)
    fine_tidx = torch.full(
        (1, fine_tokens.size(1)), fill_value=H, dtype=torch.long, device=device,
    )

    # 5c) yaw
    yaw_hist = torch.tensor(
        [VIEW_YAWS[v] for v in view_list] * H, dtype=torch.float32,
    ).unsqueeze(0)
    yaw_curr = torch.tensor(
        [VIEW_YAWS[v] for v in view_list], dtype=torch.float32,
    ).unsqueeze(0)

    # 5d) bbox_hist (CoT-driven queue; left-pad with earliest non-zero)
    bbox_hist = torch.zeros(H * V, 4, dtype=torch.float32)
    Th = len(target_bbox_hist)
    if Th > 0:
        trim_h = min(H, Th)
        mh = H - trim_h
        for t in range(mh, H):
            bh = target_bbox_hist[t - mh]
            bbox_hist[t * V + fwd_idx] = torch.tensor(bh, dtype=torch.float32)
        earliest_valid = None
        for t in range(mh, H):
            b = bbox_hist[t * V + fwd_idx]
            if b.sum().item() > 0:
                earliest_valid = b.clone()
                break
        if earliest_valid is not None and 0 < mh < H:
            for t in range(mh):
                bbox_hist[t * V + fwd_idx] = earliest_valid
    bbox_hist = bbox_hist.unsqueeze(0).to(device)

    # 5e) bbox_curr: N-IMPL-2 decision — vis_f does not read bbox_curr; pass zeros
    bbox_curr = torch.zeros(V, 4, dtype=torch.float32).unsqueeze(0).to(device)

    return {
        "coarse_tokens": coarse_tokens,
        "coarse_tidx": coarse_tidx,
        "fine_tokens": fine_tokens,
        "fine_tidx": fine_tidx,
        "yaw_hist": yaw_hist,
        "yaw_curr": yaw_curr,
        "bbox_hist": bbox_hist,
        "bbox_curr": bbox_curr,
    }


# =========================================================================
# Render (no-GT version; gray candidates + red Pred box + trajectory + text)
# =========================================================================


def render_video_frame(
    rgb_np: np.ndarray,
    cand_bbox: np.ndarray,          # (CAT_LEN, 4) normalized
    cand_slot_valid: np.ndarray,    # (CAT_LEN,) bool
    pred_slot: int,
    pred_traj: np.ndarray,          # (n_wp, 3+)
    instruction: str,
    frame_idx: int,
    total_frames: int,
    src_frame_idx: int,
    n_max: int,
    scale_factor: float = 120.0,
    extra_lines: Optional[List[str]] = None,
) -> Image.Image:
    """Render a single frame (Pred only; no GT overlay)."""
    img = Image.fromarray(rgb_np.astype(np.uint8), mode="RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # 1) Gray candidate boxes (except the Pred slot)
    K = int(cand_slot_valid[:n_max].sum())
    for k in range(n_max):
        if not bool(cand_slot_valid[k]):
            continue
        bbox = cand_bbox[k]
        if bbox.sum() <= 0:
            continue
        if k == pred_slot:
            continue
        _draw_bbox_rect(draw, bbox, w, h, color=(160, 160, 160), width=2,
                        label=f"{k + 1}")

    # 2) Pred bbox (thick red); skip if NO_EXIST / out of range
    if 0 <= pred_slot < n_max:
        bbox = cand_bbox[pred_slot]
        if bbox.sum() > 0:
            _draw_bbox_rect(draw, bbox, w, h, color=(255, 80, 80), width=4,
                            label=f"Pred #{pred_slot + 1}")

    # 3) Pred trajectory (cyan, from bottom-center of the image)
    base_xy = (w // 2, int(h * 0.86))
    _draw_traj(draw, pred_traj, base_xy, color=(0, 255, 200), scale=scale_factor)

    # 4) Top text panel
    pred_lbl = _slot_to_label(pred_slot, n_max)
    lines = [
        f"Step {frame_idx + 1}/{total_frames}   src_idx={src_frame_idx}   K={K}",
        f"Instr: {instruction[:80]}",
        f"Pred: {pred_lbl} (slot={pred_slot})",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    img = _draw_text_panel(img, lines, position="top", bg_alpha=180)
    return img


# =========================================================================
# Main pipeline
# =========================================================================


def run_video(
    ckpt_path: str,
    video_paths: Dict[str, str],            # {"forward": ..., "left": ..., ...}
    instruction: str,
    out_dir: str,
    fps_out: int = 8,
    frame_stride: int = 1,
    max_frames: int = 0,
    save_frames: bool = False,
    save_preds_jsonl: bool = True,
    save_summary: bool = True,
    copy_raw_video: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ----- 1) Load model + validate view_list -----
    print(f"[SETUP] ckpt: {ckpt_path}")
    model, model_cfg = load_refer_model(ckpt_path, device)
    view_list = list(model_cfg.view_list or ["forward"])
    if "forward" not in view_list:
        raise ValueError(f"ckpt's view_list must contain 'forward', got {view_list}")
    missing_views = [v for v in view_list if v not in video_paths]
    if missing_views:
        raise ValueError(
            f"ckpt expects view_list={view_list} but missing video for: {missing_views}. "
            f"Provide --video-{'/'.join(missing_views)}."
        )
    extra_views = [v for v in video_paths if v not in view_list]
    if extra_views:
        print(f"[WARN] Ignoring extra video inputs not in view_list: {extra_views}")
        for v in extra_views:
            video_paths.pop(v, None)

    history = int(model_cfg.history)
    n_max = int(model_cfg.max_candidates)
    n_waypoints = int(model_cfg.n_waypoints)
    print(f"[SETUP] view_list={view_list}  history={history}  N_MAX={n_max}  n_wp={n_waypoints}")

    # ----- 2) Prepare frame iterators (multi-view aligned to the shortest video) -----
    sources: Dict[str, _FrameSource] = {}
    for v in view_list:
        src = _FrameSource(video_paths[v], target_size=BBOX_IMAGE_SIZE,
                           frame_stride=frame_stride)
        sources[v] = src
        print(f"[VIDEO] {v}: {video_paths[v]}  kind={src.kind}  src_fps={src.fps:.2f}  "
              f"src_n={src.n_total}  stride={frame_stride}")

    # ----- 3) Prepare tracker + vision cache -----
    tracker = _OnlineTracker()
    vision_cache_cfg = VisionCacheConfig(
        image_size=BBOX_IMAGE_SIZE,
        batch_size=1,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    cacher = VisionFeatureCacher(vision_cache_cfg).eval()

    # ----- 4) Output layout -----
    forward_path = video_paths["forward"]
    base = Path(forward_path).stem if osp.isfile(forward_path) else Path(forward_path).name
    os.makedirs(out_dir, exist_ok=True)
    video_path_out = osp.join(out_dir, f"{base}.mp4")
    frames_dir = osp.join(out_dir, f"{base}_frames") if save_frames else None
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    preds_path = osp.join(out_dir, f"{base}_preds.jsonl") if save_preds_jsonl else None
    summary_path = osp.join(out_dir, f"{base}_summary.json") if save_summary else None
    raw_video_path = osp.join(out_dir, f"{base}__raw.mp4") if copy_raw_video else None

    print(f"[OUT] video:   {video_path_out}")
    if frames_dir:    print(f"[OUT] frames:  {frames_dir}")
    if preds_path:    print(f"[OUT] preds:   {preds_path}")
    if summary_path:  print(f"[OUT] summary: {summary_path}")
    if raw_video_path: print(f"[OUT] raw:     {raw_video_path}")

    # ----- 5) Per-episode state -----
    coarse_hist: Dict[str, deque] = {v: deque(maxlen=history) for v in view_list}
    target_bbox_hist: deque = deque(maxlen=history)
    rendered_frames: List[np.ndarray] = []
    preds_records: List[Dict[str, Any]] = []
    cot_noexist_count = 0
    catalog_empty_count = 0
    inference_times: List[float] = []
    track_times: List[float] = []
    encode_times: List[float] = []

    iters = {v: iter(s) for v, s in sources.items()}
    print(f"[RUN] starting per-frame inference  instruction={instruction!r}")
    t_start_all = time.time()

    seq_idx = 0
    while True:
        if max_frames > 0 and seq_idx >= max_frames:
            print(f"[RUN] reached --max-frames={max_frames}")
            break

        # 5a) Grab the current frame from every view
        frames_this_step: Dict[str, Tuple[int, np.ndarray]] = {}
        try:
            for v in view_list:
                seq_i, src_i, rgb = next(iters[v])
                frames_this_step[v] = (src_i, rgb)
        except StopIteration:
            break

        forward_src_idx, forward_rgb = frames_this_step["forward"]

        # 5b) Tracker (forward only)
        t0 = time.time()
        track_bboxes = tracker.track_one(forward_rgb)
        track_times.append(time.time() - t0)

        # 5c) Build catalog
        seed_key = (osp.basename(forward_path), seq_idx)
        cand_bbox, cand_slot_valid, K, tid_per_slot = build_catalog(
            track_bboxes, n_max=n_max, seed_key=seed_key, device=device,
        )
        if K == 0:
            catalog_empty_count += 1

        # 5d) Vision encode (one frame per view)
        t0 = time.time()
        Vc_dict: Dict[str, torch.Tensor] = {}
        Vf_dict: Dict[str, torch.Tensor] = {}
        for v in view_list:
            _, rgb_v = frames_this_step[v]
            vc, vf = encode_frame_tokens(cacher, rgb_v)
            Vc_dict[v] = vc
            Vf_dict[v] = vf
            coarse_hist[v].append(vc.cpu())
        encode_times.append(time.time() - t0)

        # 5e) Assemble sequence tensors
        seq_t = assemble_sequence_tensors(
            coarse_hist=coarse_hist,
            fine_dict=Vf_dict,
            target_bbox_hist=target_bbox_hist,
            view_list=view_list,
            history=history,
            device=device,
        )

        t0 = time.time()
        out = model.inference_refer_navigation(
            coarse_tokens=seq_t["coarse_tokens"],
            coarse_tidx=seq_t["coarse_tidx"],
            fine_tokens=seq_t["fine_tokens"],
            fine_tidx=seq_t["fine_tidx"],
            cand_bbox=cand_bbox,
            cand_slot_valid=cand_slot_valid,
            bbox_hist=seq_t["bbox_hist"],
            bbox_curr=seq_t["bbox_curr"],
            instructions=[instruction],
            yaw_hist=seq_t["yaw_hist"],
            yaw_curr=seq_t["yaw_curr"],
            alpha=None,
        )
        inference_times.append(time.time() - t0)

        traj = out["trajectory"][0].detach().float().cpu().numpy()  # (n_wp, 3)
        pred_slot_int = int(out["pred_slot"].item())
        is_no_exist = (pred_slot_int == n_max)
        if is_no_exist:
            cot_noexist_count += 1

        # 5g) Update target_bbox_hist queue
        if 0 <= pred_slot_int < n_max:
            tgt_bb = cand_bbox[0, pred_slot_int].detach().cpu().numpy().tolist()
            target_bbox_hist.append(tgt_bb)
        else:
            target_bbox_hist.append([0.0, 0.0, 0.0, 0.0])

        # 5h) Render
        cand_bbox_np = cand_bbox[0].detach().cpu().numpy()
        cand_valid_np = cand_slot_valid[0].detach().cpu().numpy()
        pred_token_id = int(out["pred_token_ids"].item())
        extra_lines = [
            f"K={K}  no_exist={is_no_exist}  pred_token_id={pred_token_id}",
        ]
        rendered = render_video_frame(
            rgb_np=forward_rgb,
            cand_bbox=cand_bbox_np,
            cand_slot_valid=cand_valid_np,
            pred_slot=pred_slot_int,
            pred_traj=traj,
            instruction=instruction,
            frame_idx=seq_idx,
            total_frames=sources["forward"].n_total // max(1, frame_stride),
            src_frame_idx=forward_src_idx,
            n_max=n_max,
            extra_lines=extra_lines,
        )
        rendered_np = np.array(rendered)
        rendered_frames.append(rendered_np)

        if frames_dir:
            rendered.save(osp.join(frames_dir, f"step_{seq_idx:06d}.jpg"))

        # 5i) Record predictions
        if preds_path is not None:
            rec: Dict[str, Any] = {
                "frame_idx": seq_idx,
                "src_frame_idx": forward_src_idx,
                "pred_slot": pred_slot_int,
                "is_no_exist": bool(is_no_exist),
                "pred_token_id": pred_token_id,
                "cand_num": K,
                "target_tid": (
                    tid_per_slot[pred_slot_int]
                    if 0 <= pred_slot_int < n_max
                    else -1
                ),
                "trajectory": traj.tolist(),
                "cand_bbox": cand_bbox_np[: n_max].tolist(),
                "cand_slot_valid": cand_valid_np[: n_max].tolist(),
            }
            preds_records.append(rec)

        if (seq_idx + 1) % 20 == 0:
            print(
                f"  step {seq_idx + 1} | K={K} pred_slot={pred_slot_int} no_exist={is_no_exist} | "
                f"track={np.mean(track_times[-20:]) * 1000:.1f}ms "
                f"encode={np.mean(encode_times[-20:]) * 1000:.1f}ms "
                f"infer={np.mean(inference_times[-20:]) * 1000:.1f}ms"
            )

        seq_idx += 1

    elapsed_all = time.time() - t_start_all
    n_steps = seq_idx
    if n_steps == 0:
        print("[RUN] no frames processed; aborting")
        for s in sources.values():
            s.close()
        return {"error": "no_frames"}

    print(f"[RUN] processed {n_steps} frames in {elapsed_all:.1f}s "
          f"(avg {elapsed_all / n_steps * 1000:.1f}ms/frame)")

    # ----- 6) Write main video -----
    print(f"[OUT] writing {video_path_out} ({len(rendered_frames)} frames @ {fps_out}fps)")
    _write_mp4(video_path_out, rendered_frames, fps=fps_out)

    # ----- 7) Write per-frame jsonl -----
    if preds_path is not None:
        with open(preds_path, "w") as f:
            for row in preds_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[OUT] wrote {len(preds_records)} preds → {preds_path}")

    # ----- 8) Write summary -----
    summary: Dict[str, Any] = {
        "ckpt_path": ckpt_path,
        "view_list": view_list,
        "instruction": instruction,
        "video_paths": video_paths,
        "n_frames_processed": n_steps,
        "frame_stride": frame_stride,
        "fps_out": fps_out,
        "history": history,
        "n_waypoints": n_waypoints,
        "max_candidates": n_max,
        "cot_noexist_count": cot_noexist_count,
        "cot_noexist_rate": cot_noexist_count / max(1, n_steps),
        "catalog_empty_count": catalog_empty_count,
        "catalog_empty_rate": catalog_empty_count / max(1, n_steps),
        "elapsed_seconds": elapsed_all,
        "avg_track_ms": float(np.mean(track_times) * 1000) if track_times else 0.0,
        "avg_encode_ms": float(np.mean(encode_times) * 1000) if encode_times else 0.0,
        "avg_infer_ms": float(np.mean(inference_times) * 1000) if inference_times else 0.0,
        "video_path_out": video_path_out,
        "frames_dir": frames_dir,
        "preds_path": preds_path,
        "raw_video_path": raw_video_path,
    }
    if summary_path is not None:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OUT] wrote summary → {summary_path}")

    # ----- 9) Copy/transcode the raw forward video -----
    if raw_video_path is not None:
        try:
            _copy_or_transcode_video(forward_path, raw_video_path)
            print(f"[OUT] raw video → {raw_video_path}")
        except Exception as e:
            print(f"[WARN] raw video copy failed: {e}")

    print(
        f"[DONE] frames={n_steps}  no_exist={cot_noexist_count}({100 * cot_noexist_count / max(1, n_steps):.1f}%) "
        f"empty_cat={catalog_empty_count}({100 * catalog_empty_count / max(1, n_steps):.1f}%) "
        f"avg_infer={np.mean(inference_times) * 1000:.1f}ms"
    )

    for s in sources.values():
        s.close()
    return summary


def _copy_or_transcode_video(src: str, dst: str) -> None:
    """Image folder → mux to mp4 with ffmpeg; video file → ffmpeg transcode (H.264)."""
    if osp.isfile(src):
        ext = osp.splitext(src)[1].lower()
        if ext in VIDEO_EXTS:
            cmd = [
                "ffmpeg", "-y", "-i", src,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-crf", "23",
                dst,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            return
        if ext in IMAGE_EXTS:
            shutil.copy(src, osp.splitext(dst)[0] + ext)
            return
    if osp.isdir(src):
        # Find a sorted image sequence of one ext to feed ffmpeg
        for ext in (".jpg", ".jpeg", ".png"):
            files = sorted(Path(src).glob(f"*{ext}"))
            if files:
                pattern = osp.join(src, f"*{ext}")
                cmd = [
                    "ffmpeg", "-y", "-framerate", "8",
                    "-pattern_type", "glob", "-i", pattern,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-crf", "23",
                    dst,
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return
    raise FileNotFoundError(f"unsupported raw video source: {src}")


# =========================================================================
# CLI
# =========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ReferTrack on an arbitrary video + instruction (offline, no Habitat).",
    )
    p.add_argument("--ckpt-path", type=str, required=True,
                   help="Path to ckpt; model_config.json must sit in the same directory")
    p.add_argument("--video-forward", type=str, required=True,
                   help="Forward-view video file (mp4/avi/mov) or image folder")
    p.add_argument("--video-left", type=str, default=None,
                   help="Left-view video (required for multi-view ckpts)")
    p.add_argument("--video-right", type=str, default=None,
                   help="Right-view video (required for multi-view ckpts)")
    p.add_argument("--video-back", type=str, default=None,
                   help="Back-view video (required if ckpt view_list includes back)")

    p.add_argument("--instruction", type=str, required=True,
                   help="Natural-language instruction, e.g. 'Follow the person in the red jacket.'")

    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory for mp4 / frames / jsonl / summary")

    p.add_argument("--fps-out", type=int, default=8,
                   help="Playback fps of the output mp4 (default 8)")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Sample the source video at this stride (default 1=no skip; use 4 on 30fps video → ~8fps)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Max frames to run (0=entire clip)")

    p.add_argument("--save-frames", action="store_true",
                   help="Save per-frame jpgs to <out_dir>/<basename>_frames/")
    p.add_argument("--no-save-preds", action="store_true",
                   help="Do not save per-frame prediction jsonl")
    p.add_argument("--no-save-summary", action="store_true",
                   help="Do not save summary.json")
    p.add_argument("--copy-raw-video", action="store_true",
                   help="Transcode a copy of the raw forward video into the output dir for comparison")

    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    video_paths: Dict[str, str] = {"forward": args.video_forward}
    if args.video_left:
        video_paths["left"] = args.video_left
    if args.video_right:
        video_paths["right"] = args.video_right
    if args.video_back:
        video_paths["back"] = args.video_back

    run_video(
        ckpt_path=args.ckpt_path,
        video_paths=video_paths,
        instruction=args.instruction,
        out_dir=args.out_dir,
        fps_out=args.fps_out,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        save_frames=args.save_frames,
        save_preds_jsonl=(not args.no_save_preds),
        save_summary=(not args.no_save_summary),
        copy_raw_video=args.copy_raw_video,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
