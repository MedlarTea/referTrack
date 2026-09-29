"""Training datasets for ReferTrack.

* ``ReferNavDataset``: EVT-Bench ``*_withTrack.jsonl`` from ``tool/build_refer_jsonl.py``.
* ``ReferQADataset``: SYNTH-PEDES refer-QA (``info.json`` + ``val_indices.json``).

Both read cached tokens from ``vision_cache/`` (``tool/precache_features.py``) and emit
the same CoT inputs: catalog ``cand_bbox (CAT_LEN, 4)`` whose last slot is the virtual
NO_EXIST slot, ``target_slot`` (``max_candidates`` for NO_EXIST), 4-token coarse frames
and one 64-token fine current frame.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from referTrack.constants import MAX_CANDIDATES

BBOX_NORM = 384.0


def load_tokens(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).float()


def token_path(cache_root: Path, img_rel: str, variant: str) -> Path:
    p = Path(img_rel)
    return cache_root / p.parent / f"{p.stem}_{variant}.pt"


def integrate_actions(actions: np.ndarray, n_waypoints: int, dt: float) -> np.ndarray:
    """Body-frame ``[vx, vy, wz]`` → ``n_waypoints`` local ``[x, y, yaw]`` (includes t=0)."""
    a = np.asarray(actions, dtype=np.float32)
    T = a.shape[0]
    x, y, th = (np.zeros(T, dtype=np.float32) for _ in range(3))
    for t in range(1, T):
        th[t] = th[t - 1] + a[t - 1, 2] * dt
        c, s = np.cos(th[t - 1]), np.sin(th[t - 1])
        x[t] = x[t - 1] + (c * a[t - 1, 0] - s * a[t - 1, 1]) * dt
        y[t] = y[t - 1] + (s * a[t - 1, 0] + c * a[t - 1, 1]) * dt
    idx = np.linspace(0, T - 1, n_waypoints).round().astype(int)
    return np.stack([x, y, th], axis=-1)[idx]


def _stable_seed(idx: int, seed: int = 0) -> int:
    return int(((idx * 2654435761) ^ (seed * 40503)) & 0xFFFFFFFF)


def _area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


class ReferNavDataset(Dataset):
    """One EVT split root: ``<root>/jsonl/**/*_withTrack.jsonl``, ``<root>/frames``,
    ``<root>/vision_cache``. Samples whose target is on another floor
    (``|human_y - robot_y| > 0.3``) are dropped at index time."""

    def __init__(
        self,
        root: str,
        history: int = 31,
        n_waypoints: int = 8,
        max_candidates: int = MAX_CANDIDATES,
        alpha_xy: float = 0.535,
        alpha_yaw: float = 1.572,
        dt: float = 0.1,
    ):
        self.root = Path(root)
        self.cache_root = self.root / "vision_cache"
        self.history = history
        self.n_waypoints = n_waypoints
        self.max_candidates = max_candidates
        self.alpha = torch.tensor([alpha_xy, alpha_xy, alpha_yaw], dtype=torch.float32)
        self.dt = dt
        self.index = self._build_index(sorted((self.root / "jsonl").rglob("*_withTrack.jsonl")))
        if not self.index:
            raise RuntimeError(f"no samples under {self.root / 'jsonl'}")

    def _build_index(self, files: List[Path]) -> List[Tuple[str, int]]:
        """``(file, byte_offset)`` per sample; cached next to the jsonl, keyed on path/size/mtime."""
        h = hashlib.md5()
        for fp in files:
            st = fp.stat()
            h.update(f"{fp}|{int(st.st_mtime)}|{st.st_size}".encode())
        cache = self.root / "jsonl" / f".index-{h.hexdigest()[:16]}.pkl"
        if cache.is_file():
            with open(cache, "rb") as f:
                return pickle.load(f)

        index: List[Tuple[str, int]] = []
        for fp in files:
            with open(fp, "rb") as f:
                pos = 0
                for line in f:
                    if line.strip():
                        try:
                            ex = json.loads(line)
                            hp, rp = ex.get("human_pos"), ex.get("robot_pos")
                            keep = hp is None or rp is None or abs(hp[1] - rp[1]) <= 0.3
                        except Exception:
                            keep = True
                        if keep:
                            index.append((str(fp), pos))
                    pos += len(line)
        if int(os.environ.get("RANK", "0")) == 0:
            tmp = cache.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache)
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fp, off = self.index[idx]
        with open(fp, "rb") as f:
            f.seek(off)
            ex = json.loads(f.readline())
        H, N = self.history, self.max_candidates

        fine = load_tokens(token_path(self.cache_root, ex["current"]["forward"], "vfine"))
        hist = ex["images"]["forward"]
        trim = min(H, len(hist))
        start, missing = len(hist) - trim, H - trim
        if trim:
            frames = [load_tokens(token_path(self.cache_root, p, "vcoarse")) for p in hist[start:]]
            frames = [frames[0]] * missing + frames
        else:
            frames = [load_tokens(token_path(self.cache_root, ex["current"]["forward"], "vcoarse"))] * H
        coarse = torch.cat(frames, dim=0)
        coarse_tidx = torch.arange(H).repeat_interleave(frames[0].size(0))

        wp = torch.from_numpy(integrate_actions(ex["actions"], self.n_waypoints, float(ex.get("dt", self.dt))))

        # Catalog: area top-N tracker boxes in a per-sample fixed random order.
        curr_tgt = ex["current_track_target_bbox"]["forward"]
        target_tid = int(curr_tgt[0])
        tracks = [(int(r[0]), [float(v) for v in r[1:5]]) for r in ex["track_bboxes"]["forward"] if len(r) == 5]
        if len(tracks) > N:
            tracks = sorted(tracks, key=lambda it: _area(it[1]), reverse=True)[:N]
        rng = np.random.default_rng(seed=_stable_seed(idx))
        if tracks:
            tracks = [tracks[i] for i in rng.permutation(len(tracks))]
        slot = next((k for k, (tid, _) in enumerate(tracks) if tid == target_tid), -1) if target_tid >= 0 else -1
        is_no_exist = slot < 0

        cand_bbox = torch.zeros(N + 1, 4)
        cand_valid = torch.zeros(N + 1, dtype=torch.bool)
        for k, (_, b) in enumerate(tracks):
            cand_bbox[k] = torch.tensor(b) / BBOX_NORM
            cand_valid[k] = True
        cand_valid[N] = True

        # Target box history from matched tracks; unmatched frames stay 0 / invalid,
        # left padding repeats the earliest matched frame.
        bbox_hist = torch.zeros(H, 4)
        bbox_hist_valid = torch.zeros(H, dtype=torch.bool)
        raw = ex["track_target_bbox_history"]["forward"]
        for t in range(missing, H):
            i = start + t - missing
            if i < len(raw) and int(raw[i][0]) >= 0:
                bbox_hist[t] = torch.tensor(raw[i][1:5], dtype=torch.float32) / BBOX_NORM
                bbox_hist_valid[t] = True
        first = next((t for t in range(missing, H) if bbox_hist_valid[t]), None)
        if first is not None and missing > 0:
            bbox_hist[:missing] = bbox_hist[first]
            bbox_hist_valid[:missing] = True

        return {
            "coarse_tokens": coarse,
            "coarse_tidx": coarse_tidx,
            "fine_tokens": fine,
            "fine_tidx": torch.full((fine.size(0),), H, dtype=torch.long),
            "waypoints": wp,
            "valid_mask": torch.ones(self.n_waypoints, dtype=torch.bool),
            "bbox_hist": bbox_hist,
            "bbox_hist_valid": bbox_hist_valid,
            "cand_bbox": cand_bbox,
            "cand_slot_valid": cand_valid,
            "target_slot": torch.tensor(N if is_no_exist else slot),
            "is_no_exist": torch.tensor(is_no_exist),
            "alpha": self.alpha,
            "instruction": ex.get("instruction", "follow the person"),
        }


class ReferQADataset(Dataset):
    """``info.json`` records ``{file_path, tracks: {tid: {bbox, caption}}}``; ``tracks['-1']``
    holds a caption of a person not in the image (NO_EXIST query). Records listed in
    ``val_indices.json`` are held out. The static image becomes ``history + 1`` copies of
    its coarse tokens plus one fine frame, matching the navigation input layout."""

    def __init__(
        self,
        root: str,
        history: int = 31,
        max_candidates: int = MAX_CANDIDATES,
        noexist_ratio: float = 0.06,
        split: str = "train",
        seed: int = 0,
    ):
        self.root = Path(root)
        self.cache_root = self.root / "vision_cache"
        self.history = history
        self.max_candidates = max_candidates
        self.noexist_ratio = noexist_ratio
        self.seed = seed
        records = json.loads((self.root / "info.json").read_text())
        val = json.loads((self.root / "val_indices.json").read_text())
        val = set(val.get("val_paths") if isinstance(val, dict) else val)
        self.records = [r for r in records if (r.get("file_path") in val) == (split == "val")]

    def __len__(self) -> int:
        return len(self.records)

    def _no_exist_caption(self, tracks: Dict[str, Any], rng: np.random.Generator) -> Optional[str]:
        cap = (tracks.get("-1") or {}).get("caption")
        if isinstance(cap, str) and cap:
            return cap
        for _ in range(32):
            cap = (self.records[int(rng.integers(0, len(self.records)))].get("tracks", {}).get("-1") or {}).get("caption")
            if isinstance(cap, str) and cap:
                return cap
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        tracks = rec.get("tracks", {})
        N = self.max_candidates

        pool = []
        for tid, item in tracks.items():
            try:
                if int(tid) < 0:
                    continue
            except ValueError:
                continue
            b = item.get("bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4 and any(float(v) != 0.0 for v in b):
                pool.append(([float(v) for v in b], str(item.get("caption", ""))))
        if len(pool) > N:
            pool = sorted(pool, key=lambda it: _area(it[0]), reverse=True)[:N]
        K = len(pool)

        rng = np.random.default_rng(seed=_stable_seed(idx, self.seed))
        is_no_exist = bool(rng.random() < self.noexist_ratio) or K == 0
        if K:
            pool = [pool[i] for i in rng.permutation(K)]
        no_exist_cap = self._no_exist_caption(tracks, rng)
        if is_no_exist:
            target_slot = N
            caption = no_exist_cap if no_exist_cap is not None else (pool[0][1] if K else "")
        else:
            target_slot = int(rng.integers(0, K))
            caption = pool[target_slot][1]

        cand_bbox = torch.zeros(N + 1, 4)
        cand_valid = torch.zeros(N + 1, dtype=torch.bool)
        for k, (b, _) in enumerate(pool):
            cand_bbox[k] = torch.tensor(b) / BBOX_NORM
            cand_valid[k] = True
        cand_valid[N] = True

        vc = load_tokens(token_path(self.cache_root, rec["file_path"], "vcoarse"))
        vf = load_tokens(token_path(self.cache_root, rec["file_path"], "vfine"))
        T = self.history + 1
        return {
            "coarse_tokens": vc.repeat(T, 1),
            "coarse_tidx": torch.arange(T).repeat_interleave(vc.size(0)),
            "fine_tokens": vf,
            "fine_tidx": torch.full((vf.size(0),), self.history, dtype=torch.long),
            "cand_bbox": cand_bbox,
            "cand_slot_valid": cand_valid,
            "target_slot": torch.tensor(target_slot),
            "is_no_exist": torch.tensor(is_no_exist),
            "instruction": f"Please find <{caption}> in the video. Answer with object indexes.",
        }


def collate(batch: List[Dict[str, Any]], tokenizer=None, max_length: int = 512) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "instruction"}
    out["instructions"] = [b["instruction"] for b in batch]
    if tokenizer is not None:
        tok = tokenizer(out["instructions"], return_tensors="pt", padding="longest", truncation=True, max_length=max_length)
        out["instruction_input_ids"] = tok["input_ids"]
        out["instruction_attention_mask"] = tok["attention_mask"]
    return out
