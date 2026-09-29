#!/usr/bin/env python3
"""Pre-compute DINOv3 + SigLIP grid-pooled tokens for every image under ``<data_root>/<image_dir>``.

``<data_root>/frames/.../forward/frame_00001.jpg`` →
``<data_root>/vision_cache/frames/.../forward/frame_00001_{vcoarse,vfine}.pt``
(fp16; vcoarse = 2×2 = 4 tokens, vfine = 8×8 = 64 tokens). Existing files are skipped.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from referTrack.eval.cache_gridpool import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens


class _Frames(Dataset):
    def __init__(self, tasks):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, i):
        img, vc, vf = self.tasks[i]
        return Image.open(img).convert("RGB"), vc, vf


@torch.inference_mode()
def encode_batch(pils, enc: VisionFeatureCacher):
    tok_dino, Hp, Wp = enc._encode_dino(pils)
    tok_sigl = enc._encode_siglip(pils, out_hw=(Hp, Wp))
    tok = torch.cat([tok_dino, tok_sigl], dim=-1)
    return grid_pool_tokens(tok, Hp, Wp, out_tokens=4).cpu(), grid_pool_tokens(tok, Hp, Wp, out_tokens=64).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, help="Processed split root containing frames/")
    ap.add_argument("--cache_root", default=None, help="Default: <data_root>/vision_cache")
    ap.add_argument("--image_dir", default="frames", help="Image folder under data_root (refer-QA: images)")
    ap.add_argument("--view", default="forward", help="Only leaf folders with this name; '' = all")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=0, help="Shard index; frames[rank::world_size]")
    ap.add_argument("--world_size", type=int, default=1)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve() if args.cache_root else data_root / "vision_cache"

    tasks = []
    for dirpath, _, files in os.walk(data_root / args.image_dir):
        if args.view and Path(dirpath).name != args.view:
            continue
        out_dir = cache_root / Path(dirpath).relative_to(data_root)
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            vc, vf = out_dir / f"{stem}_vcoarse.pt", out_dir / f"{stem}_vfine.pt"
            if not (vc.exists() and vf.exists()):
                tasks.append((os.path.join(dirpath, name), str(vc), str(vf)))
    tasks = sorted(tasks)[args.rank::args.world_size]
    print(f"[rank {args.rank}] {len(tasks)} frames to encode")
    if not tasks:
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    enc = VisionFeatureCacher(VisionCacheConfig(image_size=384, batch_size=args.batch_size, device=device)).eval()
    loader = DataLoader(
        _Frames(tasks), batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=lambda b: tuple(zip(*b)),
    )

    def save(t: torch.Tensor, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(t.half(), path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for pils, vc_paths, vf_paths in tqdm(loader, desc=f"rank {args.rank}"):
            vc, vf = encode_batch(list(pils), enc)
            for j in range(len(pils)):
                pool.submit(save, vc[j].clone(), vc_paths[j])
                pool.submit(save, vf[j].clone(), vf_paths[j])


if __name__ == "__main__":
    main()
