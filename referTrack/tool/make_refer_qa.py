#!/usr/bin/env python3
"""Synthesize the refer-QA set: paste 2-3 SYNTH-PEDES people onto a background image.

Inputs (``--data_root``):
    synthpedes-dataset.json, Part*/ (SYNTH-PEDES, https://github.com/Zplusdragon/PLIP)
    backgrounds/**.jpg          your own background images (not distributed)

Outputs (``--output_dir``):
    images/%07d.jpg             384x384 composites
    info.json                   [{file_path, tracks: {tid: {bbox, caption, file_path}}}],
                                tid "-1" = caption of a person that is *not* in the image
    val_indices.json            1024 held-out file paths (stride over sorted paths)

SYNTH-PEDES is for non-commercial research only; so is anything generated from it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO

from PIL import Image
from tqdm import tqdm

BG_CROP_BOTTOM = 150            # drop watermarks at the bottom of backgrounds
BG_SIZE = (384, 384)
PERSON_BASE_SIZE = (128, 256)   # (w, h) before scaling
SCALE_MIN, SCALE_MAX = 0.75, 1.5
MIN_PERSONS, MAX_PERSONS = 2, 3
MAX_PLACEMENT_RETRIES = 50
MAX_LAYOUT_RETRIES = 10
ID_MIN, ID_MAX = 0, 100
BATCH_SIZE = 10_000
VAL_N = 1024


class ReferQAMaker:
    def __init__(self, data_root: str, output_dir: str, num_workers: int):
        self.data_root = data_root
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        with open(os.path.join(data_root, "synthpedes-dataset.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.persons = []
        for item in data:
            caption = "a person"
            if item.get("captions"):
                caption = item["captions"][0]
            elif item.get("prompt_caption"):
                caption = item["prompt_caption"][0]
            self.persons.append({
                "person_id": item["id"],
                "file_path": os.path.join(data_root, item["file_path"]),
                "caption": caption,
            })
        self.backgrounds = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(os.path.join(data_root, "backgrounds"))
            for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        print(f"{len(self.persons)} person images, {len(self.backgrounds)} backgrounds")

    @lru_cache(maxsize=64)
    def _background(self, path: str) -> Image.Image:
        bg = Image.open(path).convert("RGB")
        if BG_CROP_BOTTOM > 0:
            w, h = bg.size
            bg = bg.crop((0, 0, w, h - BG_CROP_BOTTOM))
        return bg.resize(BG_SIZE, Image.LANCZOS)

    @staticmethod
    def _overlap(a, b) -> bool:
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    def _try_layout(self, bg_template, person_indices, assigned_ids, rng):
        """Non-overlapping random placement of every person, or None."""
        bg = bg_template.copy()
        bg_w, bg_h = bg.size
        base_w, base_h = PERSON_BASE_SIZE
        scale_max = min(SCALE_MAX, 1.0) if len(person_indices) >= 3 else SCALE_MAX
        boxes, tracks = [], {}
        for i, pidx in enumerate(person_indices):
            item = self.persons[pidx]
            person_img = Image.open(item["file_path"]).convert("RGB")
            for _ in range(MAX_PLACEMENT_RETRIES):
                scale = rng.uniform(SCALE_MIN, scale_max)
                pw, ph = int(base_w * scale), int(base_h * scale)
                if max(pw, ph) < 192 or pw > bg_w or ph > bg_h:
                    continue
                x, y = rng.randint(0, bg_w - pw), rng.randint(0, bg_h - ph)
                box = [x, y, x + pw, y + ph]
                if not any(self._overlap(box, b) for b in boxes):
                    bg.paste(person_img.resize((pw, ph), Image.LANCZOS), (x, y))
                    boxes.append(box)
                    tracks[str(assigned_ids[i])] = {
                        "bbox": box,
                        "caption": item["caption"],
                        "file_path": os.path.relpath(item["file_path"], self.data_root),
                    }
                    break
            else:
                return None
        return bg, tracks

    def _render(self, task):
        rng = random.Random(task["seed"])
        try:
            bg_template = self._background(task["bg_path"])
            for _ in range(MAX_LAYOUT_RETRIES):
                result = self._try_layout(bg_template, task["person_indices"], task["assigned_ids"], rng)
                if result is None:
                    continue
                img, tracks = result
                if task["neg_person_idx"] is not None:
                    neg = self.persons[task["neg_person_idx"]]
                    tracks["-1"] = {
                        "bbox": [0, 0, 0, 0],
                        "caption": neg["caption"],
                        "file_path": os.path.relpath(neg["file_path"], self.data_root),
                    }
                buf = BytesIO()
                img.save(buf, "JPEG", quality=95)
                return tracks, buf.getvalue()
        except Exception:
            pass
        return None

    def run(self, total_images: int, seed: int):
        rng = random.Random(seed)
        order = list(range(len(self.persons)))
        rng.shuffle(order)
        ptr = 0

        def next_person():
            nonlocal ptr
            if ptr >= len(order):
                rng.shuffle(order)
                ptr = 0
            ptr += 1
            return order[ptr - 1]

        def make_task():
            """2-3 distinct identities plus one absent identity for the NO_EXIST query."""
            n = rng.randint(MIN_PERSONS, MAX_PERSONS)
            selected, pids = [], set()
            for _ in range(n * 20):
                idx = next_person()
                if self.persons[idx]["person_id"] not in pids:
                    selected.append(idx)
                    pids.add(self.persons[idx]["person_id"])
                if len(selected) == n:
                    break
            if len(selected) < MIN_PERSONS:
                return None
            neg = None
            for _ in range(50):
                cand = rng.randint(0, len(self.persons) - 1)
                if self.persons[cand]["person_id"] not in pids:
                    neg = cand
                    break
            return {
                "bg_path": rng.choice(self.backgrounds),
                "person_indices": selected,
                "assigned_ids": rng.sample(range(ID_MIN, ID_MAX + 1), len(selected)),
                "neg_person_idx": neg,
                "seed": rng.randint(0, 2 ** 31),
            }

        records = []
        pbar = tqdm(total=total_images)
        while len(records) < total_images:
            tasks = []
            while len(tasks) < min(BATCH_SIZE, total_images - len(records)):
                task = make_task()
                if task is not None:
                    tasks.append(task)
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                results = [r for r in pool.map(self._render, tasks) if r is not None]
            for tracks, jpeg in results[: total_images - len(records)]:
                name = f"{len(records):07d}.jpg"
                with open(os.path.join(self.images_dir, name), "wb") as f:
                    f.write(jpeg)
                records.append({"file_path": f"images/{name}", "tracks": tracks})
                pbar.update(1)
        pbar.close()

        with open(os.path.join(self.output_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        paths = sorted(r["file_path"] for r in records)
        val = paths if len(paths) <= VAL_N else [paths[i * len(paths) // VAL_N] for i in range(VAL_N)]
        with open(os.path.join(self.output_dir, "val_indices.json"), "w") as f:
            json.dump({"val_paths": val}, f)
        print(f"wrote {len(records)} images, {len(val)} held out for val -> {self.output_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, help="SYNTH-PEDES root")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--total_images", type=int, default=1_300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=16)
    args = ap.parse_args()
    ReferQAMaker(args.data_root, args.output_dir, args.num_workers).run(args.total_images, args.seed)


if __name__ == "__main__":
    main()
