#!/usr/bin/env python3
"""Collect EVT-Bench expert episodes for one chunk of a Track train split.

    python referTrack/tool/collect_expert.py \
        --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml \
        --split-num 30 --split-id 0 --save-path data/evt_bench/stt_singleview_train/seed_101 \
        habitat.simulator.seed=101

After collection, every episode of the chunk gets the instruction of the first episode
``TrackEnv`` visits, which is how the released checkpoint's training data was labeled.
"""
from __future__ import annotations

import argparse
import random

import habitat
import numpy as np
from habitat.datasets import make_dataset

from referTrack.baseline.expert_agent import chunk_instruction, collect_episodes, write_chunk_instruction
from referTrack.dataset import evt_bench  # noqa: F401  registers the Track task


def load_chunk(exp_config: str, opts, split_id: int, split_num: int):
    config = habitat.get_config(exp_config, opts)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)
    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    return config, dataset.get_splits(split_num, allow_uneven_splits=True)[split_id]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-config", required=True)
    ap.add_argument("--split-id", type=int, required=True)
    ap.add_argument("--split-num", type=int, default=30)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--skip-collect", action="store_true", help="Only rewrite the chunk instruction")
    ap.add_argument("opts", nargs=argparse.REMAINDER, help="Habitat config overrides")
    args = ap.parse_args()

    if not args.skip_collect:
        config, split = load_chunk(args.exp_config, args.opts, args.split_id, args.split_num)
        collect_episodes(config, split, args.save_path)

    # Re-seed exactly as a fresh process would, so TrackEnv starts from the same episode.
    config, split = load_chunk(args.exp_config, args.opts, args.split_id, args.split_num)
    instruction = chunk_instruction(config, split)
    if instruction is not None:
        n = write_chunk_instruction(split, args.save_path, instruction)
        print(f"[chunk {args.split_id}] instruction '{instruction}' -> {n} episodes")


if __name__ == "__main__":
    main()
