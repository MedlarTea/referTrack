#!/usr/bin/env python3
"""ReferTrack training (DeepSpeed ZeRO-2).

Every step draws one EVT-Bench navigation batch and one SYNTH-PEDES refer-QA batch:

    loss = beta_nav * L_nav + beta_cot * L_cot_nav + beta_cot_qa * L_cot_qa

``L_nav`` is the masked MSE on the 8 waypoints (xy + yaw), ``L_cot_*`` the catalog-slot
cross-entropy. ``proj / tvi / planner / act_token`` are warm-started from the stage-1
checkpoint (``--pretrained_ckpt``); the LLM starts from the HF weights.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import glob
import json
import math
import os
import os.path as osp
import random
import re
import time

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset
from transformers.optimization import get_cosine_schedule_with_warmup

from referTrack.dataset.refer_datasets import ReferNavDataset, ReferQADataset, collate
from referTrack.model.referTrack import ReferTrack, ReferTrackConfig


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local_rank", type=int, default=-1)
    # data
    ap.add_argument("--nav_roots", nargs="+", required=True, help="Processed EVT split roots")
    ap.add_argument("--nav_samples_per_split", type=int, default=333334,
                    help="Each split is resampled to this size, except --nav_keep_all_splits")
    ap.add_argument("--nav_keep_all_splits", nargs="*", default=["dt_singleview_train"])
    ap.add_argument("--refer_qa_root", required=True)
    ap.add_argument("--refer_qa_noexist_ratio", type=float, default=0.06)
    # model
    ap.add_argument("--llm_name", required=True)
    ap.add_argument("--pretrained_ckpt", default="")
    ap.add_argument("--pretrained_load_modules", default="proj,tvi,planner,act_token")
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--n_waypoints", type=int, default=8)
    ap.add_argument("--alpha_xy", type=float, default=0.535)
    ap.add_argument("--alpha_yaw", type=float, default=1.572)
    ap.add_argument("--new_token_init", default="mean", choices=["mean", "zero", "random"])
    ap.add_argument("--gradient_checkpointing", action="store_true")
    # loss
    ap.add_argument("--beta_nav", type=float, default=10.0)
    ap.add_argument("--beta_cot", type=float, default=1.0)
    ap.add_argument("--beta_cot_qa", type=float, default=1.0)
    # optim
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--qa_batch_size", type=int, default=16)
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=2e-4)
    ap.add_argument("--wd_backbone", type=float, default=0.01)
    ap.add_argument("--wd_head", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.02)
    ap.add_argument("--grad_accumulation_steps", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0, help="Stop after this many steps (0 = full schedule)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    # io
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--max_ckpts", type=int, default=6)
    ap.add_argument("--resume", action="store_true", help="Resume from the latest out_dir/checkpoints/ds_step*")
    return ap.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=1800), device_id=device)
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), device


def build_nav_dataset(args, rank: int):
    parts = []
    for root in args.nav_roots:
        ds = ReferNavDataset(root, history=args.history, n_waypoints=args.n_waypoints,
                             alpha_xy=args.alpha_xy, alpha_yaw=args.alpha_yaw)
        n, target = len(ds), args.nav_samples_per_split
        if osp.basename(osp.normpath(root)) in args.nav_keep_all_splits:
            idx = list(range(n))
        elif n <= target:
            idx = np.random.choice(n, size=target, replace=True).tolist()
        else:
            idx = [int(i * n / target) for i in range(target)]
        parts.append(Subset(ds, idx))
        if rank == 0:
            print(f"[DATA] nav {root}: {n} -> {len(idx)} samples")
    return ConcatDataset(parts)


def build_model(args, device, rank: int) -> ReferTrack:
    cfg = ReferTrackConfig(
        llm_name=args.llm_name, view_list=["forward"], n_waypoints=args.n_waypoints, history=args.history,
        beta_nav=args.beta_nav, beta_cot=args.beta_cot, beta_cot_qa=args.beta_cot_qa,
        use_angle_tvi=False, alpha_xy=args.alpha_xy, alpha_yaw=args.alpha_yaw,
        gradient_checkpointing=args.gradient_checkpointing, new_token_init=args.new_token_init,
    )
    model = ReferTrack(cfg).to(device)
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        saved = dataclasses.replace(cfg, llm_name=osp.basename(osp.normpath(cfg.llm_name)), gradient_checkpointing=False)
        with open(osp.join(args.out_dir, "model_config.json"), "w") as f:
            json.dump(dataclasses.asdict(saved), f, indent=2)

    if args.pretrained_ckpt:
        sd = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=True)
        sd = sd.get("model_state", sd)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        modules = {m.strip() for m in args.pretrained_load_modules.split(",") if m.strip()}
        sd = {k: v for k, v in sd.items() if not modules or k.split(".", 1)[0] in modules}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f"[WARM-START] {args.pretrained_ckpt}: loaded {len(sd)} tensors "
                  f"(modules={sorted(modules)}), unexpected={len(unexpected)}")
    return model


def save_weights(model_engine, args, epoch: int, step: int, rank: int):
    """DeepSpeed state for --resume, plus a plain ``model_state`` .pt for eval."""
    ckpt_dir = osp.join(args.out_dir, "checkpoints")
    model_engine.save_checkpoint(ckpt_dir, tag=f"ds_step{step:07d}", client_state={"epoch": epoch, "step": step})
    if rank != 0:
        return
    torch.save({"epoch": epoch, "step": step, "model_state": model_engine.module.state_dict()},
               osp.join(args.out_dir, f"model_epoch{epoch:02d}_step{step:06d}.pt"))
    step_of = lambda p: int(re.search(r"step(\d+)", p).group(1))
    for pattern in (osp.join(ckpt_dir, "ds_step*"), osp.join(args.out_dir, "model_epoch*_step*.pt")):
        for old in sorted(glob.glob(pattern), key=step_of)[:-args.max_ckpts]:
            os.system(f"rm -rf '{old}'")
    print(f"[CKPT] saved step {step}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    rank, world_size, device = setup_distributed()
    if rank == 0:
        print(json.dumps(vars(args), indent=2))

    nav_ds = build_nav_dataset(args, rank)
    qa_ds = ReferQADataset(args.refer_qa_root, history=args.history,
                           noexist_ratio=args.refer_qa_noexist_ratio, split="train", seed=args.seed)
    if rank == 0:
        print(f"[DATA] nav={len(nav_ds)} refer_qa={len(qa_ds)}")

    model = build_model(args, device, rank)
    col = functools.partial(collate, tokenizer=model.tokenizer)
    dl_kw = dict(num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=col)
    if args.num_workers > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    nav_sampler = DistributedSampler(nav_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    qa_sampler = DistributedSampler(qa_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    nav_loader = DataLoader(nav_ds, batch_size=args.batch_size, sampler=nav_sampler, **dl_kw)
    qa_loader = DataLoader(qa_ds, batch_size=args.qa_batch_size, sampler=qa_sampler, **dl_kw)

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in params if n.startswith("llm.")], "lr": args.lr_backbone, "weight_decay": args.wd_backbone},
        {"params": [p for n, p in params if not n.startswith("llm.")], "lr": args.lr_head, "weight_decay": args.wd_head},
    ])
    steps_per_epoch = len(nav_loader)
    total_steps = args.epochs * math.ceil(steps_per_epoch / args.grad_accumulation_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(args.warmup_ratio * total_steps)), total_steps)
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "gradient_clipping": args.grad_clip,
        "steps_per_print": 1000,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 2, "overlap_comm": True, "contiguous_gradients": True, "reduce_scatter": True,
            "reduce_bucket_size": 5e7, "allgather_bucket_size": 5e7,
        },
        "bf16": {"enabled": True},
    }
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler, config=ds_config,
    )

    start_epoch, global_step = 0, 0
    ckpt_dir = osp.join(args.out_dir, "checkpoints")
    if args.resume and glob.glob(osp.join(ckpt_dir, "ds_step*")):
        _, client = engine.load_checkpoint(ckpt_dir)
        start_epoch, global_step = client["epoch"], client["step"]
        if rank == 0:
            print(f"[RESUME] epoch={start_epoch} step={global_step}")

    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(osp.join(args.out_dir, "tb"))

    qa_iter = None
    t0 = time.time()
    last_saved = global_step
    for epoch in range(start_epoch, args.epochs):
        engine.train()
        nav_sampler.set_epoch(epoch)
        qa_sampler.set_epoch(epoch)
        qa_iter = iter(qa_loader)
        nav_iter = iter(nav_loader)
        skip = global_step % steps_per_epoch if epoch == start_epoch else 0
        for _ in range(skip):
            next(nav_iter)

        for _ in range(skip, steps_per_epoch):
            nav = {k: v.to(device) if torch.is_tensor(v) else v for k, v in next(nav_iter).items()}
            try:
                qa = next(qa_iter)
            except StopIteration:
                qa_iter = iter(qa_loader)
                qa = next(qa_iter)
            qa = {k: v.to(device) if torch.is_tensor(v) else v for k, v in qa.items()}

            tau, loss_cot, _, _ = engine.forward_refer_navigation(
                coarse_tokens=nav["coarse_tokens"], coarse_tidx=nav["coarse_tidx"],
                fine_tokens=nav["fine_tokens"], fine_tidx=nav["fine_tidx"],
                cand_bbox=nav["cand_bbox"], cand_slot_valid=nav["cand_slot_valid"],
                target_slot=nav["target_slot"], bbox_hist=nav["bbox_hist"],
                bbox_hist_valid=nav["bbox_hist_valid"],
                instruction_input_ids=nav["instruction_input_ids"],
                instruction_attention_mask=nav["instruction_attention_mask"],
                alpha=nav["alpha"],
            )
            diff = (tau.float() - nav["waypoints"].float()) ** 2
            mask = nav["valid_mask"].float().unsqueeze(-1)
            n_valid = mask.sum().clamp_min(1e-6)
            loss_nav = (diff[..., :2] * mask).sum() / n_valid + (diff[..., 2:] * mask).sum() / n_valid
            loss_qa, _, _ = engine.forward_refer_qa(
                coarse_tokens=qa["coarse_tokens"], coarse_tidx=qa["coarse_tidx"],
                fine_tokens=qa["fine_tokens"], fine_tidx=qa["fine_tidx"],
                cand_bbox=qa["cand_bbox"], cand_slot_valid=qa["cand_slot_valid"],
                target_slot=qa["target_slot"],
                instruction_input_ids=qa["instruction_input_ids"],
                instruction_attention_mask=qa["instruction_attention_mask"],
            )
            loss = args.beta_nav * loss_nav + args.beta_cot * loss_cot + args.beta_cot_qa * loss_qa
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step + 1}: nav={loss_nav} cot={loss_cot} qa={loss_qa}")
            engine.backward(loss)
            engine.step()
            global_step += 1

            if rank == 0 and global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                gn = engine.get_global_grad_norm() or 0.0
                vals = {"loss": loss.item(), "L_nav": loss_nav.item(), "L_cot": loss_cot.item(),
                        "L_qa": loss_qa.item(), "lr": lr, "grad_norm": float(gn)}
                for k, v in vals.items():
                    writer.add_scalar(f"Train/{k}", v, global_step)
                print(f"epoch {epoch} step {global_step}/{args.epochs * steps_per_epoch} "
                      + " ".join(f"{k}={v:.4g}" for k, v in vals.items())
                      + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
            if global_step % args.save_every == 0:
                save_weights(engine, args, epoch, global_step, rank)
                last_saved = global_step
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    if global_step != last_saved:
        save_weights(engine, args, epoch, global_step, rank)
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
