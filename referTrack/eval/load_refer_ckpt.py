"""Load a released ReferTrack checkpoint for Habitat / video eval."""
from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Dict, Tuple

import torch

from referTrack.constants import ROOT_PATH
from referTrack.model.referTrack import ReferTrack, ReferTrackConfig


def find_config_dir(ckpt_path: str) -> str:
    ckpt_dir = osp.dirname(osp.abspath(ckpt_path))
    for cand_dir in (ckpt_dir, osp.dirname(ckpt_dir)):
        if osp.isfile(osp.join(cand_dir, "model_config.json")):
            return cand_dir
    raise FileNotFoundError(
        f"model_config.json not found next to {ckpt_path}. "
        "Place model_config.json in the same directory as the .pt file."
    )


def _load_json(path: str) -> Dict[str, Any]:
    if not osp.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_refer_model_config(path: str) -> ReferTrackConfig:
    data = _load_json(path)
    if not data:
        raise FileNotFoundError(f"model_config.json not found: {path}")
    allowed = {f.name for f in ReferTrackConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return ReferTrackConfig(**{k: v for k, v in data.items() if k in allowed})


def resolve_llm_path(llm_name: str) -> str:
    if llm_name and osp.isdir(llm_name):
        return llm_name
    here = osp.dirname(osp.abspath(__file__))
    project_root = osp.abspath(osp.join(here, "..", ".."))
    candidate_roots = [ROOT_PATH, os.getcwd(), project_root]
    seen = set()
    for root in candidate_roots:
        if not root or root in seen:
            continue
        seen.add(root)
        for name in (llm_name, osp.basename(llm_name), osp.basename(llm_name).lower()):
            if not name:
                continue
            p = osp.join(root, "LLM_hf", name)
            if osp.isdir(p):
                return p
    return llm_name


def _strip_ddp_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    first = next(iter(sd.keys()))
    if first.startswith("module."):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def align_vocab_rows_for_legacy_ckpt(
    model: torch.nn.Module, msd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Pad/crop embed_tokens and lm_head when special-token rows differ."""
    model_sd = model.state_dict()
    out = dict(msd)
    patched = []
    for k, src in msd.items():
        if k not in model_sd:
            continue
        tgt = model_sd[k]
        if not isinstance(src, torch.Tensor) or not isinstance(tgt, torch.Tensor):
            continue
        if tuple(src.shape) == tuple(tgt.shape):
            continue
        is_vocab_matrix = (
            src.ndim == 2
            and tgt.ndim == 2
            and src.shape[1] == tgt.shape[1]
            and (k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight"))
        )
        if not is_vocab_matrix:
            continue
        new_w = src.new_zeros(tgt.shape)
        rows = min(src.shape[0], tgt.shape[0])
        new_w[:rows] = src[:rows]
        if tgt.shape[0] > src.shape[0]:
            base = src[:rows]
            mean = base.mean(dim=0)
            std = float(base.std().item()) if rows > 1 else 0.02
            if std <= 0:
                std = 0.02
            extra = tgt.shape[0] - src.shape[0]
            noise = torch.randn((extra, tgt.shape[1]), device=src.device, dtype=src.dtype) * (std * 0.01)
            new_w[src.shape[0] :] = mean.unsqueeze(0) + noise
        out[k] = new_w
        patched.append((k, tuple(src.shape), tuple(tgt.shape)))
    if patched:
        print("[CKPT-COMPAT] patched vocab rows:")
        for k, s0, s1 in patched:
            print(f"  - {k}: {s0} -> {s1}")
    return out


def load_refer_model(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[ReferTrack, ReferTrackConfig]:
    cfg_dir = find_config_dir(ckpt_path)
    model_cfg = load_refer_model_config(osp.join(cfg_dir, "model_config.json"))

    model_cfg.llm_name = resolve_llm_path(model_cfg.llm_name)

    print(f"[LOAD] llm={model_cfg.llm_name}  max_candidates={model_cfg.max_candidates}")
    print(
        f"[LOAD] view_list={model_cfg.view_list}  "
        f"vision_feat_dim={model_cfg.vision_feat_dim}  history={model_cfg.history}  "
        f"alpha_xy={model_cfg.alpha_xy}"
    )
    if not osp.isdir(model_cfg.llm_name):
        print(
            "[LOAD] WARNING: LLM path does not exist. "
            "Run `python LLM_hf/download_llm_hf.py` or set model_config.json llm_name."
        )

    model = ReferTrack(model_cfg).to(device).eval()

    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = obj.get("model_state") or obj.get("model_state_dict") or obj
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"ckpt {ckpt_path} has no model_state(_dict).")
    state_dict = _strip_ddp_prefix(state_dict)
    state_dict.pop("alpha_task", None)  # scale from model_config.json, not the ckpt buffer
    state_dict = align_vocab_rows_for_legacy_ckpt(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if k != "alpha_task"]
    print(f"[LOAD] loaded {len(state_dict)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"  missing sample: {missing[:5]}")
    if unexpected[:5]:
        print(f"  unexpected sample: {unexpected[:5]}")
    return model, model_cfg
