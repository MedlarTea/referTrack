"""ReferTrack Habitat simulation eval.

Loads the released single-view (`view_list=['forward']`) ReferTrack checkpoint
and runs closed-loop Track eval with `ReferAgent` (online YOLO + ByteTrack).

Usage:
  python referTrack/eval/run_eval_refer_sim.py \
    --run-type eval \
    --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_infer_dt.yaml \
    --split-id 0 \
    --split-num 8 \
    --save-path ./results/refer_eval \
    --ckpt-path /path/to/refertrack_qwen3_4b.pt
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random

# ============================================================
# Determinism (must run before torch / Habitat / any CUDA op)
#   - PYTHONHASHSEED          : stable hash((ep_id, step)) for catalog shuffle
#   - CUBLAS_WORKSPACE_CONFIG : required by torch.use_deterministic_algorithms
#   - cuDNN deterministic     : pin conv algorithm selection
#   - torch.manual_seed       : fixed seed for LLM / YOLO RNGs
# Pair with the exports at the top of eval_sim_refer_base.sh.
# ============================================================
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

_DETERMINISTIC_SEED = int(os.environ.get("REFERTRACK_SEED", "0"))
random.seed(_DETERMINISTIC_SEED)
torch.manual_seed(_DETERMINISTIC_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_DETERMINISTIC_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)


def _resolve_view_list_from_ckpt(ckpt_path: str) -> list:
    """Read view_list from model_config.json next to the checkpoint."""
    ckpt_dir = osp.dirname(ckpt_path)
    for cand_dir in (ckpt_dir, osp.dirname(ckpt_dir)):
        mc_path = osp.join(cand_dir, "model_config.json")
        if osp.isfile(mc_path):
            with open(mc_path, "r") as f:
                data = json.load(f)
            vl = data.get("view_list") or ["forward"]
            return list(vl)
    raise FileNotFoundError(
        f"model_config.json not found near {ckpt_path}; cannot infer view_list for agent dispatch"
    )


def _is_single_view_forward(view_list: list) -> bool:
    return len(view_list) == 1 and view_list[0] == "forward"


def _log_determinism_env() -> None:
    """Log versions and yolo11x.pt md5 that affect eval reproducibility."""
    import hashlib
    import platform

    try:
        import ultralytics  # type: ignore
        ult_ver = ultralytics.__version__
    except Exception as e:
        ult_ver = f"<import error: {e}>"

    md5_str = "<not found>"
    yolo_pt = "yolo11x.pt"
    try:
        from referTrack.constants import REFER_TRACKER_CFG
        yolo_pt = REFER_TRACKER_CFG.get("yolo_model", yolo_pt)
        # yolo_pt = "yolo11n.pt"  # ablation study
    except Exception:
        pass
    for cand in (yolo_pt, osp.join(os.getcwd(), yolo_pt)):
        if osp.isfile(cand):
            h = hashlib.md5()
            with open(cand, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            md5_str = f"{h.hexdigest()}  ({cand})"
            break

    print(
        "[SIM-EVAL][determinism] "
        f"py={platform.python_version()}  "
        f"torch={torch.__version__} cuda={torch.version.cuda}  "
        f"ultralytics={ult_ver}  "
        f"cudnn.det={torch.backends.cudnn.deterministic} "
        f"cudnn.bench={torch.backends.cudnn.benchmark}  "
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED','<unset>')}  "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG','<unset>')}  "
        f"yolo_md5={md5_str} "
        f"yolo_pt={yolo_pt}"
    )


def main():
    parser = argparse.ArgumentParser(description="ReferTrack Simulation Eval")
    parser.add_argument(
        "--run-type", choices=["eval"], default="eval",
        help="eval (navigation) only",
    )
    parser.add_argument("--exp-config", type=str, required=True,
                        help="habitat eval config yaml")
    parser.add_argument("--split-id", type=int, required=True)
    parser.add_argument("--split-num", type=int, default=7)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="ReferTrack .pt (model_config.json must sit next to it)")
    parser.add_argument("--max-nums", type=int, default=-1)
    parser.add_argument(
        "--fallback-stop", action="store_true",
        help="Force stop when CoT predicts NO_EXIST (default: trust the planner)",
    )
    parser.add_argument(
        "--yolo-model", type=str, default=None,
        help="Override online-tracker YOLO weights (e.g. yolo11m.pt). "
             "Also accepts REFER_YOLO_MODEL; otherwise REFER_TRACKER_CFG['yolo_model'].",
    )
    parser.add_argument(
        "--history", type=int, default=None,
        help="Override history length H (default: model_config.json). "
             "Also accepts REFER_HISTORY.",
    )
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="Habitat-style config overrides",
    )
    args = parser.parse_args()

    import habitat  # noqa: WPS433
    import numpy as np  # noqa: WPS433
    from habitat.datasets import make_dataset  # noqa: WPS433

    np.random.seed(_DETERMINISTIC_SEED)
    from referTrack.dataset import evt_bench  # noqa: F401,WPS433

    # Override YOLO before the agent / determinism log so tracker and md5 agree.
    yolo_override = args.yolo_model or os.environ.get("REFER_YOLO_MODEL")
    if yolo_override:
        from referTrack.constants import REFER_TRACKER_CFG
        REFER_TRACKER_CFG["yolo_model"] = yolo_override
        print(f"[SIM-EVAL] yolo_model override → {yolo_override}")

    history_override = args.history
    if history_override is None:
        env_h = os.environ.get("REFER_HISTORY")
        if env_h is not None and str(env_h).strip() != "":
            history_override = int(env_h)
    if history_override is not None:
        print(f"[SIM-EVAL] history override → {history_override}")

    _log_determinism_env()

    view_list = _resolve_view_list_from_ckpt(args.ckpt_path)
    if not _is_single_view_forward(view_list):
        raise ValueError(
            "This release only supports the single-view forward ReferTrack checkpoint "
            f"(view_list=['forward']), got view_list={view_list}"
        )
    from referTrack.baseline.trained_agent_refer import evaluate_agent
    print(f"[SIM-EVAL] view_list={view_list} → single-view ReferAgent")

    config = habitat.get_config(args.exp_config, args.opts)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)
    os.makedirs(args.save_path, exist_ok=True)

    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    dataset_split = dataset.get_splits(args.split_num, allow_uneven_splits=True)[args.split_id]

    evaluate_agent(
        config=config,
        dataset_split=dataset_split,
        save_path=args.save_path,
        ckpt_path=args.ckpt_path,
        max_nums=args.max_nums,
        fallback_stop_on_noexist=args.fallback_stop,
        history=history_override,
    )


if __name__ == "__main__":
    main()
