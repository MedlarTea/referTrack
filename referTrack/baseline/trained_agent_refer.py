"""
ReferTrack Habitat Agent

Replaces the bbox source used in `trained_agent_multiview_unified_tvbi.py`:

- **Original TVBI agent**: each step reads the **GT target bbox** from lab sensor
  `agent_1_main_humanoid_detector_sensor['box']` and feeds it into the model's
  TVBI path. This is only an upper-bound reference; real deployment has no GT.
- **This ReferAgent**: each step runs online YOLO+ByteTrack tracking that is
  **identical to the data-generation pipeline** (`REFER_TRACKER_CFG`,
  `persist=True`) on the forward-view image to get `track_bboxes`, then on the
  agent side:
    1. Keep area top-N_MAX + shuffle (identical to `JsonReferTrackingDataset`)
    2. Build catalog (21 slots; last is virtual NO_EXIST)
    3. Every step runs `inference_refer_navigation` (two forwards + KV-cache
       reuse, §M-3); constrained CoT argmax picks the target slot and also
       produces a planner trajectory
    4. **Target bbox history**: maintain `self._target_bbox_hist`; **append the
       CoT-selected cand_bbox once per step** (append a zero vector on
       NO_EXIST / empty catalog). Does not rely on tracker track_id continuity
       — so history stays correct even when YOLO id-switches.
    5. Pass 0 for `bbox_curr` (training N-IMPL-2: vis_f does not read bbox_curr)
    6. **Run the model even when the catalog is empty (K=0)** — only the
       trailing virtual NO_EXIST slot remains; the model should output
       `<NO_EXIST>` plus a "how to move when the target is lost" trajectory
       from history bbox + vision (training Q-C: L_nav is still supervised on
       NO_EXIST samples).
    7. `fallback_stop_on_noexist` (default False): optionally force a stop when
       CoT predicts NO_EXIST; by default trust the planner output

Decision alignment:
  - E-1: Tracker config hard-aligned to `REFER_TRACKER_CFG`, persist=True
  - Q-C: Planner still emits a trajectory on NO_EXIST samples ("how to move
    when lost"); no special-case handling
  - D-1/T-2: forward view only
  - N-IMPL-2: current-frame vis_f does not inject bbox (handled inside model
    forward)
  - D-6: bbox_hist scheme A (write 0 when tid<0; here write 0 on CoT miss)
  - M-3: inference uses "two forwards + KV-cache reuse" (see
    `inference_refer_navigation`)

Usage: see `run_eval_refer_sim.py`.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import json
import os
import os.path as osp
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import habitat
import imageio
import numpy as np
import torch
from habitat.config.default_structured_configs import AgentConfig
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel
from tqdm import trange

from referTrack.constants import (
    REFER_TRACKER_CFG,
    VIEW_YAWS,
)
from referTrack.eval.load_refer_ckpt import load_refer_model
from referTrack.eval.cache_gridpool import (
    VisionCacheConfig,
    VisionFeatureCacher,
    grid_pool_tokens,
)
from referTrack.model.referTrack import ReferTrack

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Matches training-side tracking_multiview_dataset_refer.py
BBOX_IMAGE_SIZE = 384.0


# =========================================================================
# Sensor adapter
# =========================================================================


def _get_forward_rgb(observations) -> Optional[np.ndarray]:
    """Get forward-view RGB from habitat obs (same as TVBI agent: jaw_rgb)."""
    rgb = observations.get("agent_1_articulated_agent_jaw_rgb")
    if rgb is None:
        return None
    return np.ascontiguousarray(rgb[:, :, :3]).astype(np.uint8)


def _extract_main_human_bbox(observations) -> Optional[np.ndarray]:
    """Extract the main-person bbox from lab-sensor observations (GT overlay for vis only).

    Same source as the TVBI agent: `agent_1_main_humanoid_detector_sensor['box']`.
    Returns `np.ndarray(4,)` [x1,y1,x2,y2] in pixel coords, or `None`.

    Note: ReferAgent **does not use this GT bbox for decisions** (referring is
    CoT-driven); it is only drawn as a green box on rendered frames for offline
    comparison.
    """
    if observations is None:
        return None
    det = observations.get("agent_1_main_humanoid_detector_sensor")
    if det is None:
        return None
    box = det.get("box") if isinstance(det, dict) else None
    if box is None:
        return None
    if hasattr(box, "shape") and box.shape == (4,):
        return np.asarray(box, dtype=np.float32)
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return np.asarray(box[:4], dtype=np.float32)
    return None


def _bbox_iou(a: List[float], b: List[float]) -> float:
    """IoU of pixel-coord boxes [x1,y1,x2,y2]."""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 1e-6 else 0.0


# =========================================================================
# Online tracker (YOLO + ByteTrack, persist=True)
# =========================================================================


class _OnlineTracker:
    """Lazy-load YOLO; reset once per episode; persist=True across the episode.

    Geometrically identical to `match_target_track_bbox.py::_run_tracker_on_frames_dir`:
    yolo11x + bytetrack + imgsz=384 + conf=0.1 + iou=0.9 + classes=[0].
    """

    def __init__(self):
        self._model = None
        self._track_kwargs = None
        # persist state: YOLO accumulates tid by call count; reset() rebuilds the
        # model to fully clear tid state (ultralytics has no explicit tracker-reset API)

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
        # Determinism: half=False forces fp32 inference (avoids non-deterministic
        # fp16 GEMM reduction); bind device to the current visible GPU (0 under
        # CUDA_VISIBLE_DEVICES) so ultralytics does not auto-dispatch; set
        # augment / agnostic_nms False so defaults cannot change.
        self._track_kwargs = dict(
            classes=cfg["classes"],
            conf=cfg["conf"],
            iou=cfg["iou"],
            imgsz=cfg["imgsz"],
            tracker=cfg["tracker_yaml"],
            persist=True,
            verbose=False,
            half=False,
            device=0,
            augment=False,
            agnostic_nms=False,
        )

    def reset(self):
        """Reset the tracker (new episode). Rebuild the model to clear persist state.

        Also zero ultralytics' class-level `BaseTrack._count`. Without this, track
        ids keep incrementing across episodes (reloading YOLO does not clear it).
        The agent itself does not rely on tid continuity (see class docstring),
        but resetting is friendlier for debug/vis and avoids implicit cross-episode
        state if a downstream consumer keys policy on tid.
        """
        self._model = None
        self._track_kwargs = None
        try:
            from ultralytics.trackers.basetrack import BaseTrack  # type: ignore
            if hasattr(BaseTrack, "reset_id"):
                BaseTrack.reset_id()
            else:
                BaseTrack._count = 0
        except Exception:
            pass

    def track_one(self, frame_rgb: np.ndarray) -> List[List[float]]:
        """Track one frame → `[[tid, x1,y1,x2,y2], ...]` (pixel coords)."""
        self._ensure_model()
        # YOLO accepts numpy (H,W,3) BGR or RGB; Ultralytics handles it internally
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
            out.append([int(tid) if tid is not None else -1,
                        float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])])
        return out


# =========================================================================
# Evaluate entry
# =========================================================================


def evaluate_agent(
    config,
    dataset_split,
    save_path: str,
    ckpt_path: Optional[str] = None,
    max_nums: int = -1,
    fallback_stop_on_noexist: bool = False,
    history: Optional[int] = None,
) -> None:
    """Evaluate ReferAgent in a habitat env.

    Per episode: reset tracker → step act → record metrics → save video and info json.
    history: override model_config history length (None = use ckpt config); for history ablation.
    """
    robot_config = ReferAgent(
        save_path,
        ckpt_path,
        fallback_stop_on_noexist=fallback_stop_on_noexist,
        history=history,
    )

    init_instruction = True

    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        robot_config.reset()

        num_episodes = len(env.episodes)
        if max_nums > 0:
            num_episodes = min(num_episodes, max_nums)

        for _ in trange(num_episodes):
            obs = env.reset()

            # Light preset (same as TVBI agent)
            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0],
                          model=LightPositionModel.Global),
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0],
                          model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0],
                          model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0],
                          model=LightPositionModel.Global),
            ]
            sim.set_light_setup(light_setup)

            result: Dict[str, Any] = {}
            record_infos: List[Dict[str, Any]] = []

            # Fetch instruction text for this episode if available
            if init_instruction:
                try:
                    instruction = env.current_episode.info.get('instruction', None)
                    init_instruction = False
                except Exception:
                    instruction = None

            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            too_far_count = 0
            cot_noexist_count = 0
            catalog_empty_count = 0
            status = "Normal"
            finished = False

            # New episode → reset tracker + agent buffer
            robot_config.reset_episode()

            while not env.episode_over:
                record_info: Dict[str, Any] = {}
                obs = sim.get_sensor_observations()

                # Lab-sensor GT target bbox (vis overlay only; not used for decisions)
                try:
                    detector_obs = env.task._get_observations(env.current_episode)
                except Exception:
                    detector_obs = None
                human_bbox_px = _extract_main_human_bbox(detector_obs)

                action, aux = robot_config.act(
                    obs,
                    env.current_episode.episode_id,
                    instruction,
                    gt_bbox_px=human_bbox_px,
                )

                action_dict = {
                    "action": (
                        "agent_0_humanoid_navigate_action",
                        "agent_1_base_velocity",
                        "agent_2_oracle_nav_randcoord_action_obstacle",
                        "agent_3_oracle_nav_randcoord_action_obstacle",
                        "agent_4_oracle_nav_randcoord_action_obstacle",
                        "agent_5_oracle_nav_randcoord_action_obstacle",
                    ),
                    "action_args": {
                        "agent_1_base_vel": action,
                    },
                }
                iter_step += 1
                env.step(action_dict)

                info = env.get_metrics()
                if info["human_following"] == 1.0:
                    followed_step += 1
                    too_far_count = 0
                else:
                    pass

                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = "Lost"
                        finished = False
                        break

                if aux.get("cot_noexist"):
                    cot_noexist_count += 1
                if aux.get("catalog_empty"):
                    catalog_empty_count += 1

                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(
                    np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)
                )
                record_info["facing"] = float(info["human_following"])
                record_info["base_velocity"] = action
                record_info["cot_pred_slot"] = aux.get("pred_slot")
                record_info["cot_noexist"] = aux.get("cot_noexist")
                record_info["catalog_empty"] = aux.get("catalog_empty")
                record_info["cand_num"] = aux.get("cand_num")
                record_info["target_tid"] = aux.get("target_tid")
                record_info["gt_bbox_px"] = (
                    human_bbox_px.tolist() if human_bbox_px is not None else None
                )
                record_info["gt_slot"] = aux.get("gt_slot")
                record_info["gt_iou"] = aux.get("gt_iou")
                record_info["cot_correct"] = aux.get("cot_correct")
                record_infos.append(record_info)

                if info["human_collision"] == 1.0:
                    status = "Collision"
                    finished = False
                    break

            info = env.get_metrics()
            robot_config.reset(env.current_episode)

            if env.episode_over:
                finished = True

            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split(".")[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{env.current_episode.episode_id}_info.json"), "w") as f:
                json.dump(record_infos, f, indent=2)

            result["finish"] = finished
            result["status"] = status
            if iter_step < 300:
                result["success"] = info["human_following_success"] and info["human_following"]
            else:
                result["success"] = info["human_following"]
            result["following_rate"] = followed_step / max(iter_step, 1)
            result["following_step"] = followed_step
            result["total_step"] = iter_step
            result["collision"] = info["human_collision"]
            result["cot_noexist_rate"] = cot_noexist_count / max(iter_step, 1)
            result["catalog_empty_rate"] = catalog_empty_count / max(iter_step, 1)
            if instruction is not None:
                result["instruction"] = instruction
            with open(os.path.join(save_dir, f"{env.current_episode.episode_id}.json"), "w") as f:
                json.dump(result, f, indent=2)


# =========================================================================
# ReferAgent
# =========================================================================


class ReferAgent(AgentConfig):
    """ReferTrack agent: vision encoding + online tracker + per-step CoT + planner.

    **CoT-driven target history** (revised 2026-04-25; sticky tracking removed):
      Do not rely on tracker track_id continuity (yolo+bytetrack can id-switch).
      **Re-run CoT every step** and append the CoT-selected cand_bbox to
      `self._target_bbox_hist`, which becomes `bbox_hist` for the next step.
      If CoT predicts `<NO_EXIST>` → write 0 for this frame (aligned with
      training scheme A).

    Forward flow (each step):
      1. jaw_rgb → DINO+SigLIP encode → (4, C) coarse, (64, C) fine
      2. tracker.track_one(frame) → `[[tid, x1,y1,x2,y2], ...]`
         (tid is no longer consumed by the agent; only bbox geometry builds the catalog)
      3. Area top-N_MAX keep + shuffle (seed=hash(ep_id, step)) → catalog
      4. **Always run the model regardless of K** (when K=0 the catalog is only
         the trailing virtual NO_EXIST slot; the model learns to emit NO_EXIST
         plus a "how to move when lost" trajectory from history bbox + visual
         context, matching training Q-C; no early fallback in the agent)
      5. `inference_refer_navigation(...)` → `pred_slot` + `tau_pred`
         (two forwards + KV-cache reuse, §M-3; train/infer strictly aligned)
      6. Append to history:
         - `pred_slot ∈ [0, N_MAX)` → `cand_bbox[pred_slot]` (normalized 0~1)
         - `pred_slot = N_MAX` (NO_EXIST) → zero vector (tid<0 semantics)
      7. action = velocity from `tau_pred` (trust planner by default; only force
         a stop on NO_EXIST if `fallback_stop_on_noexist` is on)
    """

    def __init__(
        self,
        result_path: str,
        ckpt_path: Optional[str] = None,
        fallback_stop_on_noexist: bool = False,
        history: Optional[int] = None,
    ):
        super().__init__()
        print("[ReferAgent] Initializing")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)

        # Controlled by eval shell SAVE_VIDEO=0/1; when off, skip per-frame vis
        # and mp4 write (info.json / metrics unchanged). Default on if unset.
        self.save_video: bool = os.environ.get("SAVE_VIDEO", "1").strip().lower() not in (
            "0", "false", "no", ""
        )
        print(f"[ReferAgent] save_video={self.save_video}")

        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback_stop_on_noexist = fallback_stop_on_noexist
        # GT slot match threshold (aligned with `REFER_TRACKER_CFG["target_iou_thresh"]`)
        self.gt_iou_thresh: float = float(REFER_TRACKER_CFG.get("target_iou_thresh", 0.2))

        if ckpt_path is None:
            raise ValueError("Must provide checkpoint path for ReferAgent.")
        self._ckpt_path = ckpt_path

        self.history = 31
        self._history_override = history
        self.model: Optional[ReferTrack] = None
        self.model_config = None
        self._vision_cache = None

        self._init_model()

        if self._history_override is not None:
            self.history = int(self._history_override)
            if self.history < 0:
                raise ValueError(f"history must be >= 0, got {self.history}")
            print(f"[ReferAgent] history override → {self.history}")

        self.view_list = self.model_config.view_list or ["forward"]
        if "forward" not in self.view_list:
            raise ValueError(f"ReferAgent requires 'forward' in view_list, got {self.view_list}")
        self.N_MAX = int(self.model_config.max_candidates)

        # Per-episode state
        self._tracker = _OnlineTracker()
        # maxlen at least 1: history=0 still needs _ep_step; buffer is not fed to the model
        _buf_len = max(int(self.history), 1)
        self._coarse_hist: Dict[str, deque] = {
            v: deque(maxlen=_buf_len) for v in self.view_list
        }
        # CoT-driven target bbox history (already normalized [0,1], length <= history)
        # Append once per step: CoT-selected cand_bbox or [0,0,0,0] (NO_EXIST / catalog empty)
        self._target_bbox_hist: deque = deque(maxlen=_buf_len)
        self._ep_step: int = 0

        # Visualization / video
        self.rgb_list: List[np.ndarray] = []
        self._last_pred_traj: Optional[np.ndarray] = None
        self._last_pred_slot: int = -1

        # Episode id (stashed at reset for saving video)
        self.episode_id: Optional[str] = None

        print(
            f"[ReferAgent] ready; history={self.history}  "
            f"view_list={self.view_list}  N_MAX={self.N_MAX}"
        )
    # ==================== Model init ====================

    def _init_model(self):
        print(f"[ReferAgent] Loading ckpt: {self._ckpt_path}")
        model, model_cfg = load_refer_model(self._ckpt_path, self.model_device)
        self.model = model
        self.model_config = model_cfg
        self.history = int(model_cfg.history)
        print(
            f"[ReferAgent] model loaded; history={self.history}  "
            f"N_MAX={int(self.model_config.max_candidates)}"
        )

    def _ensure_vision_cache(self):
        if self._vision_cache is None:
            cfg = VisionCacheConfig(
                image_size=384,
                batch_size=1,
                device=("cuda" if torch.cuda.is_available() else "cpu"),
            )
            self._vision_cache = VisionFeatureCacher(cfg)
            self._vision_cache.eval()
        return self._vision_cache

    def _encode_frame_tokens(self, rgb_np: np.ndarray) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        enc = self._ensure_vision_cache()
        try:
            from PIL import Image
            pil = Image.fromarray(rgb_np.astype(np.uint8))
            tok_dino, Hp, Wp = enc._encode_dino([pil])
            tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))
            Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)
            Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)[0].float()
            Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)[0].float()
            return Vcoarse, Vfine
        except Exception as e:
            print(f"[ReferAgent] encode failed: {e}")
            return None, None

    # ==================== Reset ====================

    def reset(self, episode: Optional[NavigationEpisode] = None):
        """Called at episode end: save video + clear vision/bbox buffers + reset tracker."""
        if self.save_video and self.rgb_list and episode is not None:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split(".")[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            out_video = os.path.join(save_dir, f"{episode.episode_id}.mp4")
            try:
                imageio.mimsave(out_video, self.rgb_list, fps=10, codec="libx264",
                                quality=8, macro_block_size=1)
                print(f"[ReferAgent] saved episode video: {out_video}")
            except Exception as e:
                print(f"[ReferAgent] save video failed: {e}")
        self.rgb_list = []
        self.reset_episode()

    def reset_episode(self):
        """Called before an episode starts (clear state; do not save video)."""
        for v in self.view_list:
            self._coarse_hist[v].clear()
        self._target_bbox_hist.clear()
        self._ep_step = 0
        self._last_pred_traj = None
        self._last_pred_slot = -1
        self._tracker.reset()

    # ==================== Catalog building ====================

    @staticmethod
    def _bbox_area(b: List[float]) -> float:
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    def _build_catalog(
        self,
        track_bboxes: List[List[float]],
        step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, List[int]]:
        """Build catalog with the same rules as the training side.

        Returns:
            cand_bbox:       (1, N_MAX+1, 4) normalized
            cand_slot_valid: (1, N_MAX+1) bool
            K:               number of real candidates
            tid_per_slot:    list of tid for slot [0..K-1] (-1 for pad slots; NO_EXIST excluded)
        """
        N_MAX = self.N_MAX

        # 1) Clean
        cleaned: List[Tuple[int, List[float]]] = []
        for row in track_bboxes:
            if not isinstance(row, (list, tuple)) or len(row) != 5:
                continue
            tid = int(row[0])
            bbox = [float(v) for v in row[1:5]]
            cleaned.append((tid, bbox))

        # 2) Area top-N_MAX
        if len(cleaned) > N_MAX:
            cleaned = sorted(cleaned, key=lambda it: self._bbox_area(it[1]), reverse=True)[:N_MAX]

        # 3) Shuffle (seed matches training: hash(ep_id, step_idx))
        K = len(cleaned)
        if K > 0:
            seed = int((hash((self.episode_id, step_idx)) & 0xFFFFFFFF))
            rng = np.random.default_rng(seed=seed)
            order = rng.permutation(K)
            cleaned = [cleaned[i] for i in order]

        # 4) Build tensors
        cand_bbox = torch.zeros(N_MAX + 1, 4, dtype=torch.float32)
        cand_slot_valid = torch.zeros(N_MAX + 1, dtype=torch.bool)
        tid_per_slot: List[int] = [-1] * N_MAX

        for k, (tid, bbox) in enumerate(cleaned):
            cand_bbox[k] = torch.tensor(bbox, dtype=torch.float32) / BBOX_IMAGE_SIZE
            cand_slot_valid[k] = True
            tid_per_slot[k] = tid

        # Trailing virtual NO_EXIST slot
        cand_slot_valid[N_MAX] = True

        return (
            cand_bbox.unsqueeze(0).to(self.model_device),
            cand_slot_valid.unsqueeze(0).to(self.model_device),
            K,
            tid_per_slot,
        )

    # ==================== Act ====================

    def act(
        self,
        observations,
        episode_id,
        instruction: Optional[str] = None,
        gt_bbox_px: Optional[np.ndarray] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Called by the sim main loop.

        Args:
            observations:  habitat sim obs
            episode_id:    current episode id (used for shuffle seed)
            instruction:   follow-person instruction
            gt_bbox_px:    lab-sensor main-person GT bbox [x1,y1,x2,y2] (pixels);
                           **vis overlay and aux metrics only**, not used for decisions.
                           None (e.g. target off-screen) is treated as GT=NO_EXIST.

        Returns:
            action: [vx, vy, wz]
            aux:    per-step debug info (pred_slot / gt_slot / cot_correct / ...)
        """
        self.episode_id = episode_id

        # 1) Get forward RGB
        fwd_rgb = _get_forward_rgb(observations)
        if fwd_rgb is None:
            print("[ReferAgent] jaw_rgb missing; stopping")
            return [0.0, 0.0, 0.0], {"error": "no_rgb"}

        step_idx = self._ep_step
        self._ep_step += 1

        # 2) Online tracker (tid is debug-only; not consumed by the agent)
        track_bboxes = self._tracker.track_one(fwd_rgb)

        # 3) Build catalog
        cand_bbox, cand_slot_valid, K, tid_per_slot = self._build_catalog(
            track_bboxes, step_idx
        )

        # 4) Vision encode
        Vc_dict: Dict[str, torch.Tensor] = {}
        Vf_dict: Dict[str, torch.Tensor] = {}
        for v in self.view_list:
            if v == "forward":
                rgb = fwd_rgb
            else:
                raw = observations.get(f"agent_1_articulated_agent_{v}_rgb")
                rgb = np.ascontiguousarray(raw[:, :, :3]).astype(np.uint8) if raw is not None else fwd_rgb
            vc, vf = self._encode_frame_tokens(rgb)
            if vc is None or vf is None:
                return [0.0, 0.0, 0.0], {"error": "encode_failed"}
            Vc_dict[v] = vc
            Vf_dict[v] = vf

        for v in self.view_list:
            self._coarse_hist[v].append(Vc_dict[v].cpu())

        # 5) Prepare tensors for inference_refer_navigation
        H = self.history
        V = len(self.view_list)
        fwd_idx = self.view_list.index("forward")
        feat_dim = int(Vc_dict["forward"].size(-1))

        # 5a) coarse_tokens / coarse_tidx (left-pad with earliest frame; aligned with dataset)
        #     history=0: feed no hist coarse (current fine only)
        if H == 0:
            coarse_tokens = torch.zeros(
                1, 0, feat_dim, dtype=torch.float32, device=self.model_device
            )
            coarse_tidx = torch.zeros(
                1, 0, dtype=torch.long, device=self.model_device
            )
        else:
            T = len(self._coarse_hist["forward"])
            trim_len = min(H, T)
            missing = H - trim_len
            coarse_list: List[torch.Tensor] = []
            coarse_tidx_list: List[torch.Tensor] = []
            first_tok: Optional[torch.Tensor] = None
            pending_pad = missing
            for t in range(H):
                if t < missing:
                    continue
                tok_per_view = [self._coarse_hist[v][t - missing].to(self.model_device)
                                for v in self.view_list]
                tok_views = torch.cat(tok_per_view, dim=0)  # (V*4, C)
                if first_tok is None:
                    first_tok = tok_views
                    for pt in range(pending_pad):
                        coarse_list.append(first_tok)
                        coarse_tidx_list.append(
                            torch.full((tok_views.size(0),), pt, device=self.model_device)
                        )
                    pending_pad = 0
                coarse_list.append(tok_views)
                coarse_tidx_list.append(
                    torch.full((tok_views.size(0),), t, device=self.model_device)
                )
            coarse_tokens = torch.cat(coarse_list, dim=0).unsqueeze(0)
            coarse_tidx = torch.cat(coarse_tidx_list, dim=0).unsqueeze(0)

        # 5b) fine_tokens / fine_tidx
        fine_tokens = torch.cat(
            [Vf_dict[v] for v in self.view_list], dim=0
        ).to(self.model_device).unsqueeze(0)
        fine_tidx = torch.full(
            (1, fine_tokens.size(1)), fill_value=H, dtype=torch.long,
            device=self.model_device,
        )

        # 5c) yaw
        if H == 0:
            yaw_hist = torch.zeros(1, 0, dtype=torch.float32)
        else:
            yaw_hist = torch.tensor(
                [VIEW_YAWS[v] for v in self.view_list] * H,
                dtype=torch.float32,
            ).unsqueeze(0)
        yaw_curr = torch.tensor(
            [VIEW_YAWS[v] for v in self.view_list], dtype=torch.float32
        ).unsqueeze(0)

        # 5d) bbox_hist source = **CoT history queue** `self._target_bbox_hist`
        #     (already normalized; one per step: CoT-selected cand_bbox or [0,0,0,0])
        #     Note: current-frame bbox is unknown until CoT runs, so history has
        #     only step_idx valid items (exactly the H history frames the model sees).
        #     Left pad: same as dataset — copy the earliest valid frame into the pad;
        #     if history is all zeros (e.g. first steps were all NO_EXIST), pad is 0 too.
        bbox_hist = torch.zeros(H * V, 4, dtype=torch.float32)
        Th = len(self._target_bbox_hist)  # actual items in the history queue
        if H > 0 and Th > 0:
            trim_h = min(H, Th)
            mh = H - trim_h
            for t in range(mh, H):
                bh = self._target_bbox_hist[t - mh]
                bbox_hist[t * V + fwd_idx] = torch.tensor(bh, dtype=torch.float32)
            # Left pad: copy the earliest non-zero (CoT-hit) frame
            earliest_valid = None
            for t in range(mh, H):
                b = bbox_hist[t * V + fwd_idx]
                if b.sum().item() > 0:
                    earliest_valid = b.clone()
                    break
            if earliest_valid is not None and 0 < mh < H:
                for t in range(mh):
                    bbox_hist[t * V + fwd_idx] = earliest_valid
        bbox_hist = bbox_hist.unsqueeze(0).to(self.model_device)

        # 5e) bbox_curr: training N-IMPL-2 — current-frame vis_f does not inject bbox.
        #     Agent passes a zero vector; model vis_f uses the TI-only path (ignores this).
        bbox_curr = torch.zeros(V, 4, dtype=torch.float32).unsqueeze(0).to(self.model_device)

        # 5f) GT slot overlay (aux metrics + vis only; not used for decisions):
        #     Normalize gt_bbox_px (pixels) and IoU-match against each catalog cand_bbox
        #     (slot with IoU >= gt_iou_thresh is GT); no match → NO_EXIST.
        gt_slot = self.N_MAX  # default NO_EXIST
        best_gt_iou = 0.0
        if gt_bbox_px is not None and K > 0:
            gt_norm = [float(gt_bbox_px[i]) / BBOX_IMAGE_SIZE for i in range(4)]
            cb = cand_bbox[0].cpu().numpy()
            for k in range(K):
                iu = _bbox_iou(gt_norm, cb[k].tolist())
                if iu > best_gt_iou:
                    best_gt_iou = iu
                    if iu >= self.gt_iou_thresh:
                        gt_slot = k
            # If best IoU is still below threshold, keep gt_slot = N_MAX (NO_EXIST)

        # 6) Forward (always run the model regardless of K; when K=0 the catalog
        #     is only the trailing virtual <NO_EXIST> slot; the model should emit
        #     <NO_EXIST> plus a "how to move when lost" trajectory, matching
        #     training Q-C. **No early fallback in the agent.**).
        aux: Dict[str, Any] = {
            "step_idx": step_idx,
            "cand_num": K,
            "catalog_empty": (K == 0),
            "gt_slot": gt_slot,
            "gt_iou": float(best_gt_iou),
        }

        alpha_fake = torch.zeros((1, 3), dtype=torch.float32).to(self.model_device)
        alpha_fake[0, 0] = 0.8
        alpha_fake[0, 1] = 0.8
        alpha_fake[0, 2] = 1.572

        # 6a) Inference-only API (two forwards + KV-cache reuse, plan §M-3):
        #     - Step 1 runs to <reasoning_open>, constrained argmax on next-token
        #       logits → pred_slot (no teacher forcing)
        #     - Step 2 feeds only [<PRED_SLOT>, </reasoning>, <act>] (3 tokens),
        #       reuses KV cache, takes <act> hidden → planner → tau
        #
        out = self.model.inference_refer_navigation(
            coarse_tokens=coarse_tokens,
            coarse_tidx=coarse_tidx,
            fine_tokens=fine_tokens,
            fine_tidx=fine_tidx,
            cand_bbox=cand_bbox,
            cand_slot_valid=cand_slot_valid,
            bbox_hist=bbox_hist,
            bbox_curr=bbox_curr,
            instructions=[instruction or "follow the person"],
            yaw_hist=yaw_hist,
            yaw_curr=yaw_curr,
            alpha=alpha_fake,
        )
        tau = out["trajectory"]
        pred_slot = out["pred_slot"]

        pred_slot_int = int(pred_slot.item())
        self._last_pred_slot = pred_slot_int
        aux["pred_slot"] = pred_slot_int
        aux["cot_correct"] = (pred_slot_int == gt_slot)

        # 7) Append to target bbox history from pred_slot (no tracker id-switch dependency)
        if 0 <= pred_slot_int < self.N_MAX:
            # CoT hit: take cand_bbox (already normalized [0,1])
            curr_tgt_bbox = cand_bbox[0, pred_slot_int].cpu().numpy().tolist()
            self._target_bbox_hist.append(curr_tgt_bbox)
            aux["cot_noexist"] = False
            aux["target_tid"] = tid_per_slot[pred_slot_int]  # record only, for debug
        else:
            # <NO_EXIST>: write 0 for this frame (aligned with training scheme A)
            self._target_bbox_hist.append([0.0, 0.0, 0.0, 0.0])
            aux["cot_noexist"] = True
            aux["target_tid"] = -1

        # 8) waypoint → velocity
        tau_cpu = tau.detach().float().cpu().numpy()
        self._last_pred_traj = tau_cpu[0]

        if aux["cot_noexist"] and self.fallback_stop_on_noexist:
            action = self._fallback_action("cot_noexist")
        else:
            wp = tau[0, 1]
            x = float(wp[0].item()); y = float(wp[1].item())
            theta = float(wp[2].item()) if wp.numel() >= 3 else 0.0
            dt = 0.1
            action = [x / dt, y / dt, theta / dt]

        # 9) Visualization (like run_eval_refer_episode.py: GT green + Pred red + traj + text)
        #    Skip entirely when SAVE_VIDEO=0 to avoid PIL draw + frame cache + mp4 encode
        if self.save_video:
            self._append_vis_frame(
                fwd_rgb=fwd_rgb,
                cand_bbox=cand_bbox,
                cand_slot_valid=cand_slot_valid,
                pred_slot=pred_slot_int,
                gt_bbox_px=gt_bbox_px,
                gt_slot=gt_slot,
                best_gt_iou=best_gt_iou,
                pred_traj=tau_cpu[0],
                instruction=instruction,
                step_idx=step_idx,
            )
        return action, aux

    def _fallback_action(self, reason: str) -> List[float]:
        # S-1 policy: stop in place; could become "slow in-place rotate search" (future work)
        _ = reason
        return [0.0, 0.0, 0.0]

    # ==================== Visualization ====================

    def _append_vis_frame(
        self,
        fwd_rgb: np.ndarray,
        cand_bbox: torch.Tensor,          # (1, N_MAX+1, 4)
        cand_slot_valid: torch.Tensor,    # (1, N_MAX+1)
        pred_slot: int,
        gt_bbox_px: Optional[np.ndarray] = None,   # lab-sensor GT [x1,y1,x2,y2] (pixels)
        gt_slot: int = -1,
        best_gt_iou: float = 0.0,
        pred_traj: Optional[np.ndarray] = None,    # (n_waypoints, 3+)
        instruction: Optional[str] = None,
        step_idx: int = 0,
    ):
        """Render one vis frame, aligned with `run_eval_refer_episode.py::render_frame`:

          - Thin gray boxes + labels: all catalog candidates except GT/Pred
          - Thick green box: GT target bbox (from lab sensor; skip if NO_EXIST/missing)
          - Thick red box: Pred target bbox (CoT argmax)
          - Cyan trajectory: Pred waypoints (robot-local xy, mapped to image bottom)
          - Top text panel: step / instruction / K / GT slot / Pred slot / ✓✗
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.fromarray(fwd_rgb.astype(np.uint8), mode="RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size

            cb = cand_bbox[0].cpu().numpy()
            cv = cand_slot_valid[0].cpu().numpy()
            # MV-GC compat: under a global-catalog model cand_bbox is (CAT_LEN, V, 4);
            # vis only draws the forward view, so slice to (CAT_LEN, 4).
            # Single-view / old ReferAgentMV already has cb as (CAT_LEN, 4); skip.
            if cb.ndim == 3 and cb.shape[-1] == 4:
                fwd_v_idx = self.view_list.index("forward") if "forward" in self.view_list else 0
                cb = cb[:, fwd_v_idx, :]      # (CAT_LEN, 4)
                # cand_slot_valid_per_view: (CAT_LEN, V) → take forward col; slot-level (CAT_LEN,) unchanged
                if cv.ndim == 2 and cv.shape[1] == cand_bbox.shape[2]:
                    cv = cv[:, fwd_v_idx]
            K = int(cv[: self.N_MAX].sum())

            # 1) Gray boxes + labels for all candidates except GT/Pred slots
            for k in range(self.N_MAX):
                if not bool(cv[k]):
                    continue
                b = cb[k]
                if b.sum() <= 0:
                    continue
                if k == pred_slot or k == gt_slot:
                    continue
                x1, y1, x2, y2 = (int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h))
                if x2 > x1 and y2 > y1:
                    draw.rectangle([x1, y1, x2, y2], outline=(160, 160, 160), width=2)
                    # Label at top-left
                    try:
                        font_s = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
                        )
                    except Exception:
                        font_s = ImageFont.load_default()
                    lbl = f"{k + 1}"
                    tx, ty = x1, max(0, y1 - 14)
                    bbox_t = draw.textbbox((tx, ty), lbl, font=font_s)
                    draw.rectangle(bbox_t, fill=(160, 160, 160))
                    draw.text((tx, ty), lbl, fill=(0, 0, 0), font=font_s)

            # 2) GT bbox (thick green) — draw in lab-sensor raw pixel coords
            if gt_bbox_px is not None:
                gx = [int(round(float(v))) for v in gt_bbox_px[:4]]
                gx1, gy1, gx2, gy2 = gx
                if gx2 > gx1 and gy2 > gy1:
                    draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 255, 0), width=4)
                    try:
                        font_s = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
                        )
                    except Exception:
                        font_s = ImageFont.load_default()
                    gt_lbl = (
                        f"GT #{gt_slot + 1}"
                        if 0 <= gt_slot < self.N_MAX
                        else "GT (no match)"
                    )
                    tx, ty = gx1, max(0, gy1 - 14)
                    bbox_t = draw.textbbox((tx, ty), gt_lbl, font=font_s)
                    draw.rectangle(bbox_t, fill=(0, 255, 0))
                    draw.text((tx, ty), gt_lbl, fill=(0, 0, 0), font=font_s)

            # 3) Pred bbox (thick red)
            if 0 <= pred_slot < self.N_MAX:
                b = cb[pred_slot]
                if b.sum() > 0:
                    x1, y1, x2, y2 = (int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h))
                    if x2 > x1 and y2 > y1:
                        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
                        try:
                            font_s = ImageFont.truetype(
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
                            )
                        except Exception:
                            font_s = ImageFont.load_default()
                        pred_lbl = f"Pred #{pred_slot + 1}"
                        tx, ty = x1, min(h - 14, y2 + 2)
                        bbox_t = draw.textbbox((tx, ty), pred_lbl, font=font_s)
                        draw.rectangle(bbox_t, fill=(255, 80, 80))
                        draw.text((tx, ty), pred_lbl, fill=(255, 255, 255), font=font_s)

            # 4) Trajectory (Pred cyan)
            if pred_traj is not None and pred_traj.size > 0:
                base_x = w // 2
                base_y = int(h * 0.86)
                scale = 120.0
                pts: List[Tuple[int, int]] = []
                n = min(pred_traj.shape[0], 64)
                for i in range(n):
                    xv = float(pred_traj[i, 0])
                    yv = float(pred_traj[i, 1]) if pred_traj.shape[1] >= 2 else 0.0
                    px = base_x - int(yv * scale)
                    py = base_y - int(xv * scale)
                    pts.append((px, py))
                for i in range(1, len(pts)):
                    draw.line([pts[i - 1], pts[i]], fill=(0, 0, 0), width=8)
                for i in range(1, len(pts)):
                    draw.line([pts[i - 1], pts[i]], fill=(0, 255, 200), width=4)
                if pts:
                    r = 4
                    sx, sy = pts[0]
                    draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(0, 255, 0))

            # 5) Top text panel
            try:
                font_n = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
                )
            except Exception:
                font_n = ImageFont.load_default()

            pred_text = (
                "<NO_EXIST>"
                if pred_slot >= self.N_MAX
                else f"<obj_{pred_slot + 1}>"
            )
            if 0 <= gt_slot < self.N_MAX:
                gt_text = f"<obj_{gt_slot + 1}>"
            elif gt_slot == self.N_MAX:
                gt_text = "<NO_EXIST>"
            else:
                gt_text = "N/A"
            cot_correct = (pred_slot == gt_slot) if gt_slot >= 0 else None
            mark = "" if cot_correct is None else (" ✓" if cot_correct else " ✗")
            lines = [
                f"Step {step_idx + 1}   K={K}   bestIoU={best_gt_iou:.2f}",
                (instruction or "")[:80],
                f"GT:   {gt_text}",
                f"Pred: {pred_text}{mark}",
            ]
            # Semi-transparent black panel
            panel_h = 18 * len(lines) + 8
            draw.rectangle([0, 0, w, panel_h], fill=(0, 0, 0))
            y0 = 4
            for ln in lines:
                draw.text((4, y0), ln, fill=(255, 255, 255), font=font_n)
                y0 += 18

            self.rgb_list.append(np.asarray(img))
        except Exception as e:
            print(f"[ReferAgent] vis failed: {e}")
            self.rgb_list.append(fwd_rgb)
