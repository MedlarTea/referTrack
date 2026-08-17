"""ReferTrack: referring-then-tracking VLA (eval).

Single-file model. Checkpoint keys stay `llm / proj / tvi / planner / act_token /
null_bbox_emb` — class rename does not change the .pt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from referTrack.constants import (
    MAX_CANDIDATES,
    NO_EXIST_TOKEN,
    OBJ_TOKEN_FMT,
    REFER_NEW_SPECIAL_TOKENS_COUNT,
    SEG_MARKER_TOKENS,
    VIEW_YAWS,
)


@dataclass
class ReferTrackConfig:
    pretrained_ckpt: str = ""
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = False
    view_list: Optional[List[str]] = None

    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = False

    n_waypoints: int = 8
    max_time: int = 4096
    beta_nav: float = 10.0
    use_angle_tvi: bool = True
    use_tanh_actions: bool = True
    alpha_xy: Optional[float] = 0.8
    alpha_yaw: Optional[float] = 1.572
    bbox_image_w: int = 384
    bbox_image_h: int = 384
    vision_feat_dim: int = 1536
    history: int = 31

    beta_qa: float = 1.0
    max_answer_length: int = 256

    freeze_proj: bool = False
    freeze_tvi: bool = False
    freeze_planner: bool = False
    freeze_act_token: bool = False

    use_refer: bool = True
    max_candidates: int = MAX_CANDIDATES
    cot_answer_max_tokens: int = 1
    beta_cot: float = 1.0
    beta_cot_qa: float = 1.0

    no_exist_token: str = NO_EXIST_TOKEN
    obj_token_fmt: str = OBJ_TOKEN_FMT
    cat_open_token: str = SEG_MARKER_TOKENS["cat_open"]
    cat_close_token: str = SEG_MARKER_TOKENS["cat_close"]
    vis_open_token: str = SEG_MARKER_TOKENS["vis_open"]
    vis_close_token: str = SEG_MARKER_TOKENS["vis_close"]
    reasoning_open_token: str = SEG_MARKER_TOKENS["reasoning_open"]
    reasoning_close_token: str = SEG_MARKER_TOKENS["reasoning_close"]

    add_refer_special_tokens: bool = True
    constrained_cot_ce: bool = True
    new_token_init: str = "mean"
    bbox_injection: str = "add"
    tvbi_inject_all_views: bool = False


class TVIEmbedder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.base_emb = nn.Embedding(1, d_model)
        self.yaw_proj = nn.Sequential(
            nn.Linear(2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.bbox_proj = nn.Sequential(
            nn.Linear(4, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def make_tvi_token(self, t_scalar: int, theta: float, device: Optional[torch.device] = None) -> torch.Tensor:
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        sincos = torch.tensor(
            [math.sin(theta), math.cos(theta)],
            dtype=next(self.yaw_proj.parameters()).dtype,
            device=device,
        )
        yaw_embedding = self.yaw_proj(sincos)
        t_tensor = torch.tensor(t_scalar, dtype=next(self.time_proj.parameters()).dtype, device=device)
        t_embedding = self.time_proj(t_tensor.unsqueeze(0))
        tok = self.base_emb.weight[0] + yaw_embedding + t_embedding
        return tok.to(device) if device is not None else tok

    def make_ti_token(self, t_scalar: int, device: Optional[torch.device] = None) -> torch.Tensor:
        t_tensor = torch.tensor(t_scalar, dtype=next(self.time_proj.parameters()).dtype, device=device)
        t_embedding = self.time_proj(t_tensor.unsqueeze(0))
        tok = self.base_emb.weight[0] + t_embedding
        return tok.to(device) if device is not None else tok

    def make_tvbi_token(self, t_scalar: int, theta: float, bbox, device: Optional[torch.device] = None) -> torch.Tensor:
        bbox_dtype = next(self.bbox_proj.parameters()).dtype
        bbox_in = bbox.to(dtype=bbox_dtype, device=device) if device is not None else bbox.to(dtype=bbox_dtype)
        bbox_embedding = self.bbox_proj(bbox_in)
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        sincos = torch.tensor(
            [math.sin(theta), math.cos(theta)],
            dtype=next(self.yaw_proj.parameters()).dtype,
            device=device,
        )
        yaw_embedding = self.yaw_proj(sincos)
        t_tensor = torch.tensor(t_scalar, dtype=next(self.time_proj.parameters()).dtype, device=device)
        t_embedding = self.time_proj(t_tensor.unsqueeze(0))
        tok = self.base_emb.weight[0] + yaw_embedding + t_embedding + bbox_embedding
        return tok.to(device) if device is not None else tok


class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class PlannerHead3L(nn.Module):
    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool = True):
        super().__init__()
        hid = d_model * 2
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, n_waypoints * action_dims),
        )
        self.nw = n_waypoints
        self.ad = action_dims
        self.use_tanh = use_tanh

    def forward(self, act_h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(act_h)
        if self.use_tanh:
            y = torch.tanh(y)
        return y.view(-1, self.nw, self.ad)



class ReferTrack(nn.Module):
    """ReferTrack policy. Same state_dict layout as the released checkpoint."""

    def __init__(self, cfg: ReferTrackConfig):
        super().__init__()
        self.cfg = cfg
        self.view_list = cfg.view_list or ["forward"]
        if not all(v in VIEW_YAWS for v in self.view_list):
            raise ValueError(f"Invalid view_list: {self.view_list}")

        attn_impl = "eager"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                attn_impl = "sdpa"
        print(f"[MODEL] Using attention implementation: {attn_impl}")

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=attn_impl,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if cfg.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            print(f"[LoRA] Applying LoRA: rank={cfg.lora_rank}, alpha={cfg.lora_alpha}")
            self.llm.config.use_cache = False
            self.llm = get_peft_model(
                self.llm,
                LoraConfig(
                    r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    task_type=TaskType.CAUSAL_LM, bias="none",
                ),
            )
        else:
            self.llm.requires_grad_(not cfg.freeze_llm)

        self.D = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size

        self.proj = CrossModalityProjector(int(cfg.vision_feat_dim), self.D)
        self.proj.requires_grad_(not cfg.freeze_proj)
        self.tvi = TVIEmbedder(self.D)
        self.tvi.requires_grad_(not cfg.freeze_tvi)

        self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token, std=0.02)
        self.act_token.requires_grad_(not cfg.freeze_act_token)

        self.action_dims = 3
        self.planner = PlannerHead3L(self.D, cfg.n_waypoints, self.action_dims, use_tanh=cfg.use_tanh_actions)
        self.planner.requires_grad_(not cfg.freeze_planner)

        alpha_vec = torch.ones(1, 1, self.action_dims)
        if cfg.alpha_xy is not None:
            alpha_vec[0, 0, 0] = cfg.alpha_xy
            alpha_vec[0, 0, 1] = cfg.alpha_xy
        if cfg.alpha_yaw is not None:
            alpha_vec[0, 0, 2] = cfg.alpha_yaw
        self.register_buffer("alpha_task", alpha_vec)

        if cfg.bbox_injection not in ("add", "concat"):
            raise ValueError(f"bbox_injection must be 'add' or 'concat', got {cfg.bbox_injection!r}")
        if cfg.add_refer_special_tokens:
            self._add_refer_special_tokens()
        self._register_refer_token_ids()
        print(f"[REFER] bbox_injection mode: {cfg.bbox_injection!r}")
        self.null_bbox_emb = nn.Parameter(torch.zeros(self.D))

    def _embed_text(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, max_length=512)
        tok = {k: v.to(device) for k, v in tok.items()}
        return self.llm.get_input_embeddings()(tok["input_ids"]), tok["attention_mask"]

    def _embed_text_from_ids(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        return self.llm.get_input_embeddings()(input_ids), attention_mask

    def _interleave_tvi(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        token_size: int,
        yaw_per_frame: Optional[torch.Tensor] = None,
        bbox_per_frame: Optional[torch.Tensor] = None,
        skip_yaw: bool = False,
        bbox_valid_per_frame: Optional[torch.Tensor] = None,
        inject_all_views: Optional[bool] = None,
    ) -> torch.Tensor:
        if inject_all_views is None:
            inject_all_views = bool(getattr(self.cfg, "tvbi_inject_all_views", False))
        if tokens.size(1) == 0:
            return tokens.new_zeros(tokens.size(0), 0, tokens.size(2))

        use_concat = (
            self.cfg.bbox_injection == "concat" and (not skip_yaw) and bbox_per_frame is not None
        )
        B, N, D = tokens.shape
        out_list = []
        num_views = 1 if skip_yaw else len(self.view_list)
        fwd_v_idx = self.view_list.index("forward") if "forward" in self.view_list else 0
        if not skip_yaw and yaw_per_frame is not None:
            yaw_per_token = torch.repeat_interleave(yaw_per_frame, repeats=token_size, dim=-1)
        else:
            yaw_per_token = None
        bbox_dtype = next(self.tvi.bbox_proj.parameters()).dtype

        for b in range(B):
            tb, xb = t_idx[b], tokens[b]
            items: List[torch.Tensor] = []
            i = 0
            while i < N:
                tcur = int(tb[i].item())
                for v_idx in range(num_views):
                    start = i + v_idx * token_size
                    end = start + token_size
                    if end > N:
                        break
                    if skip_yaw:
                        tok = self.tvi.make_ti_token(tcur, device=xb.device).unsqueeze(0)
                        items.append(tok)
                        items.append(xb[start:end])
                        continue
                    if yaw_per_token is not None:
                        theta = float(yaw_per_token[b, start].item())
                    else:
                        theta = VIEW_YAWS.get(self.view_list[v_idx], 0.0)
                    is_fwd = v_idx == fwd_v_idx
                    fv_idx = start // token_size
                    if inject_all_views and bbox_per_frame is not None:
                        if bbox_valid_per_frame is not None:
                            want_bbox = bool(bbox_valid_per_frame[b, fv_idx].item())
                        else:
                            want_bbox = is_fwd
                    else:
                        want_bbox = is_fwd and (bbox_per_frame is not None)

                    if use_concat:
                        tvi_tok = self.tvi.make_tvi_token(tcur, theta, device=xb.device).unsqueeze(0)
                        if want_bbox:
                            bbox_in = bbox_per_frame[b, fv_idx].to(dtype=bbox_dtype, device=xb.device)
                            items.extend([tvi_tok, self.tvi.bbox_proj(bbox_in).unsqueeze(0), xb[start:end]])
                        else:
                            items.extend([tvi_tok, xb[start:end]])
                    else:
                        if want_bbox:
                            tok = self.tvi.make_tvbi_token(
                                tcur, theta, bbox_per_frame[b, fv_idx], device=xb.device
                            ).unsqueeze(0)
                        else:
                            tok = self.tvi.make_tvi_token(tcur, theta, device=xb.device).unsqueeze(0)
                        items.append(tok)
                        items.append(xb[start:end])
                i += num_views * token_size
            out_list.append(torch.cat(items, dim=0) if items else xb.new_zeros(0, D))
        return torch.stack(out_list, dim=0)

    # ==================== Tokenizer vocabulary ====================

    def _build_new_token_list(self) -> List[str]:
        """<obj_1>..<obj_N>, <NO_EXIST>, then cat/vis/reasoning markers."""
        cfg = self.cfg
        cand_tokens = (
            [cfg.obj_token_fmt.format(i) for i in range(1, cfg.max_candidates + 1)]
            + [cfg.no_exist_token]
        )
        marker_tokens = [
            cfg.cat_open_token, cfg.cat_close_token,
            cfg.vis_open_token, cfg.vis_close_token,
            cfg.reasoning_open_token, cfg.reasoning_close_token,
        ]
        all_new = cand_tokens + marker_tokens
        assert len(all_new) == cfg.max_candidates + 1 + 6, (
            f"expected {cfg.max_candidates + 7} new tokens, got {len(all_new)}"
        )
        return all_new

    def _add_refer_special_tokens(self) -> None:
        """Add refer special tokens and resize the LLM embedding.

        Existing tokens (reload) make ``add_special_tokens`` return 0.
        New rows follow ``cfg.new_token_init`` (default mean + 0.01*std noise).
        Untied ``lm_head`` is written with the same init.
        """
        cfg = self.cfg
        new_tokens = self._build_new_token_list()
        expected = cfg.max_candidates + 7  # 20 + 1 + 6 = 27 @ max_candidates=20
        assert expected == REFER_NEW_SPECIAL_TOKENS_COUNT, (
            f"REFER_NEW_SPECIAL_TOKENS_COUNT={REFER_NEW_SPECIAL_TOKENS_COUNT} "
            f"mismatches cfg.max_candidates={cfg.max_candidates} (→ expected={expected})"
        )

        vocab_before = len(self.tokenizer)
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )

        emb_layer = self.llm.get_input_embeddings()
        old_emb_size = emb_layer.weight.size(0)

        if added > 0:
            old_mean = emb_layer.weight.data.mean(dim=0).clone()
            old_std = emb_layer.weight.data.std().item()

            self.llm.resize_token_embeddings(len(self.tokenizer))
            new_emb_layer = self.llm.get_input_embeddings()
            new_emb_size = new_emb_layer.weight.size(0)

            new_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
            with torch.no_grad():
                if cfg.new_token_init == "mean":
                    for tid in new_ids:
                        noise = torch.randn_like(old_mean) * (old_std * 0.01)
                        new_emb_layer.weight.data[tid] = old_mean + noise
                elif cfg.new_token_init == "zero":
                    for tid in new_ids:
                        new_emb_layer.weight.data[tid].zero_()
                elif cfg.new_token_init == "random":
                    # resample from N(0, old_std) instead of zeros
                    for tid in new_ids:
                        new_emb_layer.weight.data[tid] = torch.randn_like(old_mean) * old_std
                else:
                    raise ValueError(f"Unknown new_token_init: {cfg.new_token_init}")

                lm_head = self.llm.get_output_embeddings()
                if lm_head is not None and lm_head.weight.data_ptr() != new_emb_layer.weight.data_ptr():
                    if cfg.new_token_init == "mean":
                        old_out = lm_head.weight.data[:old_emb_size].mean(dim=0).clone()
                        out_std = lm_head.weight.data[:old_emb_size].std().item()
                        for tid in new_ids:
                            noise = torch.randn_like(old_out) * (out_std * 0.01)
                            lm_head.weight.data[tid] = old_out + noise
                    elif cfg.new_token_init == "zero":
                        for tid in new_ids:
                            lm_head.weight.data[tid].zero_()
                    elif cfg.new_token_init == "random":
                        old_out = lm_head.weight.data[:old_emb_size].mean(dim=0).clone()
                        out_std = lm_head.weight.data[:old_emb_size].std().item()
                        for tid in new_ids:
                            lm_head.weight.data[tid] = torch.randn_like(old_out) * out_std

            if True:
                print(
                    f"[REFER] tokenizer.add_special_tokens: added={added} new tokens, "
                    f"vocab {vocab_before} → {len(self.tokenizer)}, "
                    f"embedding resized {old_emb_size} → {new_emb_size}, init={cfg.new_token_init}"
                )
        else:
            if True:
                print(
                    f"[REFER] tokenizer already contains all {len(new_tokens)} refer special tokens; "
                    f"skip resize (vocab={vocab_before})"
                )

    def _register_refer_token_ids(self) -> None:
        """Register refer token-id buffers.

        ``cand_token_ids`` is ``[<obj_1>, ..., <obj_N>, <NO_EXIST>]``, one per catalog slot.
        Segment markers are scalar LongTensors for ``_marker_emb``.
        """
        cfg = self.cfg
        cand_tokens = (
            [cfg.obj_token_fmt.format(i) for i in range(1, cfg.max_candidates + 1)]
            + [cfg.no_exist_token]
        )
        cand_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(t) for t in cand_tokens],
            dtype=torch.long,
        )
        assert (cand_ids >= 0).all(), (
            f"some refer candidate tokens not in tokenizer: "
            f"{list(zip(cand_tokens, cand_ids.tolist()))}"
        )
        self.register_buffer("cand_token_ids", cand_ids, persistent=False)

        def _id(tok: str) -> torch.Tensor:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            assert tid is not None and tid >= 0, f"token {tok!r} not in tokenizer"
            return torch.tensor(tid, dtype=torch.long)

        self.register_buffer("cat_open_id",        _id(cfg.cat_open_token),        persistent=False)
        self.register_buffer("cat_close_id",       _id(cfg.cat_close_token),       persistent=False)
        self.register_buffer("vis_open_id",        _id(cfg.vis_open_token),        persistent=False)
        self.register_buffer("vis_close_id",       _id(cfg.vis_close_token),       persistent=False)
        self.register_buffer("reasoning_open_id",  _id(cfg.reasoning_open_token),  persistent=False)
        self.register_buffer("reasoning_close_id", _id(cfg.reasoning_close_token), persistent=False)

        print(
            f"[REFER] registered cand_token_ids({cand_ids.numel()}): "
            f"first={cand_ids[0].item()} last={cand_ids[-1].item()} "
            f"<NO_EXIST>={cand_ids[-1].item()}"
        )
        print(
            f"[REFER] markers: cat={int(self.cat_open_id)}/{int(self.cat_close_id)} "
            f"vis={int(self.vis_open_id)}/{int(self.vis_close_id)} "
            f"reasoning={int(self.reasoning_open_id)}/{int(self.reasoning_close_id)}"
        )

    # ==================== Catalog encoding ====================

    def _embed_candidate_catalog(
        self,
        cand_bbox: torch.Tensor,        # (B, CAT_LEN, 4) or (B, CAT_LEN, V, 4) normalized [0,1]
        cand_slot_valid: torch.Tensor,  # (B, CAT_LEN) bool
        device: torch.device,
        cand_slot_valid_per_view: Optional[torch.Tensor] = None,  # (B, CAT_LEN, V) optional
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build catalog embeddings: each slot is ``[<obj_k>, bbox_v0, ..., bbox_v{V-1}]``.

        * ``cand_bbox.dim() == 3``: V=1, length ``2*CAT_LEN``
        * ``cand_bbox.dim() == 4``: length ``(1+V)*CAT_LEN``

        The last slot is virtual NO_EXIST ``[0,0,0,0]``. Optional
        ``cand_slot_valid_per_view`` masks bbox tokens per (slot, view);
        ``<obj_k>`` still follows the slot-level mask.
        """
        cfg = self.cfg

        # (B, CAT_LEN, 4) -> (B, CAT_LEN, 1, 4)
        if cand_bbox.dim() == 3:
            cand_bbox = cand_bbox.unsqueeze(2)              # (B, CAT_LEN, 1, 4)
        elif cand_bbox.dim() != 4:
            raise ValueError(
                f"cand_bbox must have 3 or 4 dims, got shape {tuple(cand_bbox.shape)}"
            )
        B, CAT_LEN, V, bbox_dim = cand_bbox.shape
        assert bbox_dim == 4, f"cand_bbox last dim must be 4, got {bbox_dim}"
        assert CAT_LEN == cfg.max_candidates + 1, (
            f"CAT_LEN must be max_candidates+1={cfg.max_candidates + 1}, got {CAT_LEN}"
        )
        assert self.cand_token_ids.numel() == CAT_LEN, (
            f"cand_token_ids length {self.cand_token_ids.numel()} != CAT_LEN {CAT_LEN}"
        )
        if cand_slot_valid_per_view is not None:
            pv_shape = tuple(cand_slot_valid_per_view.shape)
            assert pv_shape == (B, CAT_LEN, V), (
                f"cand_slot_valid_per_view must be (B, CAT_LEN, V)={B,CAT_LEN,V}, "
                f"got {pv_shape}"
            )

        # Shared TVI.bbox_proj. Invalid (slot, view) cells use null_bbox_emb
        # so the projector only sees real boxes; tokens stay for fixed length.
        bbox_dtype = next(self.tvi.bbox_proj.parameters()).dtype
        bbox_in_flat = cand_bbox.reshape(B * CAT_LEN * V, 4).to(
            device=device, dtype=bbox_dtype
        )

        if cand_slot_valid_per_view is not None:
            # per-(slot, view) proj vs null
            view_valid = (
                cand_slot_valid_per_view.to(device=device, dtype=torch.bool)
                & cand_slot_valid.to(device=device, dtype=torch.bool).unsqueeze(-1)
            )                                                  # (B, CAT_LEN, V)
        else:
            # No per-view mask: treat every cell as valid.
            view_valid = cand_slot_valid.to(device=device, dtype=torch.bool)\
                                        .unsqueeze(-1)\
                                        .expand(B, CAT_LEN, V)

        flat_valid = view_valid.reshape(-1)                     # (B*CAT_LEN*V,)

        # Default null; overwrite valid cells with bbox_proj.
        null_emb = self.null_bbox_emb.to(device=device, dtype=bbox_dtype)        # (D,)
        bbox_emb_flat = null_emb.unsqueeze(0).expand(flat_valid.numel(), -1).clone()

        if bool(flat_valid.any().item()):
            real_in = bbox_in_flat[flat_valid]                  # (N_valid, 4)
            real_out = self.tvi.bbox_proj(real_in)               # (N_valid, D)
            bbox_emb_flat[flat_valid] = real_out

        bbox_emb = bbox_emb_flat.view(B, CAT_LEN, V, -1)         # (B, CAT_LEN, V, D)

        # obj / NO_EXIST token per slot
        emb_layer = self.llm.get_input_embeddings()
        tok_ids = self.cand_token_ids.to(device).unsqueeze(0).expand(B, -1)  # (B, CAT_LEN)
        tok_emb = emb_layer(tok_ids).unsqueeze(2)                             # (B, CAT_LEN, 1, D)

        # [<obj_k>, bbox_v0, ..., bbox_v{V-1}] per slot
        cat = torch.cat([tok_emb, bbox_emb.to(tok_emb.dtype)], dim=2)         # (B, CAT_LEN, 1+V, D)
        cat = cat.reshape(B, (1 + V) * CAT_LEN, -1)                           # (B, (1+V)*CAT_LEN, D)

        # 4) attention mask
        slot_mask = cand_slot_valid.to(device)                                # (B, CAT_LEN)
        if cand_slot_valid_per_view is not None:
            # <obj_k> follows the slot; bbox_vk = slot_valid AND view_valid
            view_mask = cand_slot_valid_per_view.to(device=device, dtype=torch.bool)
            bbox_token_mask = view_mask & slot_mask.unsqueeze(-1)              # (B, CAT_LEN, V)
            per_token_mask = torch.cat(
                [slot_mask.unsqueeze(-1), bbox_token_mask], dim=-1
            )                                                                  # (B, CAT_LEN, 1+V)
            cat_mask = per_token_mask.reshape(B, (1 + V) * CAT_LEN).long()
        else:
            # Slot-level mask: all (1+V) tokens on or off together.
            cat_mask = slot_mask.unsqueeze(-1).expand(-1, -1, 1 + V)\
                                .reshape(B, (1 + V) * CAT_LEN).long()
        return cat, cat_mask

    def _marker_emb(self, token_id: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Embed one token id and expand to (B, 1, D)."""
        emb_layer = self.llm.get_input_embeddings()
        ids = token_id.to(device).view(1, 1).expand(batch_size, 1)
        return emb_layer(ids)  # (B, 1, D)


    @torch.no_grad()
    def inference_refer_navigation(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        cand_bbox: torch.Tensor,
        cand_slot_valid: torch.Tensor,
        bbox_hist: torch.Tensor,
        bbox_curr: Optional[torch.Tensor] = None,
        bbox_hist_valid: Optional[torch.Tensor] = None,
        instructions: Optional[List[str]] = None,
        instruction_input_ids: Optional[torch.Tensor] = None,
        instruction_attention_mask: Optional[torch.Tensor] = None,
        yaw_hist: Optional[torch.Tensor] = None,
        yaw_curr: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Two-pass Refer-Nav inference with KV-cache reuse.

        Pass 1 runs through ``<reasoning_open>`` and constrained-argmaxes the
        catalog slot. Pass 2 feeds ``[<PRED_SLOT>, </reasoning>, <act>]`` and
        reads the planner from the act hidden.
        """
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)

        if alpha is not None:
            valid_shape = (
                (alpha.dim() == 2 and alpha.size(-1) == self.action_dims)
                or (alpha.dim() == 3 and alpha.size(1) == 1 and alpha.size(-1) == self.action_dims)
            )
            if not valid_shape:
                raise ValueError(
                    f"alpha must be (B, {self.action_dims}) or (B, 1, {self.action_dims}), "
                    f"got {tuple(alpha.shape)}"
                )

        proj_dtype = next(self.proj.parameters()).dtype
        vis_c = self.proj(coarse_tokens.to(device=device, dtype=proj_dtype))
        vis_f_proj_raw = self.proj(fine_tokens.to(device=device, dtype=proj_dtype))

        vis_c = self._interleave_tvi(
            vis_c, coarse_tidx.to(device), token_size=4,
            yaw_per_frame=yaw_hist,
            bbox_per_frame=bbox_hist.to(device) if bbox_hist is not None else None,
            bbox_valid_per_frame=bbox_hist_valid.to(device) if bbox_hist_valid is not None else None,
        )
        vis_f = self._interleave_tvi(
            vis_f_proj_raw, fine_tidx.to(device), token_size=64,
            yaw_per_frame=yaw_curr,
            bbox_per_frame=None,
        )

        if instruction_input_ids is not None and instruction_attention_mask is not None:
            txt_emb, txt_mask = self._embed_text_from_ids(
                instruction_input_ids, instruction_attention_mask, device
            )
        else:
            if instructions is None:
                raise ValueError(
                    "Either instructions or (instruction_input_ids, instruction_attention_mask) must be provided"
                )
            txt_emb, txt_mask = self._embed_text(instructions, device)

        cat_emb, cat_mask = self._embed_candidate_catalog(cand_bbox, cand_slot_valid, device)
        cat_open_e = self._marker_emb(self.cat_open_id, B, device)
        cat_close_e = self._marker_emb(self.cat_close_id, B, device)
        vis_open_e = self._marker_emb(self.vis_open_id, B, device)
        vis_close_e = self._marker_emb(self.vis_close_id, B, device)
        reas_open_e = self._marker_emb(self.reasoning_open_id, B, device)
        reas_close_e = self._marker_emb(self.reasoning_close_id, B, device)

        llm_dtype = self.llm.dtype
        context_seq = torch.cat([
            txt_emb,
            cat_open_e, cat_emb, cat_close_e,
            vis_open_e, vis_c, vis_f, vis_close_e,
            reas_open_e,
        ], dim=1).to(llm_dtype)

        ones = lambda n: torch.ones(B, n, dtype=torch.long, device=device)
        context_mask = torch.cat([
            txt_mask.to(device),
            ones(1), cat_mask.to(device), ones(1),
            ones(1), ones(vis_c.size(1) + vis_f.size(1)), ones(1),
            ones(1),
        ], dim=1)

        out_ctx = self.llm(
            inputs_embeds=context_seq,
            attention_mask=context_mask,
            use_cache=True,
            output_hidden_states=False,
        )
        past_key_values = out_ctx.past_key_values

        next_token_logits = out_ctx.logits[:, -1, :]
        cand_ids = self.cand_token_ids.to(device)
        sub_logits = next_token_logits.index_select(dim=-1, index=cand_ids)
        sub_logits = sub_logits.masked_fill(~cand_slot_valid.to(device), float("-inf"))
        pred_slot = sub_logits.argmax(dim=-1)
        pred_token_ids = cand_ids[pred_slot]

        pred_slot_emb = self.llm.get_input_embeddings()(pred_token_ids.view(B, 1))
        act = self.act_token.expand(B, 1, -1).to(llm_dtype)
        step2_seq = torch.cat([pred_slot_emb.to(llm_dtype), reas_close_e, act], dim=1)
        step2_mask = torch.cat([context_mask, ones(step2_seq.size(1))], dim=1)

        out_step2 = self.llm(
            inputs_embeds=step2_seq,
            attention_mask=step2_mask,
            past_key_values=past_key_values,
            use_cache=False,
            output_hidden_states=True,
        )
        last_hs = out_step2.hidden_states[-1]
        h_act = last_hs[:, -1, :].to(next(self.planner.parameters()).dtype)
        a_hat = self.planner(h_act)

        if alpha is not None:
            a_alpha = alpha.to(a_hat.device, a_hat.dtype)
            if a_alpha.dim() == 2:
                a_alpha = a_alpha.unsqueeze(1)
            tau_pred = a_hat * a_alpha
        else:
            tau_pred = a_hat * self.alpha_task

        return {
            "trajectory": tau_pred.float(),
            "pred_slot": pred_slot,
            "sub_logits": sub_logits.detach(),
            "pred_token_ids": pred_token_ids,
        }
