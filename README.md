<h1 align="center">ReferTrack: Referring Then Tracking for Embodied Visual Tracking</h1>

<p align="center">
    <a href="https://medlartea.github.io/">Hanjing Ye</a><sup>1,2</sup>
    &nbsp;
    <a>Tianle Zeng</a><sup>1</sup>
    &nbsp;
    <a href="https://jzhzhang.github.io/">Jiazhao Zhang</a><sup>3</sup>
    &nbsp;
    <a href="https://wsakobe.github.io/">Shaoan Wang</a><sup>3</sup>
    &nbsp;
    <a>Zibo Zhang</a><sup>4</sup>
    <br>
    <a href="https://situ-weixi.github.io/">Weisi Situ</a><sup>1</sup>
    &nbsp;
    <a href="https://yuchen2199.github.io/">Yuchen Zhou</a><sup>2</sup>
    &nbsp;
    <a href="https://ygling2008.github.io/">Yonggen Ling</a><sup>2,4*</sup>
    &nbsp;
    <a href="https://scholar.google.com/citations?user=J7UkpAIAAAAJ&hl=en">Hong Zhang</a><sup>1*</sup>
</p>

<p align="center">
    <sup>1</sup>RCV Laboratory, SUSTech
    &nbsp;
    <sup>2</sup>Tencent Robotics X
    <br>
    <sup>3</sup>Peking University
    &nbsp;
    <sup>4</sup>Futian Laboratory
</p>

<p align="center">
    <a href="https://medlartea.github.io/referTrack/">Project Page</a>
    &nbsp;|&nbsp;
    <a href="https://arxiv.org/abs/2607.20061">arXiv</a>
    &nbsp;|&nbsp;
    <a href="https://huggingface.co/hjyeee/ReferTrack-Qwen3-4B">Hugging Face</a>
    &nbsp;|&nbsp;
    <a href="https://youtu.be/CP7h-tWWABU">Video</a>
</p>

## Overview

**_ReferTrack_** is a *referring-then-tracking* paradigm for embodied visual tracking that first grounds a language-described target to an image-space bounding box and then decodes tracking waypoints from this decision, using temporal-viewpoint-bbox indicator (TVBI) tokens to inject previously selected bounding boxes into the visual history and preserve target motion cues over time, achieving state-of-the-art single-view performance on EVT-Bench with robust sim-to-real transfer to legged and humanoid robots.

<p align="center">
    <img src="assets/method.png" alt="ReferTrack method overview" width="95%">
</p>

---

## 📢 News

* **[17/08]** Evaluation code and the [ReferTrack-Qwen3-4B](https://huggingface.co/hjyeee/ReferTrack-Qwen3-4B) checkpoint are released.
* **[23/07]** Paper is available on [arXiv](https://arxiv.org/abs/2607.20061).

---

## Comparison with recent VLA policies

Each cell is SR↑ / TR↑ / CR↓. The main comparison is the single-view group; multi-camera results are external references only, not a ranking.

| Method | Size | RL | STT | DT | AT |
| --- | :---: | :---: | :---: | :---: | :---: |
| *Multi-camera references (three or four cameras)* | | | | | |
| ABot-N0<sup>∗</sup> | 4B | – | 86.9 / 87.6 / 8.54 | 66.7 / 75.4 / 11.6 | 67.3 / 79.5 / 7.05 |
| NavFoM<sup>∗</sup> | 7B | – | 88.4 / 80.7 / – | 62.0 / 67.9 / – | – |
| CoMaTrack<sup>∗</sup> | 3B | ✓ | 92.1 / 90.3 / 0.9 | 74.2 / 80.5 / 2.1 | 57.5 / 73.4 / 12.0 |
| TrackVLA++ | 7B | – | 90.9 / 82.7 / 1.50 | 74.0 / 73.7 / 3.51 | 55.9 / 63.8 / 15.1 |
| *Single-view (forward camera only)* | | | | | |
| Uni-NaVid<sup>∗</sup> | 7B | – | 53.3 / 67.2 / 12.6 | 31.9 / 50.1 / 21.3 | 15.8 / 41.5 / 26.5 |
| NavFoM<sup>∗</sup> | 7B | – | 85.0 / 80.5 / – | 61.4 / 68.2 / – | – |
| VLingNav<sup>∗</sup> | 7B | ✓ | 88.4 / 81.2 / 2.1 | 67.7 / 73.5 / 5.5 | – |
| TrackVLA | 7B | – | 85.1 / 78.6 / 1.7 | 57.6 / 63.2 / 5.8 | 50.2 / 63.7 / 17.1 |
| TrackVLA++ | 7B | – | 86.0 / 81.0 / 2.10 | 66.5 / 68.8 / 4.71 | 51.2 / 63.4 / 15.9 |
| **ReferTrack (ours)** | **4B** | – | **89.4 / 92.5 / 1.6** | **73.3 / 81.8 / 7.6** | **74.1 / 85.7 / 7.7** |

<sup>∗</sup> co-trained with general navigation data. All baseline numbers are quoted from Table 1 of the [paper](https://arxiv.org/abs/2607.20061), which in turn takes them from the original publications.

---

## 1. Installation

Habitat-Sim **0.3.1** (with Bullet) is required, same as [TrackVLA](https://github.com/wsakobe/TrackVLA) / EVT-Bench.

### 1.1 NVIDIA OpenGL

Match `xxx` to the driver version from `nvidia-smi` (e.g. `570`):

```bash
sudo apt-get install libnvidia-gl-xxx
sudo apt-get install --reinstall libglvnd-dev
```

### 1.2 Conda env + Habitat-Sim + PyTorch

```bash
conda create -n refertrack python=3.9 cmake=3.14.0
conda activate refertrack
conda install habitat-sim==0.3.1 withbullet -c conda-forge -c aihabitat
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

### 1.3 Clone and install this repo

```bash
git clone https://github.com/MedlarTea/referTrack.git
cd referTrack
pip install -e habitat-lab
pip install -e .
pip install flash-attn --no-build-isolation   # optional; falls back to SDPA
```

Video eval additionally needs `ffmpeg` on `PATH`.

---

## 2. Habitat / EVT-Bench assets

Scene assets follow the [TrackVLA](https://github.com/wsakobe/TrackVLA) / EVT-Bench layout. Accept the respective licenses before downloading. **Video-only inference can skip this section.**

This repo already ships:

- EVT-Bench episode files: `data/datasets/track/{DT,STT,AT}/val/val.json.gz`
- Spot robot assets: `data/robots/hab_spot_arm/`
- `humanoid_infos.json` at the repo root (Habitat Track reads it)

You still need **HM3D**, **MP3D**, and **humanoid meshes**.

### 2.1 HM3D + MP3D scenes

Request access via Habitat: [HM3D](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) and [MP3D](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset). Extract them under `data/scene_datasets`:

```
data/
  scene_datasets/
    hm3d/
      train/...
      val/...
      minival/...
    mp3d/
      1LXtFkjw3qL/...
      ...
```

```bash
mv /path/to/hm3d data/scene_datasets/
mv /path/to/mp3d data/scene_datasets/
```

Keep the directory names lowercase (`hm3d`, `mp3d`). DT / STT / AT val episodes reference these scenes; `TrackEnv` loads each episode's `scene_id` from `data/scene_datasets`.

### 2.2 Humanoid avatars

Download `humanoids.zip` from the TrackVLA [Google Drive](https://drive.google.com/file/d/1aE_wyvPqvOuVmF8px2vTO3trr70DKf1l/view) (same file as EVT-Bench), then:

```bash
unzip humanoids.zip -d data/
```

Confirm `data/humanoids/humanoid_data/` exists (e.g. `female_2/female_2.urdf`).

### 2.3 Optional: HSSD-HAB

The Track yaml lists `data/hssd-data/hssd-hab.scene_dataset_config.json`. EVT-Bench val episodes overwrite this with the episode scene (HM3D / MP3D), so **HSSD is not required** for the released DT / STT / AT eval. If you still want it:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="hssd/hssd-hab", repo_type="dataset",
                  local_dir="data/hssd-data", local_dir_use_symlinks=False)
PY
```

---

## 3. Download weights

Released checkpoint: [hjyeee/ReferTrack-Qwen3-4B](https://huggingface.co/hjyeee/ReferTrack-Qwen3-4B) (~7.6 GB).

```
data/logs/ReferTrack-Qwen3-4B/
  ├── refertrack_qwen3_4b.pt
  └── model_config.json
```

### 3.1 Qwen3-4B backbone

Tokenizer configs are already in `LLM_hf/qwen3-4b/`. Download the weight shards:

```bash
cd LLM_hf/
python download_llm_hf.py
cd ..
```

If `huggingface.co` is unreachable, prefix with `HF_ENDPOINT=https://hf-mirror.com`.

### 3.2 ReferTrack checkpoint

```bash
bash scripts/eval/download_ckpt.sh
```

This pulls `refertrack_qwen3_4b.pt` and `model_config.json` from Hugging Face. Same mirror env if needed:

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/eval/download_ckpt.sh
```

### 3.3 Vision towers and YOLO

Pulled automatically on the first Habitat / video run:

- DINOv3: `facebook/dinov3-vits16-pretrain-lvd1689m`
- SigLIP: `google/siglip-so400m-patch14-384`
- YOLO: `yolo11x.pt` (Ultralytics)

To use local copies instead of Hugging Face, place them at the repo root as `facebook/dinov3-vits16-pretrain-lvd1689m/` and `google/siglip-so400m-patch14-384/`, or set `DINOV3_MODEL_PATH` to the DINOv3 folder.

---

## 4. Habitat evaluation (EVT-Bench)

Closed-loop Track eval: online YOLO + ByteTrack, then ReferTrack (catalog → CoT slot → planner waypoints).

```bash
conda activate refertrack

# full DT eval (example: 8 GPUs)
CHUNKS=30 NUM_PARALLEL=8 bash scripts/eval/eval_sim_refer_base.sh
```

Default split is **DT**. For STT / AT, edit `SPLITS` in `scripts/eval/eval_sim_refer_base.sh`:

```bash
SPLITS=("stt")          # or ("at"), or ("dt" "stt" "at")
```

Optional:

| Env | Effect |
| --- | --- |
| `SAVE_VIDEO=1` | Write per-episode mp4 (slow) |
| `YOLO_MODEL=yolo11x.pt` | Detector weights (default) |
| `FALLBACK_STOP=1` | Force stop when CoT predicts `NO_EXIST` |

Results go to:

```
data/logs/ReferTrack-Qwen3-4B/eval_sim_refer_refertrack_qwen3_4b/<split>/
```

Summarize SR / TR / CR (and refer metrics):

```bash
python -m referTrack.tool.print_eval_result \
  --result-dir data/logs/ReferTrack-Qwen3-4B/eval_sim_refer_refertrack_qwen3_4b \
  --splits dt
```

Use `--splits dt stt at` after running those splits.

---

## 5. Video evaluation (no Habitat)

Forward RGB video or an image folder + a language instruction. Same checkpoint as Habitat eval. `CKPT_PATH` defaults to `data/logs/ReferTrack-Qwen3-4B/refertrack_qwen3_4b.pt`.

```bash
conda activate refertrack

VIDEO_FORWARD=/path/to/your.mp4 \
INSTRUCTION="Walk after the person with a white and black armored suit." \
OUT_DIR=./video_out \
SAVE_FRAMES=1 COPY_RAW=1 \
bash scripts/eval/eval_refer_video.sh
```

Useful knobs: `FRAME_STRIDE` (e.g. `4` on 30 fps video), `MAX_FRAMES` (smoke), `FPS_OUT` (default 8), `DEVICE=cuda`.

Outputs under `OUT_DIR`:

| File | Content |
| --- | --- |
| `<name>.mp4` | Overlay: gray candidates, red Pred box, trajectory |
| `<name>__raw.mp4` | Copy of the input (`COPY_RAW=1`) |
| `<name>_frames/*.jpg` | Per-frame jpg (`SAVE_FRAMES=1`) |
| `<name>_preds.jsonl` | Per-frame slot / bbox / trajectory |
| `<name>_summary.json` | Run summary |

This release is **single-view** (`view_list=['forward']`); only `VIDEO_FORWARD` is required.

---

## 6. What this tree contains

| Path | Role |
| --- | --- |
| `scripts/eval/eval_sim_refer_base.sh` | Habitat closed-loop eval launcher |
| `scripts/eval/eval_refer_video.sh` | Offline video inference + visualization |
| `scripts/eval/download_ckpt.sh` | Download [ReferTrack-Qwen3-4B](https://huggingface.co/hjyeee/ReferTrack-Qwen3-4B) |
| `referTrack/eval/run_eval_refer_sim.py` | Habitat eval entry |
| `referTrack/eval/run_eval_refer_video.py` | Video eval entry |
| `referTrack/baseline/trained_agent_refer.py` | ReferAgent (YOLO + ByteTrack + CoT + planner) |
| `referTrack/model/referTrack.py` | ReferTrack model |
| `referTrack/dataset/evt_bench/` | Habitat Track registrations |
| `habitat-lab/` | Habitat-Lab with Track task configs |
| `LLM_hf/qwen3-4b/` | Qwen3-4B tokenizer (weights via `download_llm_hf.py`) |

---

## TODO List

* [x] Release model checkpoints and evaluation code.
* [ ] Release the dataset.
* [ ] Release the training code.
* [ ] Release the data engine.

---

## Acknowledgment

This codebase is built on [OmTrackVLA](https://github.com/om-ai-lab/OmTrackVLA). We thank the OmTrackVLA authors for the open-source Tracking-VLA infrastructure.

Habitat evaluation follows the [EVT-Bench](https://github.com/wsakobe/TrackVLA) protocol and scene / humanoid layout from [TrackVLA](https://github.com/wsakobe/TrackVLA). We thank the TrackVLA and Habitat teams for the benchmark and simulator.

---

## Citation

If you use this code or the released checkpoint, please cite:

```bibtex
@article{ye2026refertrack,
  title={ReferTrack: Referring Then Tracking for Embodied Visual Tracking},
  author={Ye, Hanjing and Zeng, Tianle and Zhang, Jiazhao and Wang, Shaoan and Zhang, Zibo and Situ, Weixi and Zhou, Yuchen and Ling, Yonggen and Zhang, Hong},
  journal={arXiv preprint arXiv:2607.20061},
  year={2026},
  url={https://arxiv.org/abs/2607.20061}
}
```

---

## Contact

Questions and issues: open a GitHub issue, or see the [project page](https://medlartea.github.io/referTrack/).

This project is released under the Apache-2.0 license. HM3D, MP3D, HSSD, and humanoid assets remain under their original licenses.
