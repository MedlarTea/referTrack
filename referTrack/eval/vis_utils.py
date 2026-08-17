"""Lightweight drawing helpers for ReferTrack video visualization."""
from __future__ import annotations

import os
import os.path as osp
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _try_load_font(size: int = 16):
    for name in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


_FONT_NORMAL = None
_FONT_SMALL = None


def _get_fonts():
    global _FONT_NORMAL, _FONT_SMALL
    if _FONT_NORMAL is None:
        _FONT_NORMAL = _try_load_font(16)
        _FONT_SMALL = _try_load_font(12)
    return _FONT_NORMAL, _FONT_SMALL


def _draw_bbox_rect(
    draw: ImageDraw.ImageDraw,
    bbox_norm: np.ndarray,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    width: int = 3,
    label: Optional[str] = None,
    label_bg: bool = True,
):
    x1 = int(bbox_norm[0] * w)
    y1 = int(bbox_norm[1] * h)
    x2 = int(bbox_norm[2] * w)
    y2 = int(bbox_norm[3] * h)
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        _, font_s = _get_fonts()
        tx, ty = x1, max(0, y1 - 14)
        if label_bg:
            bbox_t = draw.textbbox((tx, ty), label, font=font_s)
            draw.rectangle(bbox_t, fill=color)
            draw.text((tx, ty), label, fill=(0, 0, 0), font=font_s)
        else:
            draw.text((tx, ty), label, fill=color, font=font_s)


def _draw_traj(
    draw: ImageDraw.ImageDraw,
    traj: np.ndarray,
    base_xy: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    outline_width: int = 10,
    line_width: int = 6,
):
    bx, by = base_xy
    pts: List[Tuple[int, int]] = []
    n = min(traj.shape[0], 64)
    for i in range(n):
        x, y = float(traj[i, 0]), float(traj[i, 1])
        px = bx - int(y * scale)
        py = by - int(x * scale)
        pts.append((px, py))
    for i in range(1, len(pts)):
        draw.line([pts[i - 1], pts[i]], fill=outline_color, width=outline_width)
    for i in range(1, len(pts)):
        draw.line([pts[i - 1], pts[i]], fill=color, width=line_width)
    return pts


def _draw_text_panel(
    img: Image.Image,
    lines: List[str],
    pad: int = 8,
    bg_alpha: int = 200,
    position: str = "top",
) -> Image.Image:
    font_n, _ = _get_fonts()
    w, h = img.size
    dummy_draw = ImageDraw.Draw(img)
    line_heights = []
    max_w = 0
    for ln in lines:
        bbox = dummy_draw.textbbox((0, 0), ln, font=font_n)
        line_heights.append(bbox[3] - bbox[1] + 4)
        max_w = max(max_w, bbox[2] - bbox[0])
    panel_h = sum(line_heights) + 2 * pad
    panel_w = min(w, max_w + 2 * pad)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    o_draw = ImageDraw.Draw(overlay)
    y0 = 0 if position == "top" else h - panel_h
    o_draw.rectangle([0, y0, panel_w, y0 + panel_h], fill=(0, 0, 0, bg_alpha))
    cy = y0 + pad
    for ln, lh in zip(lines, line_heights):
        o_draw.text((pad, cy), ln, fill=(255, 255, 255, 255), font=font_n)
        cy += lh
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _slot_to_label(slot: int, n_max: int) -> str:
    if slot == n_max:
        return "<NO_EXIST>"
    return f"<obj_{slot + 1}>"


def _pad_to_even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ph = h % 2
    pw = w % 2
    if ph == 0 and pw == 0:
        return img
    new = np.zeros((h + ph, w + pw, img.shape[2]), dtype=img.dtype)
    new[:h, :w] = img
    return new


def _write_mp4(path: str, frames_rgb: List[np.ndarray], fps: int = 8) -> None:
    if not frames_rgb:
        return
    frames_even = [_pad_to_even(f) for f in frames_rgb]
    h, w = frames_even[0].shape[:2]
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="refer_vis_")
    try:
        import cv2  # noqa: WPS433

        for i, f in enumerate(frames_even):
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                osp.join(tmp_dir, f"frame_{i:05d}.jpg"),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tmp_mp4 = tf.name
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", osp.join(tmp_dir, "frame_%05d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-movflags", "+faststart",
            tmp_mp4,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if osp.getsize(tmp_mp4) < 1024:
                raise RuntimeError(f"ffmpeg output too small ({osp.getsize(tmp_mp4)} B)")
            shutil.move(tmp_mp4, path)
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            try:
                os.unlink(tmp_mp4)
            except Exception:
                pass
            print(f"  [mp4] ffmpeg failed ({e}); trying cv2 fallback...")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        import cv2  # noqa: WPS433

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tmp_path = tf.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open tmp")
        for f in frames_even:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        if osp.getsize(tmp_path) < 1024:
            raise RuntimeError(f"cv2 mp4 too small ({osp.getsize(tmp_path)} B)")
        shutil.move(tmp_path, path)
        return
    except Exception as e:
        print(f"  [mp4] cv2 writer failed ({e}); writing GIF instead")
        try:
            os.unlink(tmp_path)  # type: ignore[name-defined]
        except Exception:
            pass

    try:
        import imageio.v2 as iio  # noqa: WPS433

        gif_path = osp.splitext(path)[0] + ".gif"
        iio.mimsave(gif_path, frames_even, duration=1.0 / max(1, fps))
        print(f"  [mp4] saved GIF → {gif_path}")
    except Exception as e:
        print(f"  [mp4] GIF writer also failed ({e}); giving up")
