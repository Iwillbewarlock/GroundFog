"""temporal_demo_post.py <dir> [--blend 0.2] [--clamp 1] : accumulate the noisy frames of temporal_demo.py and compare.

history_k = lerp(clamp(reproject(history_{k-1})), raw_k, blend), with the history reprojected for the camera's yaw (a
pure rotation about the eye: pixel -> ray in the previous camera -> pixel now, bilinear) and clamped per channel to the
min/max of raw_k's 3x3 neighbourhood (the plugin's TemporalClamp). Pixels whose ray left the previous view take raw_k.
Writes acc_NNN.png, side_NNN.png (raw | accumulated | clean reference) and prints per segment (still, pan):
  flicker = mean |f(t+1) - 2 f(t) + f(t-1)| in the bush region (frame-to-frame jumps),
  error   = mean |f - reference| (noise left, and trails while turning).
Needs numpy and ffmpeg on PATH; reads and writes PNG through ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess

import numpy as np

log = logging.getLogger("temporal_demo_post")


def read_png(path: str, w: int, h: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True, timeout=60).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(np.float32)


def write_png(path: str, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    data = np.clip(img + 0.5, 0, 255).astype(np.uint8).tobytes()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
                    path], input=data, check=True, timeout=60)


def rays(w: int, h: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera-space ray slopes (x right, y up, looking along -z) of every pixel centre."""
    t = math.tan(math.radians(fov_deg) / 2.0)
    xs = ((np.arange(w) + 0.5) / w * 2.0 - 1.0) * t
    ys = (1.0 - (np.arange(h) + 0.5) / h * 2.0) * t * h / w
    return np.meshgrid(xs, ys)


def cam_matrix(pitch_down_deg: float, yaw_deg: float) -> np.ndarray:
    """World-from-camera rotation of temporal_demo.py's camera: Blender XYZ Euler (90 - pitch, 0, -yaw), R = Rz Ry Rx."""
    ax, az = math.radians(90.0 - pitch_down_deg), math.radians(-yaw_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return rz @ rx


def reproject(prev: np.ndarray, yaw_prev: float, yaw_now: float, pitch_down: float, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample prev where each pixel's ray (a pure rotation, so no depth is needed) lay in the previous camera."""
    h, w = prev.shape[:2]
    gx, gy = rays(w, h, fov_deg)
    d = np.stack([gx, gy, -np.ones_like(gx)], axis=-1)                   # camera space: x right, y up, looking along -z
    m = cam_matrix(pitch_down, yaw_prev).T @ cam_matrix(pitch_down, yaw_now)   # previous camera from this camera
    q = d @ m.T
    zr = q[..., 2]
    px, py = q[..., 0] / -zr, q[..., 1] / -zr
    t = math.tan(math.radians(fov_deg) / 2.0)
    u = (px / t + 1.0) * 0.5 * w - 0.5
    v = (1.0 - py / (t * h / w)) * 0.5 * h - 0.5
    ok = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1) & (zr < 0)
    u0, v0 = np.clip(np.floor(u).astype(int), 0, w - 2), np.clip(np.floor(v).astype(int), 0, h - 2)
    fu, fv = np.clip(u - u0, 0, 1)[..., None], np.clip(v - v0, 0, 1)[..., None]
    s = (prev[v0, u0] * (1 - fu) * (1 - fv) + prev[v0, u0 + 1] * fu * (1 - fv) + prev[v0 + 1, u0] * (1 - fu) * fv
         + prev[v0 + 1, u0 + 1] * fu * fv)
    return s, ok


def neighbourhood(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    stack = np.stack([p[y:y + img.shape[0], x:x + img.shape[1]] for y in range(3) for x in range(3)])
    return stack.min(axis=0), stack.max(axis=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--blend", type=float, default=0.2)
    ap.add_argument("--clamp", type=int, default=1)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.dir, "demo_meta.json"), encoding="utf-8"))
    w, h, frames = meta["width"], meta["height"], meta["still"] + meta["pan"]
    yaw = meta["yaw_deg"]
    raw = [read_png(os.path.join(a.dir, f"raw_{f:03d}.png"), w, h) for f in range(frames)]
    ref = [read_png(os.path.join(a.dir, f"ref_{f:03d}.png"), w, h) for f in range(frames)]
    # the yaw sign of the reprojection, checked on the clean frames: the right sign must bring frame f-1 closer to f
    # than no reprojection at all (refuses to go on otherwise)
    sign_err = {s: 0.0 for s in (0.0, 1.0, -1.0)}
    pan_pairs = [f for f in range(1, frames) if abs(yaw[f] - yaw[f - 1]) > 0.5]
    pitch = meta["pitch_down_deg"]
    for f in pan_pairs:
        warped = {s: reproject(ref[f - 1], s * yaw[f - 1], s * yaw[f], pitch, meta["fov_deg"]) for s in sign_err}
        common = np.logical_and.reduce([ok for _, ok in warped.values()])   # the same pixels for every sign (API review)
        for s, (r, _) in warped.items():
            # compared on 4 x 4 block means, so the reference's own sampling noise does not decide it
            hb, wb = h // 4 * 4, w // 4 * 4
            d = np.where(common[:hb, :wb, None], r[:hb, :wb] - ref[f][:hb, :wb], np.nan)
            with np.errstate(all="ignore"):
                sign_err[s] += float(np.nanmean(np.abs(np.nanmean(d.reshape(hb // 4, 4, wb // 4, 4, 3), axis=(1, 3)))))
    sign = 1.0
    if pan_pairs:
        print("reprojection check (mean |reprojected f-1 - f| on clean frames): " +
              ", ".join(f"sign {s:+.0f} {v / len(pan_pairs):.3f}" for s, v in sign_err.items()))
        sign = min((1.0, -1.0), key=lambda s: sign_err[s])
        if not sign_err[sign] < sign_err[0.0]:
            raise SystemExit("neither yaw sign beats no reprojection: the camera model does not match")
    acc: list[np.ndarray] = []
    for f in range(frames):
        if f == 0:
            acc.append(raw[0].copy())
            continue
        hist, ok = reproject(acc[-1], sign * yaw[f - 1], sign * yaw[f], pitch, meta["fov_deg"])
        if a.clamp:
            lo, hi = neighbourhood(raw[f])
            hist = np.clip(hist, lo, hi)
        out = hist + (raw[f] - hist) * a.blend
        out[~ok] = raw[f][~ok]
        acc.append(out)
    for f in range(frames):
        write_png(os.path.join(a.dir, f"acc_{f:03d}.png"), acc[f])
        write_png(os.path.join(a.dir, f"side_{f:03d}.png"), np.concatenate([raw[f], acc[f], ref[f]], axis=1))
    # bush region (API review): the bush (x -5..5 m, 15-21 m ahead, up to 2.8 m) spans about 30-70 % across and
    # 30-50 % down at zero yaw (horizon at 38 %)
    x0, x1, y0, y1 = int(0.30 * w), int(0.70 * w), int(0.30 * h), int(0.50 * h)
    for name, lo_f, hi_f in (("still", 0, meta["still"]), ("pan", meta["still"], frames)):
        if hi_f - lo_f < 3:
            continue
        for label, seq in (("raw", raw), ("accumulated", acc), ("reference", ref)):
            s = np.stack([g[y0:y1, x0:x1].mean(axis=2) for g in seq[lo_f:hi_f]])
            d2 = float(np.abs(s[2:] - 2 * s[1:-1] + s[:-2]).mean())
            err = float(np.mean([np.abs(seq[f][y0:y1, x0:x1] - ref[f][y0:y1, x0:x1]).mean() for f in range(lo_f, hi_f)]))
            print(f"{name:5s} {label:11s} flicker {d2:7.3f}  error vs reference {err:7.3f}")


if __name__ == "__main__":
    main()
