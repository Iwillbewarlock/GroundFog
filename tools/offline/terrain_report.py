"""Picture and numbers for one TerrainFogRepro view (build-terrainfog.cmd).

Picture: the land with the fog of 0.2.0, 0.2.1 and the prototype side by side (fog_<old|off|new>.u8 over scene.rgb).
Numbers: the far layer alone, even thickness (faronly_<old|new>.f32), against the same layer lying on the true land
(ideal.f32, every fourth pixel): mean and 90th-percentile absolute difference of the fog alpha over land pixels, by
distance band; and how much the fog differs between neighbouring pixels (root mean square of pixel - 3x3 mean).
Usage: python -I terrain_report.py <view folder> <output.png> [title]
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 768
PANEL = 620
GAP = 30
TOP = 100
FOG = np.array([214.0, 221.0, 231.0])
PANELS = [("old", "0.2.0"), ("off", "0.2.1 (핫픽스)"), ("new", "새 방식 (시제품)")]
BANDS = [(0, 2253, "~32 m"), (2253, 3891, "32~56 m"), (3891, 8192, "56~117 m"), (8192, 20000, "117~286 m"), (20000, 1e9, "286 m~")]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


def grain(a: np.ndarray, land: np.ndarray) -> float:
    """Root mean square of (pixel - mean of its 3x3 block) over land pixels whose whole block is land."""
    pad = np.pad(a, 1, mode="edge")
    padl = np.pad(land, 1, mode="constant")
    mean = sum(pad[dy:dy + SIZE, dx:dx + SIZE] for dy in range(3) for dx in range(3)) / 9.0
    whole = np.ones_like(land)
    for dy in range(3):
        for dx in range(3):
            whole &= padl[dy:dy + SIZE, dx:dx + SIZE]
    d = (a - mean)[whole]
    return float(np.sqrt(np.mean(d * d))) if d.size else 0.0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    folder = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else folder.name
    scene = np.fromfile(folder / "scene.rgb", dtype=np.uint8).reshape(SIZE, SIZE, 3).astype(np.float64)
    land = np.fromfile(folder / "label.u8", dtype=np.uint8).reshape(SIZE, SIZE) == 1
    dist = np.fromfile(folder / "dist.f32", dtype=np.float32).reshape(SIZE, SIZE)
    width = PANEL * 3 + GAP * 2
    canvas = Image.new("RGB", (width, PANEL + TOP), (24, 26, 30))
    d = ImageDraw.Draw(canvas)
    fogs = {}
    for i, (key, name) in enumerate(PANELS):
        a = np.fromfile(folder / f"fog_{key}.u8", dtype=np.uint8).reshape(SIZE, SIZE).astype(np.float64) / 255.0
        fogs[key] = a
        rgb = np.clip(scene * (1.0 - a[..., None]) + FOG * a[..., None], 0, 255).astype(np.uint8)
        x = i * (PANEL + GAP)
        canvas.paste(Image.fromarray(rgb).resize((PANEL, PANEL), Image.LANCZOS), (x, TOP))
        d.text((x + 10, 8), name, fill=(255, 255, 255), font=font(34))
    d.text((10, 58), title, fill=(190, 200, 210), font=font(22))
    canvas.save(out)

    ideal = np.fromfile(folder / "ideal.f32", dtype=np.float32).reshape(SIZE, SIZE)
    have = land & (ideal >= 0.0)
    print(f"== {title}: {int(land.sum())} land pixels, {int(have.sum())} with the ideal")
    print("grain of what the player sees (field + far, patches on): " + ", ".join(f"{k} {grain(fogs[k], land):.4f}" for k, _ in PANELS))
    print(f"{'far layer alone vs ideal':<26} {'pixels':>7} {'ideal':>7} | {'old mean':>9} {'old p90':>8} | {'new mean':>9} {'new p90':>8}")
    far = {k: np.fromfile(folder / f"faronly_{k}.f32", dtype=np.float32).reshape(SIZE, SIZE) for k in ("old", "new")}
    for lo, hi, name in BANDS + [(0, 1e9, "all")]:
        m = have & (dist >= lo) & (dist < hi)
        if not m.any():
            continue
        row = f"{name:<26} {int(m.sum()):>7} {float(ideal[m].mean()):>7.3f} |"
        for k in ("old", "new"):
            e = np.abs(far[k][m] - ideal[m])
            row += f" {float(e.mean()):>9.3f} {float(np.percentile(e, 90)):>8.3f} |"
        print(row)
    print("grain of the far layer alone: " + ", ".join(f"{k} {grain(far[k].astype(np.float64), land):.4f}" for k in ("old", "new")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
