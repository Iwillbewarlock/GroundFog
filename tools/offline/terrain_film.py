"""Side-by-side film of the far fog on real land: 0.2.0, 0.2.1 and the prototype, from TerrainFogRepro's film frames
(build-terrainfilm.cmd). One view after another, each with its own caption.

Input per view folder: scene.rgb (768 x 768 x 3, the land without fog), video_<old|off|new>.u8 (frames x 768 x 768
bytes, fog alpha). Only the fog amount comes from the shaders; the land's colours are plain lighting, no trees or rocks.
Usage: python -I terrain_film.py <output.mp4> <still prefix> <view folder>=<caption> [<view folder>=<caption> ...]
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 768
PANEL = 620
GAP = 30
TOP = 100
FPS = 24
FOG = np.array([214.0, 221.0, 231.0])
PANELS = [("old", "0.2.0"), ("off", "0.2.1 (지금 올라간 핫픽스)"), ("new", "새 방식 (시제품)")]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    out = pathlib.Path(sys.argv[1])
    still = sys.argv[2]
    width = PANEL * 3 + GAP * 2
    height = PANEL + TOP
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], stdin=subprocess.PIPE)
    assert ff.stdin is not None
    total = 0
    for spec in sys.argv[3:]:
        path, _, caption = spec.partition("=")
        folder = pathlib.Path(path)
        base = np.fromfile(folder / "scene.rgb", dtype=np.uint8).reshape(SIZE, SIZE, 3).astype(np.float32)
        data = {}
        frames = None
        for key, _ in PANELS:
            a = np.memmap(folder / f"video_{key}.u8", dtype=np.uint8, mode="r")
            n = a.size // (SIZE * SIZE)
            if n == 0 or a.size != n * SIZE * SIZE:
                print(f"{folder}/video_{key}.u8 has {a.size} bytes: not a whole number of {SIZE}x{SIZE} frames")
                return 1
            data[key] = a.reshape(n, SIZE, SIZE)
            frames = n if frames is None else min(frames, n)
        header = Image.new("RGB", (width, TOP), (24, 26, 30))
        d = ImageDraw.Draw(header)
        for i, (_, title) in enumerate(PANELS):
            d.text((i * (PANEL + GAP) + 10, 8), title, fill=(255, 255, 255), font=font(34))
        d.text((10, 58), caption or folder.name, fill=(190, 200, 210), font=font(22))
        head = np.asarray(header)
        for f in range(frames):
            canvas = np.full((height, width, 3), 24, dtype=np.uint8)
            canvas[:TOP] = head
            for i, (key, _) in enumerate(PANELS):
                a = data[key][f].astype(np.float32)[..., None] / 255.0
                rgb = np.clip(base * (1.0 - a) + FOG.astype(np.float32) * a, 0, 255).astype(np.uint8)
                x = i * (PANEL + GAP)
                canvas[TOP:, x:x + PANEL] = np.asarray(Image.fromarray(rgb).resize((PANEL, PANEL), Image.LANCZOS))
            ff.stdin.write(canvas.tobytes())
            if f == frames // 2:
                Image.fromarray(canvas).save(f"{still}_{folder.name}.png")
        total += frames
        print(f"{folder.name}: {frames} frames")
    ff.stdin.close()
    code = ff.wait()
    print(f"{total} frames, {width}x{height}, ffmpeg exit {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
