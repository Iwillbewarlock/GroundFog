"""Test terrain for the placement look render: a lake, a marsh with pools, a hot spring, a dry hollow and hills.

Shared by the scene writer (plain Python + numpy) and the Blender script, so the mesh and the fog use one terrain.

Write scene.bin for PlacementExport.exe:
  python -I tools/blender/placement_scene.py <scene.bin> <global> <cold> <wind> <use placement 0|1> [valley]
"valley" adds a wide shallow valley (about 120 m across, 4 m deep) crossing the field from west to east: wider than
the fine grid's pools reach, inside the coarse grid's (GroundFog CoarsePools). scene.bin also carries the coarse
grid (65 x 65 cells of 512 units around the player, Terrain::kCoarseSize / kCoarseCell).

Conventions: Z up, metres, 70 game units per metre. The fog field is centred on the player at the origin, as in the
game (WorldSize 8192 units = 117 m, HeightResolution 129, wind Resolution 256: GroundFog.ini in TAKEALOOK).
Kinds are PlacementRules::Kind: 0 land, 1 marsh land, 2 lake, 4 marsh water, 5 hot spring.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np

UNITS = 70.0
EXTENT = 8192.0            # units
ORIGIN = -EXTENT / 2.0
HEIGHT_SIZE = 129
WIND_RES = 256
THICKNESS = 126.0           # GroundFog.ini Thickness
WATER_Z = 0.0              # lake and marsh water surface, metres

LAKE = (-15.0, 18.0, 9.0, 3.5)          # centre x, y, basin radius (Gaussian sigma), depth below the base level
MARSH = (17.0, 16.0, 9.0, 7.0)          # centre x, y, half sizes x, y (smooth box)
SPRING = (5.0, 32.0, 3.0, 1.2)          # centre x, y, pool radius, basin depth (surface = 0.3 m below the rim)
HOLLOW = (-28.0, -12.0, 7.0, 0.8)       # dry hollow: centre, sigma, depth
LAND_BASE = 1.0
COARSE_SIZE = 65
COARSE_CELL = 512.0
WIDE_VALLEY = os.environ.get("GF_SCENE") == "valley"   # main() also takes a "valley" argument (the Blender script reads GF_SCENE)
VALLEY = (20.0, 60.0, 4.0)   # axis y, half width (m), depth (m): spans most of the 117 m field


def _gauss(x, y, cx, cy, sigma):
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma * sigma))


def _smooth_box(x, y, cx, cy, hx, hy, soft=3.0):
    def edge(d, h):
        t = np.clip((h - np.abs(d)) / soft, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    return edge(x - cx, hx) * edge(y - cy, hy)


def _marsh_noise(x, y):
    return 0.5 * np.sin(x * 0.9 + 1.3) * np.cos(y * 0.7 - 0.4) + 0.5 * np.sin(x * 0.37 - y * 0.53 + 2.0)


def marsh_weight(x, y):
    return _smooth_box(x, y, MARSH[0], MARSH[1], MARSH[2], MARSH[3])


def _wide_valley(x, y):
    """Smooth trough along x (depth VALLEY[2] at its axis, VALLEY[1] half width), rising to the base beyond."""
    t = np.clip(np.abs(y - VALLEY[0]) / VALLEY[1], 0.0, 1.0)
    return VALLEY[2] * (1.0 - t * t * (3.0 - 2.0 * t))


def _land(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if WIDE_VALLEY:
        return LAND_BASE + 3.0 - _wide_valley(x, y) + 0.4 * np.sin(x * 0.11) * np.cos(y * 0.07)
    h = (LAND_BASE + 4.0 * _gauss(x, y, 0.0, 50.0, 14.0) + 3.0 * _gauss(x, y, 40.0, -20.0, 12.0)
         - HOLLOW[3] * _gauss(x, y, HOLLOW[0], HOLLOW[1], HOLLOW[2]) - LAKE[3] * _gauss(x, y, LAKE[0], LAKE[1], LAKE[2]))
    wm = marsh_weight(x, y)
    h = h * (1.0 - wm) + wm * (0.1 + 0.3 * _marsh_noise(x, y))
    return h + 0.4 * _gauss(x, y, SPRING[0], SPRING[1], 5.0)


def spring_surface() -> float:
    """Hot spring water surface: 0.3 m below the ground at the pool centre before the basin is cut."""
    return float(_land(SPRING[0], SPRING[1])) - 0.3


def terrain(x, y):
    """Ground height (metres) under any water."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.hypot(x - SPRING[0], y - SPRING[1])
    return _land(x, y) - SPRING[3] * np.clip(1.0 - d / SPRING[2], 0.0, 1.0)


def kinds_and_surface(x, y):
    """Kind per point and the height the game's grid holds (water surface over water, else ground), metres."""
    g = terrain(x, y)
    kind = np.zeros(g.shape, dtype=np.uint8)
    surface = g.copy()
    wm = marsh_weight(x, y)
    under = g < WATER_Z
    marsh = wm > 0.5
    kind[marsh & ~under] = 1
    kind[under & ~marsh] = 2
    kind[under & marsh] = 4
    surface[under] = WATER_Z
    # Water only where the ground is below the surface, as Terrain.cpp uses water only when it is above the land.
    top = spring_surface()
    spring = (np.hypot(x - SPRING[0], y - SPRING[1]) < SPRING[2]) & (g < top)
    kind[spring] = 5
    surface[spring] = top
    return kind, surface


def grid_points():
    cell = EXTENT / (HEIGHT_SIZE - 1)
    coords = (ORIGIN + np.arange(HEIGHT_SIZE) * cell) / UNITS     # metres, cell corners as in Terrain.cpp
    gx, gy = np.meshgrid(coords, coords)                            # row = y, column = x
    return gx, gy


def write_scene(path: Path, glob: float, cold: float, wind: float, use_placement: int) -> dict:
    gx, gy = grid_points()
    kind, surface = kinds_and_surface(gx, gy)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", HEIGHT_SIZE, WIND_RES))
        f.write(struct.pack("<ffffff", ORIGIN, EXTENT, THICKNESS, glob, cold, wind))
        f.write(struct.pack("<i", use_placement))
        f.write((surface * UNITS).astype("<f4").tobytes())
        f.write(kind.astype(np.uint8).tobytes())
        # Coarse grid around the player (centre at the origin), cell corners as Terrain::UpdateCoarse.
        c_origin = -COARSE_CELL * (COARSE_SIZE - 1) / 2.0
        cc = (c_origin + np.arange(COARSE_SIZE) * COARSE_CELL) / UNITS
        cx, cy = np.meshgrid(cc, cc)
        _, csurf = kinds_and_surface(cx, cy)
        f.write(struct.pack("<ifff", COARSE_SIZE, c_origin, c_origin, COARSE_CELL))
        f.write((csurf * UNITS).astype("<f4").tobytes())
    counts = {name: int((kind == k).sum()) for k, name in ((0, "land"), (1, "marsh land"), (2, "lake"), (4, "marsh water"), (5, "hot spring"))}
    return counts


def main() -> None:
    global WIDE_VALLEY
    if len(sys.argv) not in (6, 7):
        raise SystemExit("usage: placement_scene.py <scene.bin> <global> <cold> <wind> <use placement 0|1> [valley]")
    WIDE_VALLEY = len(sys.argv) == 7 and sys.argv[6] == "valley"
    counts = write_scene(Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
    print(f"wrote {sys.argv[1]}: cells {counts}")


if __name__ == "__main__":
    main()
