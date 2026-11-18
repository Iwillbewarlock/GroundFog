"""Builds a height grid of real Skyrim land from a houseCARL records export (LAND records of one worldspace box).

Export (houseCARL housecarl_records): types=["LAND"], where=["*parent.*parent.EditorID = Tamriel", "*parent.Grid.Point.X >= ..",
...], project={"form": "fields", "fields": ["*parent.Grid.Point", "VertexHeightMap.Offset", "VertexHeightMap.HeightMap[*].Value"],
"depth": 2}, to_file=<export.jsonl>. The values are the load order's winning LAND records, so landscape mods are in.

A LAND record holds 33 x 33 height steps: the cell's base is Offset x 8; the first value of each row steps the row's
start from the row before, the other values step along the row; one step = 8 units. Vertices are 128 units apart and
neighbouring cells share their edge rows and columns.

Output: <out>.f32 (rows x cols float32, row 0 = lowest y, 128 units apart) and <out>.json (origin, size, checks).
Usage: python -I land_from_housecarl.py <export.jsonl> <out prefix>
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

CELL = 4096.0
STEP = 128.0
N = 33


def decode(offset: float, steps: list[int]) -> np.ndarray:
    if len(steps) != N * N:
        raise ValueError(f"expected {N * N} height steps, got {len(steps)}")
    g = np.asarray(steps, dtype=np.float64).reshape(N, N)   # [y][x]
    h = np.empty((N, N))
    row = offset * 8.0
    for y in range(N):
        row += g[y, 0] * 8.0
        h[y, 0] = row
        h[y, 1:] = row + np.cumsum(g[y, 1:]) * 8.0
    return h


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    cells: dict[tuple[int, int], np.ndarray] = {}
    with src.open(encoding="utf-8") as f:
        next(f)   # manifest
        for line in f:
            rec = json.loads(line)
            point = None
            offset = None
            steps: list[int] = []
            for field in rec["fields"]:
                path = field["path"]
                if "value" not in field:
                    continue
                if path == "*parent.Grid.Point":
                    x, y = field["value"].split(",")
                    point = (int(x), int(y))
                elif path == "VertexHeightMap.Offset":
                    offset = float(field["value"])
                elif path.startswith("VertexHeightMap.HeightMap["):
                    steps.append(int(field["value"]))
            if point is None or offset is None:
                print(f"skipped {rec['formid']}: no grid point or no height map")
                continue
            cells[point] = decode(offset, steps)
    if not cells:
        print("no cells in the export")
        return 1
    xs = sorted({p[0] for p in cells})
    ys = sorted({p[1] for p in cells})
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    cols = (x1 - x0 + 1) * (N - 1) + 1
    rows = (y1 - y0 + 1) * (N - 1) + 1
    grid = np.full((rows, cols), np.nan)
    worst = 0.0
    shared = 0
    for (cx, cy), h in cells.items():
        r = (cy - y0) * (N - 1)
        c = (cx - x0) * (N - 1)
        old = grid[r:r + N, c:c + N]
        both = ~np.isnan(old)
        if both.any():
            worst = max(worst, float(np.abs(old[both] - h[both]).max()))
            shared += int(both.sum())
        grid[r:r + N, c:c + N] = np.where(both, old, h)
    missing = int(np.isnan(grid).sum())
    meta = {
        "origin_x": x0 * CELL, "origin_y": y0 * CELL, "step": STEP, "cols": cols, "rows": rows,
        "cells_x": [x0, x1], "cells_y": [y0, y1], "cells": len(cells),
        "shared_vertices_checked": shared, "worst_shared_difference": worst, "missing_vertices": missing,
        "min": float(np.nanmin(grid)), "max": float(np.nanmax(grid)),
    }
    np.nan_to_num(grid, nan=-1.0e30).astype(np.float32).tofile(out.with_suffix(".f32"))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0 if missing == 0 else 1   # edge mismatches come from the load order (landscape patches), not from the decoding


if __name__ == "__main__":
    raise SystemExit(main())
