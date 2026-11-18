"""Cross-section model of the far haze: which base height keeps the fog natural when the player is not on level land.

One vertical slice through the player (distance r along the view, height z). Compares, for ground points at several
distances, the fog alpha of
  ideal   the layer lying on the true ground everywhere (what would look natural),
  field   the wind field only (fades out between 55% and 95% of the half WorldSize): 0.2.1,
  B1      field + a level sheet at the player's ground height,
  B2      field + a level sheet at the ground height where the ray leaves the field's fade (per direction),
  B3      as B2, but the sheet follows the true ground while it is known (KNOWN_M) and holds the last height beyond.
  B4      the sheet follows the ground read from a coarse table (COARSE_M apart, linear between) while known, and
          beyond continues with the mean slope of the known land past the field; that part fades by how far the
          scene point is off the continued ground (either way).
Lengths in metres (70 units = 1 m). Usage: python -I far_haze_section.py
"""
from __future__ import annotations

import math
from collections.abc import Callable

K = 0.18            # extinction per metre (ini Opacity)
A = 3.0             # profile steepness (ini ProfileSteepness)
H = 0.9             # layer thickness used for every tier, metres (the mean of the patches)
EYE = 1.7
HALF = 8192.0 / 2 / 70.0          # half WorldSize, metres
R0, R1 = 0.55 * HALF, 0.95 * HALF  # the field's fade ring
BELOW0, BELOW1 = 150.0 / 70.0, 900.0 / 70.0   # kFarBelow0, kFarBelow1
KNOWN_M = 8192.0 / 70.0           # land the game always knows around the player
COARSE_M = 512.0 / 70.0           # the coarse terrain table's spacing
STEPS = 3000


def smooth(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def density(h: float) -> float:
    """Per metre at height h above the layer's base (0 outside the layer)."""
    if h < 0.0 or h >= H:
        return 0.0
    return K * A * math.exp(-A * h / H) / (1.0 - math.exp(-A))


def alpha(ground: Callable[[float], float], r: float, mode: str) -> float:
    """Fog alpha over the ground point at distance r, seen from the player at r = 0."""
    ez = ground(0.0) + EYE
    pz = ground(r)
    length = math.hypot(r, pz - ez)
    base1 = ground(0.0)
    base2 = ground(R0)
    tau = 0.0
    for i in range(STEPS):
        f = (i + 0.5) / STEPS
        d = r * f
        z = ez + (pz - ez) * f
        far = smooth(R0, R1, d)
        if mode == "ideal":
            rho = density(z - ground(d))
        else:
            rho = (1.0 - far) * density(z - ground(d))
            if mode == "B1":
                rho += far * density(z - base1) * (1.0 - smooth(BELOW0, BELOW1, base1 - pz))
            elif mode == "B2":
                rho += far * density(z - base2) * (1.0 - smooth(BELOW0, BELOW1, base2 - pz))
            elif mode == "B3":
                edge = ground(KNOWN_M)
                b = ground(d) if d <= KNOWN_M else edge
                keep = 1.0 if d <= KNOWN_M else 1.0 - smooth(BELOW0, BELOW1, edge - pz)
                rho += far * density(z - b) * keep
            elif mode == "B4":
                if d <= KNOWN_M:
                    rho += far * density(z - coarse(ground, d))
                else:
                    edge = coarse(ground, KNOWN_M)
                    s = (edge - coarse(ground, R1)) / (KNOWN_M - R1)
                    off = abs(pz - (edge + s * (r - KNOWN_M)))
                    rho += far * density(z - (edge + s * (d - KNOWN_M))) * (1.0 - smooth(BELOW0, BELOW1, off))
        tau += rho * length / STEPS
    return 1.0 - math.exp(-tau)


def coarse(ground: Callable[[float], float], d: float) -> float:
    """Ground from samples COARSE_M apart, linear between (the coarse table)."""
    i = math.floor(d / COARSE_M)
    f = d / COARSE_M - i
    return ground(i * COARSE_M) * (1.0 - f) + ground((i + 1) * COARSE_M) * f


def rough(d: float) -> float:
    """Rolling land with 1 m bumps 12 m apart (finer than the coarse table)."""
    return 2.0 * math.sin(2.0 * math.pi * d / 80.0) + 0.5 * math.sin(2.0 * math.pi * d / 12.0)


def slope(deg: float) -> Callable[[float], float]:
    t = math.tan(math.radians(deg))
    return lambda d: t * d


def hilltop(d: float) -> float:
    """Flat top 30 m across the radius, 15 degrees down to a plain 20 m lower."""
    return -min(max(d - 30.0, 0.0) * math.tan(math.radians(15.0)), 20.0)


def valley(d: float) -> float:
    """Flat floor to 80 m, then a 20-degree side."""
    return max(d - 80.0, 0.0) * math.tan(math.radians(20.0))


def drop(d: float) -> float:
    """Level to 70 m, a 10 m drop over 10 m, level again."""
    return -10.0 * min(max((d - 70.0) / 10.0, 0.0), 1.0)


def rolling(d: float) -> float:
    return 2.0 * math.sin(2.0 * math.pi * d / 80.0)


SCENES: list[tuple[str, Callable[[float], float]]] = [
    ("level", lambda d: 0.0),
    ("uphill 5 deg", slope(5.0)),
    ("uphill 10 deg", slope(10.0)),
    ("downhill 5 deg", slope(-5.0)),
    ("downhill 10 deg", slope(-10.0)),
    ("hilltop over a plain 20 m lower", hilltop),
    ("valley floor, sides from 80 m", valley),
    ("10 m drop at 70 m", drop),
    ("rolling 2 m / 80 m", rolling),
    ("rolling + 1 m bumps 12 m apart", rough),
]
MODES = ["ideal", "field", "B1", "B2", "B3", "B4"]
DISTS = [20, 30, 40, 50, 60, 80, 100, 150, 200, 400]


def main() -> None:
    for name, ground in SCENES:
        print(f"== {name}")
        print("   r(m) " + " ".join(f"{m:>6}" for m in MODES))
        worst = {m: 0.0 for m in MODES}
        for r in DISTS:
            row = {m: alpha(ground, float(r), m) for m in MODES}
            for m in MODES:
                worst[m] = max(worst[m], abs(row[m] - row["ideal"]))
            print(f"  {r:5d} " + " ".join(f"{row[m]:6.3f}" for m in MODES))
        print("  worst |x - ideal|: " + ", ".join(f"{m} {worst[m]:.2f}" for m in MODES[1:]))


if __name__ == "__main__":
    main()
