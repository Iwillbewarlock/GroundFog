"""Look render of the GroundFog placement (lake, marsh, hot spring, hollow) in Blender 5.2 Cycles.

Usage:
  blender.exe -b --factory-startup --python tools/blender/placement_render.py -- <data dir> <out.png>
      [view eye|over|lake|hollow|valley] [width] [height] [samples] [sun elevation deg] [sun rotation deg] [opacity per metre]
      [anisotropy g] [far pools 0|1]

<data dir> holds columns.bin and columns.json from PlacementExport.exe (the plugin's PlacementRules and column
shader with the conditions written into scene.bin by placement_scene.py). The fog density is
fog_reference.make_fog_material, the formula the numeric comparison checked against the HLSL. The column ground is
used as the game uses it (water surface over water), not replaced.

Far pools (last argument 1, coarse.bin from PlacementExport with coarse = 1): the coarse grid's pooled fog beyond the
field as a third volume, the model of RaymarchShaders FarPoolTau (coarse ground and depth, the tall layer's profile a = 1, share
1 - the field's round fade, fade over the grid's last two cells). Limit: the far tier's thin layer (FarLayerTau) is
not drawn, and fog below the coarse ground (FarPoolTau counts the density at the ground there) is left out, as the
volume material drops h < 0 (valley scene: the coarse ground sits above the true one by 4.3 cm at the 99th
percentile, 0.45 m at most).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fog_reference as ref  # noqa: E402
import fog_scene as look  # noqa: E402
import placement_scene as ps  # noqa: E402

UNITS = ref.UNITS_PER_METRE
STEEPNESS = 3.0            # GroundFog.ini ProfileSteepness
UPPER_STEEPNESS = 1.0      # RaymarchShaders.h kUpperSteepness (tall layer)
OPACITY = 0.3              # GroundFog.ini Opacity
TERRAIN_SIDE = 3.0         # terrain mesh side as a multiple of the field side
EYE_HEIGHT_M = 1.7


def parse_args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("usage: ... -- <data dir> <out.png> [eye|over] [width] [height] [samples] [sun elev] [sun rot] [opacity]")
    get = lambda i, d, f: f(argv[i]) if len(argv) > i else d  # noqa: E731
    view = get(2, "eye", str)
    if view not in ("eye", "over", "lake", "hollow", "valley", "across"):
        raise SystemExit(f"view '{view}' is not eye, over or lake")
    return {
        "data": Path(argv[0]), "out": Path(argv[1]), "view": view,
        "width": get(3, 480, int), "height": get(4, 270, int), "samples": get(5, 64, int),
        "sun_elev": get(6, 6.0, float), "sun_rot": get(7, 100.0, float), "opacity": get(8, OPACITY, float),
        "anisotropy": get(9, look.FOG_ANISOTROPY, float), "far": get(10, 0, int) != 0,
    }


def material(name: str, color, roughness: float, metallic: float = 0.0, emission=None) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(nd for nd in mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission[0], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission[1]
    return mat


def add_water(scene: bpy.types.Scene, side: float) -> None:
    # One plane at the lake/marsh surface; the terrain is above it everywhere else (placement_scene keeps land >= 0.2 m).
    bpy.ops.mesh.primitive_plane_add(size=side, location=(0.0, 0.0, ps.WATER_Z))
    plane = bpy.context.active_object
    plane.name = "Water"
    plane.data.materials.append(material("Water", (0.02, 0.05, 0.06), 0.05))
    bpy.ops.mesh.primitive_circle_add(vertices=48, radius=ps.SPRING[2], fill_type="NGON", location=(ps.SPRING[0], ps.SPRING[1], ps.spring_surface()))
    spring = bpy.context.active_object
    spring.name = "HotSpring"
    spring.data.materials.append(material("HotSpringWater", (0.10, 0.22, 0.20), 0.1))


def add_camera(scene: bpy.types.Scene, view: str) -> None:
    if view == "eye":
        eye_xy, look_xy, lens = (0.0, -2.0), (0.0, 25.0), 24.0
        eye = Vector((*eye_xy, float(ps.terrain(*eye_xy)) + EYE_HEIGHT_M))
        look = Vector((*look_xy, 0.0))
    elif view == "hollow":
        # Low and across the dry hollow toward the lake, to see whether fog fills low ground with a level top.
        eye_xy, look_xy, lens = (-36.0, -26.0), (-20.0, 10.0), 30.0
        eye = Vector((*eye_xy, float(ps.terrain(*eye_xy)) + EYE_HEIGHT_M))
        look = Vector((*look_xy, 0.0))
    elif view == "valley":
        # From the valley's south slope down the valley (axis y = VALLEY[0], along x), past the field edge (about 58 m)
        # toward 200 m: the far pools continue the field's valley pool or the fog stops at the field.
        eye_xy, look_xy, lens = (-30.0, ps.VALLEY[0] - 75.0), (150.0, ps.VALLEY[0]), 26.0
        eye = Vector((*eye_xy, float(ps.terrain(*eye_xy)) + 20.0))
        look = Vector((*look_xy, 0.0))
    elif view == "across":
        # Across the valley at x = 110 m, outside the field (about 58 m): only the far pools can hold fog there.
        eye_xy, look_xy, lens = (110.0, ps.VALLEY[0] + 75.0), (110.0, ps.VALLEY[0] - 60.0), 30.0
        eye = Vector((*eye_xy, float(ps.terrain(*eye_xy)) + 2.5))
        look = Vector((*look_xy, 1.0))
    elif view == "lake":
        # From the lake's south shore across the water, like a lakeside photograph.
        eye_xy, look_xy, lens = (-13.0, 3.0), (-17.0, 40.0), 35.0
        eye = Vector((*eye_xy, float(ps.terrain(*eye_xy)) + EYE_HEIGHT_M))
        look = Vector((*look_xy, 1.0))
    else:
        eye, look, lens = Vector((-6.0, -40.0, 22.0)), Vector((2.0, 14.0, 0.0)), 28.0
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = lens
    cam_data.clip_start = 0.05
    cam_data.clip_end = 2000.0
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = eye
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (look - eye).to_track_quat("-Z", "Y")
    scene.camera = cam


def far_pool_columns(coarse: np.ndarray, cfg: dict, origin: float, extent: float, res: int) -> tuple[np.ndarray, float, float]:
    """Column image (H, W, ground, known) over the coarse grid for the far pools, as FarPoolTau reads them.

    Returns the columns, the image origin and its extent (game units)."""
    size = int(cfg["coarse_size"])
    cell = float(cfg["coarse_cell"])
    cx0, cy0 = float(cfg["coarse_origin_x"]), float(cfg["coarse_origin_y"])
    if abs(cx0 - cy0) > 1e-3:
        # make_fog_material takes one origin for both axes (the test scenes centre the grid on the origin).
        raise SystemExit(f"coarse origins differ ({cx0}, {cy0}); far_pool_columns needs a square grid at one origin")
    span = cell * (size - 1)
    xs = cx0 + (np.arange(res) + 0.5) / res * span
    gx, gy = np.meshgrid((xs - cx0) / cell, (xs - cy0) / cell)          # row = y, column = x
    c = np.clip(np.floor(gx).astype(int), 0, size - 2), np.clip(np.floor(gy).astype(int), 0, size - 2)
    fx, fy = np.clip(gx - c[0], 0.0, 1.0), np.clip(gy - c[1], 0.0, 1.0)

    def bilinear(ch: int) -> np.ndarray:
        a = coarse[c[1], c[0], ch]
        b = coarse[c[1], c[0] + 1, ch]
        d = coarse[c[1] + 1, c[0], ch]
        e = coarse[c[1] + 1, c[0] + 1, ch]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy

    unknown = coarse[..., 0] < -1.0e29
    near_unknown = unknown[c[1], c[0]] | unknown[c[1], c[0] + 1] | unknown[c[1] + 1, c[0]] | unknown[c[1] + 1, c[0] + 1]
    safe = coarse.copy()
    safe[unknown] = 0.0
    coarse = safe
    depth = bilinear(1)
    ground = bilinear(0)
    # CoarsePoolAt: the lid mixes only the pooled corners' lids (weights x saturate(depth), renormalised).
    corners = [(c[1], c[0], (1 - fx) * (1 - fy)), (c[1], c[0] + 1, fx * (1 - fy)), (c[1] + 1, c[0], (1 - fx) * fy), (c[1] + 1, c[0] + 1, fx * fy)]
    wsum = np.zeros_like(fx)
    lsum = np.zeros_like(fx)
    for r, q, wgt in corners:
        wp = wgt * np.clip(coarse[r, q, 1], 0.0, 1.0)
        wsum += wp
        lsum += wp * (coarse[r, q, 0] + coarse[r, q, 1])
    lid = np.where(wsum > 1e-6, lsum / np.maximum(wsum, 1e-6), ground)
    H = np.where((depth >= 1.0) & ~near_unknown, np.minimum(lid - ground, float(cfg["pool_cap"])), 0.0)
    H = np.where(H >= 1.0, H, 0.0)
    wx, wy = np.meshgrid(xs, cy0 - cx0 + xs)
    ux, uy = (wx - origin) / extent - 0.5, (wy - origin) / extent - 0.5
    t = np.clip((np.hypot(ux, uy) * 2.0 - 0.55) / 0.4, 0.0, 1.0)
    share = t * t * (3.0 - 2.0 * t)                                     # FarWeight
    last = size - 1.0
    fade = np.clip(np.minimum(np.minimum(gx, gy), np.minimum(last - gx, last - gy)) / 2.0, 0.0, 1.0)
    W = H / UNITS * float(cfg["pool_density"]) * share * fade           # W / (H / 70) = PoolF.x
    cols = np.stack([H, W, np.where(H > 0.0, ground, 0.0), (H > 0.0).astype(np.float32)], axis=-1).astype(np.float32)
    return cols, cx0, span


def add_far_pools(data: Path, cfg: dict, origin: float, extent: float, low_m: float, a: dict) -> None:
    path = data / "coarse.bin"
    if int(cfg.get("coarse_size", 0)) < 2 or not path.exists():
        raise SystemExit(f"far pools asked for, but {path} or coarse_size is missing (PlacementExport ... coarse 1)")
    size = int(cfg["coarse_size"])
    coarse = np.fromfile(path, dtype=np.float32).reshape(size, size, 2)
    cols, c0, span = far_pool_columns(coarse, cfg, origin, extent, 512)
    if float(cols[..., 1].max()) <= 0.0:
        print("[placement_render] far pools: none outside the field")
        return
    pooled = cols[..., 1] > 0.0
    top = float(((cols[..., 2] + cols[..., 0]) / UNITS)[pooled].max()) + 0.1
    low = min(low_m, float((cols[..., 2] / UNITS)[pooled].min()) - 0.1)
    centre = (c0 + span / 2.0) / UNITS
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(centre, centre, (top + low) / 2.0))
    box = bpy.context.active_object
    box.name = "FogVolumeFarPools"
    box.scale = (span / UNITS, span / UNITS, top - low)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    img = ref.make_columns_image(cols)
    img.name = "FogFarPools"
    box.data.materials.append(ref.make_fog_material(img, c0, span, a["opacity"], UPPER_STEEPNESS,
                                                    scatter={"color": look.FOG_COLOR, "anisotropy": a["anisotropy"]}))
    print(f"[placement_render] far pools: box {span / UNITS:.0f} m, top {top:.2f} m, max W {float(cols[..., 1].max()):.3f}")


def main() -> None:
    a = parse_args()
    cfg = json.loads((a["data"] / "columns.json").read_text(encoding="utf-8"))
    res, origin, extent = int(cfg["res"]), float(cfg["origin"]), float(cfg["extent"])
    cols = ref.load_columns(a["data"], res)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    look.setup_render(scene, {"samples": a["samples"], "width": a["width"], "height": a["height"], "out": a["out"]})
    field_m = extent / UNITS
    side = field_m * (4.2 if a["far"] or a["view"] in ("valley", "across") else TERRAIN_SIDE)   # the coarse grid reaches about 233 m
    look.build_terrain(ps.terrain, 0.0, side)
    add_water(scene, side)

    known = cols[..., 3] > 0.5
    ground_m = np.where(known, cols[..., 2] / np.maximum(cols[..., 3], 1e-3), 0.0) / UNITS
    top_m = float((ground_m + cols[..., 0] / UNITS)[known].max()) + 0.1
    low_m = float(ground_m[known].min()) - 0.1
    centre_m = (origin + extent / 2.0) / UNITS
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(centre_m, centre_m, (top_m + low_m) / 2.0))
    box = bpy.context.active_object
    box.name = "FogVolume"
    box.scale = (field_m, field_m, top_m - low_m)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    img = ref.make_columns_image(cols)
    box.data.materials.append(ref.make_fog_material(img, origin, extent, a["opacity"], STEEPNESS,
                                                    scatter={"color": look.FOG_COLOR, "anisotropy": a["anisotropy"]}))
    # The tall layer (steam, pooled fog) as a second volume over the same ground; Cycles adds the two densities, as the
    # plugin's march adds the two layers' optical depths. Profile steepness 1 (RaymarchShaders kUpperSteepness).
    upper_path = a["data"] / "upper.bin"
    if upper_path.exists():
        up = np.fromfile(upper_path, dtype=np.float32).reshape(res, res, 4)
        if float(up[..., 1].max()) > 0.0:
            up_known = up[..., 3] > 0.5
            up_top = float((ground_m + up[..., 0] / UNITS)[up_known].max()) + 0.1
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(centre_m, centre_m, (up_top + low_m) / 2.0))
            box2 = bpy.context.active_object
            box2.name = "FogVolumeUpper"
            box2.scale = (field_m * 0.999, field_m * 0.999, up_top - low_m)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            up_img = ref.make_columns_image(up)
            up_img.name = "FogUpper"
            box2.data.materials.append(ref.make_fog_material(up_img, origin, extent, a["opacity"], UPPER_STEEPNESS,
                                                             scatter={"color": look.FOG_COLOR, "anisotropy": a["anisotropy"]}))

    if a["far"]:
        add_far_pools(a["data"], cfg, origin, extent, low_m, a)

    look.setup_world_and_sun(scene, a["sun_elev"], a["sun_rot"])
    add_camera(scene, a["view"])
    print(f"[placement_render] placement {cfg['placement']} global {cfg['global']} cold {cfg['cold']} wind {cfg['wind']}, "
          f"fog box z {low_m:.2f}..{top_m:.2f} m, view {a['view']}")
    bpy.ops.render.render(write_still=True)
    print(f"[placement_render] wrote {a['out']}")


if __name__ == "__main__":
    main()
