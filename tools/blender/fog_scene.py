"""Picture of the GroundFog fog over hills in Blender 5.2 Cycles (a look render, not the numeric comparison).

Usage:
  blender.exe -b --factory-startup --python tools/blender/fog_scene.py -- <data dir> <out.png> [width] [height]
      [samples] [seed] [sun elevation deg] [sun rotation deg] [eye height m] [opacity per metre]

Uses H and W from <data dir>/columns.bin (written by RaymarchExport.exe from the real wind/fog sim) and puts the
fog on a seeded hill terrain: the column image's ground channel is replaced by the terrain height, as the game does
with its height grid. Density formula: fog_reference.make_fog_material (the one the numeric comparison checked).

Conventions: Z up, metres, 70 game units per metre, field square [-2048, 2048] units = 58.5 m.
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

UNITS = ref.UNITS_PER_METRE
TERRAIN_RES = 481          # vertices per side of the terrain mesh (about 1 m spacing)
TERRAIN_SPAN = 8.0         # terrain side as a multiple of the field side (hides the horizon edge)
EYE_HEIGHT_M = 1.7
FOG_COLOR = (0.82, 0.86, 0.92)
FOG_ANISOTROPY = 0.35


def parse_args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("usage: ... -- <data dir> <out.png> [width] [height] [samples] [seed] [sun elev] [sun rot]")
    get = lambda i, d, f: f(argv[i]) if len(argv) > i else d  # noqa: E731
    return {
        "data": Path(argv[0]), "out": Path(argv[1]),
        "width": get(2, 480, int), "height": get(3, 270, int), "samples": get(4, 64, int),
        "seed": get(5, 7, int), "sun_elev": get(6, 9.0, float), "sun_rot": get(7, 60.0, float),
        "eye": get(8, EYE_HEIGHT_M, float), "opacity": get(9, None, float),
    }


def make_height_fn(seed: int, field_m: float):
    """Smooth hills: a few Gaussian bumps and dips plus a gentle tilt (metres). Same function for mesh and fog."""
    rng = np.random.default_rng(seed)
    bumps = [(rng.uniform(-1.5, 1.5) * field_m, rng.uniform(-1.5, 1.5) * field_m,
              rng.uniform(0.15, 0.4) * field_m, rng.uniform(-2.5, 4.0)) for _ in range(9)]
    tilt = rng.uniform(-0.02, 0.02, size=2)

    def height(x, y):
        h = tilt[0] * x + tilt[1] * y
        for bx, by, r, a in bumps:
            h = h + a * np.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2.0 * r * r))
        return h
    return height


def build_terrain(height, centre: float, side: float) -> bpy.types.Object:
    xs = np.linspace(centre - side / 2, centre + side / 2, TERRAIN_RES)
    gx, gy = np.meshgrid(xs, xs)
    gz = height(gx, gy)
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    n = TERRAIN_RES
    faces = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i) for j in range(n - 1) for i in range(n - 1)]
    mesh = bpy.data.meshes.new("Terrain")
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.scene.collection.objects.link(obj)

    mat = bpy.data.materials.new("Grass")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.6
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.10, 0.16, 0.05, 1.0)
    ramp.color_ramp.elements[1].color = (0.24, 0.26, 0.10, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    obj.data.materials.append(mat)
    return obj


def columns_on_terrain(cols: np.ndarray, height, origin: float, extent: float) -> np.ndarray:
    """Replace the ground channel with the terrain height at each texel centre (units), known = 1."""
    res = cols.shape[0]
    centres = (origin + (np.arange(res) + 0.5) * extent / res) / UNITS     # metres
    gx, gy = np.meshgrid(centres, centres)                                  # row = y, column = x
    out = cols.copy()
    out[..., 2] = height(gx, gy) * UNITS
    out[..., 3] = 1.0
    return out


def setup_world_and_sun(scene: bpy.types.Scene, elev: float, rot: float) -> None:
    world = bpy.data.worlds.new("Sky")
    world.use_nodes = True
    nt = world.node_tree
    bg = next(nd for nd in nt.nodes if nd.type == "BACKGROUND")
    sky = nt.nodes.new("ShaderNodeTexSky")
    types = [i.identifier for i in bpy.types.ShaderNodeTexSky.bl_rna.properties["sky_type"].enum_items]
    sky.sky_type = next((t for t in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "HOSEK_WILKIE") if t in types), types[0])
    sky.sun_elevation = math.radians(elev)
    sky.sun_rotation = math.radians(rot)
    sky.sun_disc = False            # the sun lamp below gives the sun light and shadows
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.25
    scene.world = world

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.5
    sun_data.color = (1.0, 0.86, 0.68)
    sun_data.angle = math.radians(0.6)
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    # Light travels along the lamp's -Z; point -Z away from the sun (sun at elevation elev, azimuth rot).
    to_sun = Vector((math.cos(math.radians(elev)) * math.sin(math.radians(rot)),
                     math.cos(math.radians(elev)) * math.cos(math.radians(rot)), math.sin(math.radians(elev))))
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = to_sun.to_track_quat("Z", "Y")


def setup_render(scene: bpy.types.Scene, a: dict) -> None:
    try:
        scene.render.engine = "CYCLES"
    except TypeError as err:
        raise SystemExit(f"Cycles not available: {err}")
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.refresh_devices()
            if any(d.type == backend for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                print(f"[fog_scene] Cycles device {backend}")
                break
        except TypeError:
            continue
    scene.cycles.samples = a["samples"]
    scene.cycles.use_denoising = True
    scene.cycles.volume_bounces = 2
    scene.render.resolution_x = a["width"]
    scene.render.resolution_y = a["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(a["out"])


def main() -> None:
    a = parse_args()
    cfg = json.loads((a["data"] / "cameras.json").read_text(encoding="utf-8"))
    origin = float(cfg["columns"]["origin"])
    extent = float(cfg["columns"]["extent"])
    cols = ref.load_columns(a["data"], int(cfg["columns"]["res"]))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    setup_render(scene, a)
    field_m = extent / UNITS
    centre_m = (origin + extent / 2.0) / UNITS
    height = make_height_fn(a["seed"], field_m)
    build_terrain(height, centre_m, field_m * TERRAIN_SPAN)

    terrain_cols = columns_on_terrain(cols, height, origin, extent)
    ground_m = terrain_cols[..., 2] / UNITS
    top_m = float((terrain_cols[..., 2] + terrain_cols[..., 0]).max()) / UNITS + 0.1
    low_m = float(ground_m.min()) - 0.1
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(centre_m, centre_m, (top_m + low_m) / 2.0))
    box = bpy.context.active_object
    box.name = "FogVolume"
    box.scale = (field_m, field_m, top_m - low_m)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    img = ref.make_columns_image(terrain_cols)
    box.data.materials.append(ref.make_fog_material(img, origin, extent, a["opacity"] if a["opacity"] is not None else float(cfg["opacity_per_metre"]),
                                                    float(cfg["steepness"]),
                                                    scatter={"color": FOG_COLOR, "anisotropy": FOG_ANISOTROPY}))

    setup_world_and_sun(scene, a["sun_elev"], a["sun_rot"])

    # Camera: eye height above the terrain at the field centre (where the game keeps the player), looking north.
    cx, cy = centre_m, centre_m
    eye = Vector((cx, cy, float(height(np.array(cx), np.array(cy))) + a["eye"]))
    look = Vector((cx + 4.0, cy + 30.0, float(height(np.array(cx + 4.0), np.array(cy + 30.0)))))
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 28.0
    cam_data.clip_start = 0.05
    cam_data.clip_end = 2000.0
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = eye
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (look - eye).to_track_quat("-Z", "Y")
    scene.camera = cam

    print(f"[fog_scene] field {field_m:.1f} m, fog box z {low_m:.2f}..{top_m:.2f} m, eye {tuple(round(v, 2) for v in eye)}")
    bpy.ops.render.render(write_still=True)
    print(f"[fog_scene] wrote {a['out']}")


if __name__ == "__main__":
    main()
