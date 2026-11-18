"""Mid-distance grass standing in the GroundFog layer, in Blender 5.2 Cycles (a physical reference picture).

Question it answers (DEBUG-LEDGER.md 2026-11-15, hypothesis H-grass-1): with a thin fog layer that ends at a hard top,
does physically correct light transport also show the grass at 15 to 60 m as dark outlines against bright fog?

Usage:
  blender.exe -b --factory-startup --python tools/blender/grass_fog_scene.py -- <out.png> [mode] [width] [height]
      [samples] [seed] [eye height m] [extinction per m at unit density]
  mode: nofog | fog (the plugin's layer: height H, density exp(-a h/H), cut at h = H) |
        softtop (the same amount of fog, but no cut: exp(-a h/H) continues above H)

Conventions: Z up, metres, 70 game units per metre. Flat ground at z = 0. Camera at the origin looking along +Y.
Layer: Thickness 126 units (1.8 m), ProfileSteepness a = 3, opacity 0.18 per metre x global 0.78 (clear night at
Riverwood, GroundFog.log 2026-11-15), W = H (unit mean density), so the extinction at height h is
  sigma(h) = 0.18 x 0.78 x a / (1 - exp(-a)) x exp(-a h / H)   per metre, for 0 <= h < H.
The blades are opaque dark diffuse surfaces (no light through the leaf): judge against the nofog picture.
Grass: seeded tapered blades 0.30 to 0.75 m tall and 2 cm wide in clumps, and wider fern-like fans, from 6 to 70 m.
"""
from __future__ import annotations

import math
import sys

import bpy
import numpy as np

H_M = 126.0 / 70.0
STEEPNESS = 3.0
FOG_COLOR = (0.82, 0.86, 0.92)
FOG_ANISOTROPY = 0.35


def parse_args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 1:
        raise SystemExit("usage: ... -- <out.png> [mode] [width] [height] [samples] [seed] [eye m] [extinction]")
    get = lambda i, d, f: f(argv[i]) if len(argv) > i else d  # noqa: E731
    cfg = {"out": argv[0], "mode": get(1, "fog", str), "width": get(2, 640, int), "height": get(3, 400, int),
           "samples": get(4, 64, int), "seed": get(5, 7, int), "eye": get(6, 3.2, float),
           "sigma": get(7, 0.18 * 0.78, float)}
    if cfg["mode"] not in ("nofog", "fog", "softtop"):
        raise SystemExit(f"unknown mode {cfg['mode']!r}: nofog, fog or softtop")
    return cfg


def diffuse(name: str, color: tuple[float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def make_grass(seed: int) -> bpy.types.Object:
    """One mesh of all blades (two-sided quads tapering to a point, bent a little), in clumps on a flat ground."""
    rng = np.random.default_rng(seed)
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    def blade(x: float, y: float, height: float, width: float, lean: float, yaw: float) -> None:
        cx, sx = math.cos(yaw), math.sin(yaw)
        base = len(verts)
        segs = 3
        for k in range(segs + 1):
            t = k / segs
            w = 0.5 * width * (1.0 - t) ** 0.7
            off = lean * t * t                      # bends over toward its tip
            px, py, pz = x + off * sx, y - off * cx, height * t
            verts.append((px - w * cx, py - w * sx, pz))
            verts.append((px + w * cx, py + w * sx, pz))
        for k in range(segs):
            a = base + 2 * k
            faces.append((a, a + 1, a + 3, a + 2))

    # clumps: denser near the camera's line of sight, from 6 m to 70 m along +Y, 30 m to each side
    for _ in range(2600):
        cy = rng.uniform(6.0, 70.0)
        cx = rng.uniform(-0.55, 0.55) * (cy + 6.0)
        fern = rng.random() < 0.12
        count = int(rng.integers(5, 12)) if fern else int(rng.integers(8, 22))
        radius = 0.25 if fern else rng.uniform(0.15, 0.45)
        for _ in range(count):
            ang, rr = rng.uniform(0.0, 2.0 * math.pi), radius * math.sqrt(rng.random())
            if fern:
                blade(cx + rr * math.cos(ang), cy + rr * math.sin(ang), rng.uniform(0.35, 0.6), rng.uniform(0.10, 0.18),
                      rng.uniform(0.25, 0.5), ang)
            else:
                blade(cx + rr * math.cos(ang), cy + rr * math.sin(ang), rng.uniform(0.30, 0.75), 0.02,
                      rng.uniform(0.02, 0.2), rng.uniform(0.0, math.pi))
    mesh = bpy.data.meshes.new("Grass")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Grass", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(diffuse("GrassGreen", (0.045, 0.09, 0.03)))
    return obj


def make_fog(mode: str, sigma: float) -> None:
    """A box of volume from the ground up. fog: cut at H. softtop: the same exponential, not cut (box to 4 H)."""
    top = H_M if mode == "fog" else 4.0 * H_M
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 60.0, 0.5 * top))
    box = bpy.context.active_object
    box.name = "FogBox"
    box.scale = (240.0, 240.0, top)
    mat = bpy.data.materials.new("GroundFogLayer")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Z"], mul.inputs[0])
    mul.inputs[1].default_value = -STEEPNESS / H_M
    ex = nt.nodes.new("ShaderNodeMath")
    ex.operation = "EXPONENT"
    nt.links.new(mul.outputs[0], ex.inputs[0])
    dens = nt.nodes.new("ShaderNodeMath")
    dens.operation = "MULTIPLY"
    nt.links.new(ex.outputs[0], dens.inputs[0])
    # a / (1 - exp(-a)): the vertical integral over 0..H is sigma x H, as the plugin's profile has it. softtop keeps
    # that total over its taller box (0..4 H), so its density at the ground is 0.95 of the cut layer's.
    span = 1.0 if mode == "fog" else 4.0
    dens.inputs[1].default_value = sigma * STEEPNESS / -math.expm1(-STEEPNESS * span)
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (*FOG_COLOR, 1.0)
    vol.inputs["Anisotropy"].default_value = FOG_ANISOTROPY
    nt.links.new(dens.outputs[0], vol.inputs["Density"])
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    box.data.materials.append(mat)


def main() -> None:
    cfg = parse_args()
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 60.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.scale = (400.0, 400.0, 1.0)
    ground.data.materials.append(diffuse("Dirt", (0.07, 0.06, 0.05)))
    make_grass(cfg["seed"])
    if cfg["mode"] != "nofog":
        make_fog(cfg["mode"], cfg["sigma"])

    # clear night: a dim blue sky and a moon from behind the camera's left, 35 degrees up
    world = bpy.data.worlds.new("NightSky")
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.02, 0.035, 0.07, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world
    moon_data = bpy.data.lights.new("Moon", "SUN")
    moon_data.energy = 0.6
    moon_data.color = (0.6, 0.72, 1.0)
    moon_data.angle = math.radians(0.5)
    moon = bpy.data.objects.new("Moon", moon_data)
    moon.rotation_euler = (math.radians(55.0), 0.0, math.radians(-40.0))
    scene.collection.objects.link(moon)

    cam_data = bpy.data.cameras.new("Eye")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(80.0)          # about the game's third-person horizontal view at 16:10
    cam_data.clip_start, cam_data.clip_end = 0.1, 2000.0
    cam = bpy.data.objects.new("Eye", cam_data)
    cam.location = (0.0, 0.0, cfg["eye"])
    cam.rotation_euler = (math.radians(90.0 - 7.0), 0.0, 0.0)   # looking along +Y, 7 degrees down
    scene.collection.objects.link(cam)
    scene.camera = cam

    scene.render.engine = "CYCLES"
    scene.cycles.samples = cfg["samples"]
    scene.cycles.use_denoising = True
    scene.cycles.volume_step_rate = 0.25
    scene.cycles.volume_max_steps = 512
    scene.cycles.volume_bounces = 4
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = True
        scene.cycles.device = "GPU"
    except TypeError:
        scene.cycles.device = "CPU"
    scene.render.resolution_x, scene.render.resolution_y = cfg["width"], cfg["height"]
    scene.render.resolution_percentage = 100
    scene.view_settings.exposure = 3.0           # night scene brought up to a readable level, the same for every mode
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = cfg["out"]
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {cfg['mode']} {cfg['width']}x{cfg['height']} spp {cfg['samples']} device {scene.cycles.device} -> {cfg['out']}")


if __name__ == "__main__":
    main()
