"""How the ground fog would look with sun light that knows the fog in its way (Blender 5.2 Cycles), next to how the
plugin lights it now. A picture for a decision (ledger 2026-11-16, sun lighting); not the plugin's shader.

Two looks of the same uneven fog layer (mounds and hollows like the plugin's LayerShape, density exp(-3 h / H)):
  flat    as the plugin does now: the fog's colour is the same everywhere, sky colour plus a sun part set by the sun's
          height (emission + absorption: no self-shadowing, no brighter sun side, no glow through thin edges)
  shadow  the target: the fog scatters sun and sky light, and the sun light reaching a point is dimmed by the fog
          between that point and the sun (Cycles volume scattering, anisotropy 0.6 = the plugin's GlowAnisotropy)
Both are matched in overall display brightness below the horizon (the flat emission is rescaled twice toward the
shadow render's lower-half mean, display-referred after AgX) so the comparison is about where the light falls.

Views: "side" (sun 12 deg high, from the right, about 85 deg off the view) and "back" (camera facing the sun, 25 deg off its azimuth).
Scene, Z up, metres: ground with gentle hills, a few trees (cones on trunks), rocks; fog box 120 x 120 m, layer top
H = 1.8 m with the mounds; camera 1.8 m up.

Usage:
  blender.exe -b --factory-startup --python tools/blender/sun_light_demo.py -- --out-dir <dir>
      [--width 960 --height 540 --samples 512 --sigma 0.42 --seed 3]
Writes <view>_<look>.png and sun_meta.json. Existing PNGs are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

import bpy
import numpy as np

log = logging.getLogger("sun_light_demo")
H_M = 1.8
STEEPNESS = 3.0
SUN_ELEV_DEG = 12.0
SUN_AZIMUTH_DEG = 150.0          # direction the sun light travels toward, about Z (0 = +Y)
SUN_ENERGY = 4.0
SUN_COLOUR = (1.0, 0.78, 0.55)
SKY_COLOUR = (0.55, 0.62, 0.78)
FOG_ALBEDO = (0.9, 0.92, 0.95)
ANISOTROPY = 0.6
EYE_M = 1.8
VIEWS = {"side": 55.0, "back": None}   # camera yaw in degrees; None = face the sun (25 deg off)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=0.42)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eye", type=float, default=EYE_M)
    a = ap.parse_args(argv)
    if a.width < 16 or a.height < 16 or a.samples < 1 or a.sigma <= 0.0:
        raise SystemExit(f"bad arguments: {a}")
    return a


def diffuse(name: str, colour: tuple[float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def ground(rng: np.random.Generator) -> None:
    n = 120
    xs = np.linspace(-80.0, 80.0, n)
    phase = rng.uniform(0, 2 * math.pi, 4)
    verts = []
    for y in np.linspace(-20.0, 140.0, n):
        for x in xs:
            z = 0.6 * math.sin(x / 13.0 + phase[0]) * math.sin(y / 17.0 + phase[1]) + 0.3 * math.sin(x / 5.0 + phase[2] + y / 7.0)
            verts.append((float(x), float(y), z))
    faces = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i) for j in range(n - 1) for i in range(n - 1)]
    mesh = bpy.data.meshes.new("Ground")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Ground", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(diffuse("Grass", (0.07, 0.10, 0.04)))


def props(rng: np.random.Generator) -> None:
    bark, leaves, rock = diffuse("Bark", (0.08, 0.05, 0.03)), diffuse("Needles", (0.03, 0.07, 0.03)), diffuse("Rock", (0.3, 0.3, 0.29))
    for _ in range(18):
        x, y = rng.uniform(-40, 40), rng.uniform(12, 90)
        h = rng.uniform(6, 11)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=h * 0.4, location=(x, y, h * 0.2))
        bpy.context.active_object.data.materials.append(bark)
        bpy.ops.mesh.primitive_cone_add(radius1=rng.uniform(1.6, 2.4), depth=h * 0.8, location=(x, y, h * 0.4 + h * 0.4))
        bpy.context.active_object.data.materials.append(leaves)
    for _ in range(10):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=rng.uniform(0.8, 2.2), location=(rng.uniform(-30, 30), rng.uniform(8, 60), 0.2))
        o = bpy.context.active_object
        o.scale = (1.0, rng.uniform(0.7, 1.3), rng.uniform(0.5, 0.8))
        o.data.materials.append(rock)


def fog_density(nt: bpy.types.NodeTree, sigma: float) -> bpy.types.NodeSocket:
    """Density: sigma x a/(1-e^-a) x exp(-a z / Htop), cut at Htop, Htop = H x (0.35 + 0.9 x clamp((noise(xy / 13 m) - 0.3) / 0.4)).
    z is world height (the ground rolls by under 1 m), not height above the ground."""
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "2D"
    noise.inputs["Scale"].default_value = 1.0 / 13.0
    noise.inputs["Detail"].default_value = 2.0
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], comb.inputs["X"])
    nt.links.new(sep.outputs["Y"], comb.inputs["Y"])
    nt.links.new(comb.outputs[0], noise.inputs["Vector"])
    top = nt.nodes.new("ShaderNodeMapRange")
    top.inputs["From Min"].default_value, top.inputs["From Max"].default_value = 0.3, 0.7
    top.inputs["To Min"].default_value, top.inputs["To Max"].default_value = 0.35 * H_M, 1.25 * H_M
    nt.links.new(noise.outputs["Fac"], top.inputs["Value"])
    rel = nt.nodes.new("ShaderNodeMath")
    rel.operation = "DIVIDE"
    nt.links.new(sep.outputs["Z"], rel.inputs[0])
    nt.links.new(top.outputs[0], rel.inputs[1])
    expo = nt.nodes.new("ShaderNodeMath")
    expo.operation = "MULTIPLY"
    nt.links.new(rel.outputs[0], expo.inputs[0])
    expo.inputs[1].default_value = -STEEPNESS
    ex = nt.nodes.new("ShaderNodeMath")
    ex.operation = "EXPONENT"
    nt.links.new(expo.outputs[0], ex.inputs[0])
    cut = nt.nodes.new("ShaderNodeMath")
    cut.operation = "LESS_THAN"
    nt.links.new(rel.outputs[0], cut.inputs[0])
    cut.inputs[1].default_value = 1.0
    d0 = nt.nodes.new("ShaderNodeMath")
    d0.operation = "MULTIPLY"
    nt.links.new(ex.outputs[0], d0.inputs[0])
    nt.links.new(cut.outputs[0], d0.inputs[1])
    dens = nt.nodes.new("ShaderNodeMath")
    dens.operation = "MULTIPLY"
    nt.links.new(d0.outputs[0], dens.inputs[0])
    dens.inputs[1].default_value = sigma * STEEPNESS / -math.expm1(-STEEPNESS)
    return dens.outputs[0]


def fog_box(sigma: float, look: str, flat_emission: float) -> None:
    for o in [o for o in bpy.data.objects if o.name == "FogBox"]:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 50.0, 1.4 * H_M))
    box = bpy.context.active_object
    box.name = "FogBox"
    box.scale = (120.0, 120.0, 2.8 * H_M + 2.0)
    box.location.z = 0.5 * box.scale.z - 1.0
    mat = bpy.data.materials.new(f"Fog_{look}")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    dens = fog_density(nt, sigma)
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    if look == "shadow":
        vol.inputs["Color"].default_value = (*FOG_ALBEDO, 1.0)
        vol.inputs["Anisotropy"].default_value = ANISOTROPY
        nt.links.new(dens, vol.inputs["Density"])
    else:   # flat: absorb, and emit a fixed colour in proportion to density (light that ignores the fog in its way)
        vol.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)            # no scattering
        vol.inputs["Absorption Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # black absorption: extinction = density
        nt.links.new(dens, vol.inputs["Density"])
        sun_part = math.sin(math.radians(SUN_ELEV_DEG)) * SUN_ENERGY * 0.25
        col = tuple((s + sun_part * c) * flat_emission for s, c in zip(SKY_COLOUR, SUN_COLOUR))
        vol.inputs["Emission Color"].default_value = (*col, 1.0)
        nt.links.new(dens, vol.inputs["Emission Strength"])   # emission per unit length follows the density
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    box.data.materials.append(mat)


def setup(a: argparse.Namespace) -> bpy.types.Object:
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    rng = np.random.default_rng(a.seed)
    ground(rng)
    props(rng)
    world = bpy.data.worlds.new("MorningSky")
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (*SKY_COLOUR, 1.0)
    bg.inputs["Strength"].default_value = 0.6
    scene.world = world
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = SUN_ENERGY
    sun_data.color = SUN_COLOUR
    sun_data.angle = math.radians(0.6)
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_mode = "XYZ"
    sun.rotation_euler = (math.radians(90.0 - SUN_ELEV_DEG), 0.0, math.radians(SUN_AZIMUTH_DEG))
    scene.collection.objects.link(sun)
    cam_data = bpy.data.cameras.new("Eye")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(75.0)
    cam = bpy.data.objects.new("Eye", cam_data)
    cam.location = (0.0, 0.0, a.eye)
    cam.rotation_mode = "XYZ"
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = a.width, a.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    cyc = scene.cycles
    cyc.samples = a.samples
    cyc.use_adaptive_sampling = False
    cyc.use_denoising = False   # denoising would soften the very shadows and glow compared (API review)
    cyc.volume_bounces = 4
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        used = 0
        for dev in prefs.devices:
            dev.use = dev.type == "OPTIX"
            used += int(dev.use)
        cyc.device = "GPU" if used else "CPU"
    except (KeyError, TypeError, AttributeError) as exc:
        log.warning("GPU setup failed (%s); rendering on the CPU", exc)
        cyc.device = "CPU"
    return cam


def lower_mean(path: str) -> float:
    """Mean luminance (display values) of the image's lower half, where the fog lies."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    rgb = px.reshape(h, w, 4)[: h // 2, :, :3]   # Blender image rows start at the bottom
    return float((rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).mean())


def aim(cam: bpy.types.Object, view: str) -> None:
    yaw = VIEWS[view]
    if yaw is None:
        # the sun light travels toward SUN_AZIMUTH; the sun itself sits at the opposite azimuth; face it, 25 deg off
        yaw = (SUN_AZIMUTH_DEG + 180.0) % 360.0 + 25.0
    cam.rotation_euler = (math.radians(90.0 - 4.0), 0.0, math.radians(yaw))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cam = setup(a)
    scene = bpy.context.scene
    meta = {"width": a.width, "height": a.height, "samples": a.samples, "sigma": a.sigma, "seed": a.seed,
            "eye": a.eye, "sun_elev_deg": SUN_ELEV_DEG, "sun_azimuth_deg": SUN_AZIMUTH_DEG, "views": {}}
    for view in VIEWS:
        aim(cam, view)
        meta["views"][view] = [math.degrees(v) for v in cam.rotation_euler]
        # shadow look first; its mean brightness on the fog area sets the flat look's emission (brightness matched)
        fog_box(a.sigma, "shadow", 1.0)
        path = os.path.join(a.out_dir, f"{view}_shadow.png")
        if not os.path.exists(path):
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            print(f"RENDERED {path}", flush=True)
        fpath = os.path.join(a.out_dir, f"{view}_flat.png")
        if not os.path.exists(fpath):
            # first at emission scale 1, then again scaled so its mean brightness below the horizon matches the shadow look
            tmp = os.path.join(a.out_dir, f"{view}_flat_unscaled.png")
            fog_box(a.sigma, "flat", 1.0)
            scene.render.filepath = tmp
            bpy.ops.render.render(write_still=True)
            target = lower_mean(path)
            ratio = target / max(lower_mean(tmp), 1e-4)
            # AgX and the lit ground do not scale with the emission, so one ratio is not exact: correct once more
            # from the scaled render and record how close the final one is (API review)
            for _ in range(2):
                fog_box(a.sigma, "flat", ratio)
                scene.render.filepath = fpath
                bpy.ops.render.render(write_still=True)
                got = lower_mean(fpath)
                if abs(got - target) <= 0.02 * target:
                    break
                ratio *= target / max(got, 1e-4)
            meta.setdefault("flat_scale", {})[view] = {"scale": ratio, "lower_mean_flat": got, "lower_mean_shadow": target}
            print(f"RENDERED {fpath} (emission scale {ratio:.3f}, lower-half mean {got:.4f} vs {target:.4f})", flush=True)
    with open(os.path.join(a.out_dir, "sun_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
