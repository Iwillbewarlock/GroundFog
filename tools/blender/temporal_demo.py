"""Demonstration of temporal accumulation on a noisy ground fog in front of a dense bush (Blender 5.2 Cycles).

Not the plugin's shader: a picture of the principle for the user (ledger 2026-11-16, temporal accumulation). Every frame is rendered
with very few samples and its own sampling seed, so the fog and the bush's edges are noisy in a different way each
frame (the plugin's march jitter is the same kind of per-frame noise). temporal_demo_post.py then blends the frames
(lerp(history, frame, a), history reprojected for the camera's turn, clamped to the frame's 3x3 neighbourhood) and puts
the raw and the accumulated frames side by side; a clean render (many samples) of each frame is the reference.

Scene (Z up, metres): flat dirt ground, the grass of grass_fog_scene.py, a dense bush wall 15-21 m ahead (small leaf
quads, the far ground and rocks visible through its gaps), a trunk, rocks 40-60 m away, the plugin's fog layer
(H = 1.8 m, density exp(-3 h / H), here 4x the default extinction so it reads clearly), morning sky and sun.
Camera 2 m up, 80 deg horizontal, 7 deg down. Frames 0..S-1 hold still; frames S..S+P-1 turn the camera's yaw by
6 deg x sin(2 pi k / P) (a pure rotation, so the post step can reproject without depth).

Usage:
  blender.exe -b --factory-startup --python tools/blender/temporal_demo.py -- --out-dir <dir>
      [--width 640 --height 400 --still 24 --pan 24 --samples 4 --ref-samples 256 --sigma 0.5616 --seed 7]
Writes raw_NNN.png (noisy, seed = frame) and ref_NNN.png (clean, one seed), and demo_meta.json (camera per frame).
Existing PNGs are skipped (a stopped run resumes).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grass_fog_scene as gfs  # noqa: E402

log = logging.getLogger("temporal_demo")
CAMERA_FOV_DEG = 80.0
CAMERA_PITCH_DOWN_DEG = 7.0
EYE_M = 2.0
PAN_YAW_DEG = 6.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--still", type=int, default=24)
    ap.add_argument("--pan", type=int, default=24)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--ref-samples", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=4.0 * 0.18 * 0.78)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    if a.width < 16 or a.height < 16 or a.still < 1 or a.pan < 0 or a.samples < 1 or a.ref_samples < 1:
        raise SystemExit(f"bad arguments: {a}")
    return a


def make_bush(seed: int) -> bpy.types.Object:
    """Small two-sided leaf quads scattered in a box 10 m wide, 6 m deep, 0.2-2.8 m high: gaps show what is behind."""
    rng = np.random.default_rng(seed)
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for _ in range(9000):
        c = np.array([rng.uniform(-5.0, 5.0), rng.uniform(15.0, 21.0), rng.uniform(0.2, 2.8)])
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v = np.cross(u, rng.normal(size=3))
        v /= np.linalg.norm(v)
        s = rng.uniform(0.06, 0.12)
        base = len(verts)
        for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = c + s * (du * u + dv * v)
            verts.append((float(p[0]), float(p[1]), float(p[2])))
        faces.append((base, base + 1, base + 2, base + 3))
    mesh = bpy.data.meshes.new("Bush")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Bush", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(gfs.diffuse("Leaves", (0.03, 0.07, 0.02)))
    return obj


def build_scene(a: argparse.Namespace) -> bpy.types.Object:
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 60.0, 0.0))
    ground = bpy.context.active_object
    ground.scale = (400.0, 400.0, 1.0)
    ground.data.materials.append(gfs.diffuse("Dirt", (0.09, 0.075, 0.055)))
    gfs.make_grass(a.seed)
    make_bush(a.seed + 1)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=8.0, location=(1.5, 13.0, 4.0))
    bpy.context.active_object.data.materials.append(gfs.diffuse("Bark", (0.08, 0.05, 0.03)))
    rng = np.random.default_rng(a.seed + 2)
    rock = gfs.diffuse("Rock", (0.25, 0.25, 0.24))
    for _ in range(14):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(rng.uniform(-30, 30), rng.uniform(40, 60), 0.0))
        o = bpy.context.active_object
        o.scale = (rng.uniform(3, 8), rng.uniform(3, 8), rng.uniform(2, 6))
        o.rotation_euler = (0.0, 0.0, rng.uniform(0, math.pi))
        o.data.materials.append(rock)
    gfs.make_fog("fog", a.sigma)

    world = bpy.data.worlds.new("Morning")
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.45, 0.55, 0.75, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.0
    sun_data.color = (1.0, 0.92, 0.8)
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(60.0), 0.0, math.radians(-30.0))
    scene.collection.objects.link(sun)

    cam_data = bpy.data.cameras.new("Eye")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(CAMERA_FOV_DEG)
    cam_data.clip_start, cam_data.clip_end = 0.1, 2000.0
    cam = bpy.data.objects.new("Eye", cam_data)
    cam.location = (0.0, 0.0, EYE_M)
    cam.rotation_mode = "XYZ"
    scene.collection.objects.link(cam)
    scene.camera = cam

    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = a.width, a.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    cyc = scene.cycles
    cyc.use_adaptive_sampling = False
    cyc.use_denoising = False
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = dev.type == "OPTIX"
        cyc.device = "GPU"
    except (KeyError, TypeError) as exc:
        log.warning("GPU setup failed (%s); rendering on the CPU", exc)
        cyc.device = "CPU"
    return cam


def yaw_deg(frame: int, a: argparse.Namespace) -> float:
    if frame < a.still:
        return 0.0
    return PAN_YAW_DEG * math.sin(2.0 * math.pi * (frame - a.still) / max(a.pan, 1))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cam = build_scene(a)
    scene = bpy.context.scene
    frames = a.still + a.pan
    meta = {"width": a.width, "height": a.height, "fov_deg": CAMERA_FOV_DEG, "pitch_down_deg": CAMERA_PITCH_DOWN_DEG,
            "still": a.still, "pan": a.pan, "seed": a.seed, "samples": a.samples, "ref_samples": a.ref_samples, "sigma": a.sigma,
            "yaw_deg": [yaw_deg(f, a) for f in range(frames)]}
    with open(os.path.join(a.out_dir, "demo_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    for f in range(frames):
        cam.rotation_euler = (math.radians(90.0 - CAMERA_PITCH_DOWN_DEG), 0.0, math.radians(-yaw_deg(f, a)))
        for kind, spp, seed in (("raw", a.samples, f + 1), ("ref", a.ref_samples, 1)):
            path = os.path.join(a.out_dir, f"{kind}_{f:03d}.png")
            if os.path.exists(path):
                continue
            scene.cycles.samples = spp
            scene.cycles.seed = seed
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            print(f"RENDERED {kind} {f} spp {spp} seed {seed} yaw {yaw_deg(f, a):.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
