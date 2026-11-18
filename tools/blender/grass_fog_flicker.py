"""Frame-to-frame flicker of mid-distance grass in the GroundFog layer, in Blender 5.2 Cycles (a physical reference).

Question (DEBUG-LEDGER.md 2026-11-16, H-mid-A / H-mid-B): with the game's kind of aliasing (one point sample per
pixel at a sub-pixel jitter that changes every frame, thin blades swaying a little), how much does the physical fog
raise the frame-to-frame luminance change of grass per distance band, and how much is left when every low surface
takes the fog of the bare ground behind it ("sink")?

The scene is the one of grass_fog_scene.py (imported, not copied): flat ground, seeded grass clumps and fern fans
6-70 m, the plugin's layer (H = 1.8 m, density exp(-3 h/H) cut at H, extinction 0.18 x 0.78 per m at unit mean
density) as a volume box, night sky and moon, camera 80 deg horizontal looking along +Y, 7 deg down. Z up, metres.

Per frame k (0..N-1) only two things change:
  - camera sub-pixel jitter: Halton(2,3) at index k+1, minus 0.5 px, applied as lens shift;
  - blade sway: every blade's vertices move sideways by A_b sin(phi_b + 2 pi k / N) (t^2), t = height fraction,
    A_b = U(0.5, 1) x sway amplitude, direction and phase per blade from a separate seeded RNG.
The Cycles sampling seed is the same for every frame; denoising and adaptive sampling are off.

Renders (one Blender process per eye height, each written as a multilayer EXR and converted to <name>.npz):
  nofog_fNN   grass, no fog            (every frame also keeps Position, Depth and object index passes)
  fog_fNN     grass, fog
  noise_fog_s<seed2>, noise_nofog_s<seed2>   frame 0 again with a second sampling seed (noise floor)
  noise_bare_fog_f01   bare fogged ground at the frame-1 jitter, first seed (temporal noise floor)
  determinism_fog   frame 0 with the first seed again (only with --determinism)
  bare_nofog, bare_fog, bare_fog_black       no grass and no posts, frame-0 jitter; black = ground albedo 0 and
                                             specular 0
Two dark posts (0.5 x 0.5 x 2 m, at 8 m and 20 m) stand in every render with grass, for a vertical-surface crop.
The sink and drop looks are composited later from these (analyze_flicker.py next to the outputs), not rendered.
Pixel filter: GAUSSIAN at the Cycles minimum width 0.01 px, i.e. every sample of a pixel lands on the same point
(Cycles ignores filter_width for BOX and uses 1 px, an area average, which would hide the aliasing).
meta.json in the output folder records the settings; a resume with different settings is refused.

Usage:
  blender.exe -b --factory-startup --python tools/blender/grass_fog_flicker.py -- --out-dir <dir> [--eye 2.0]
      [--width 1280 --height 800 --frames 12 --samples 256 --seed 1 --seed2 2 --grass-seed 7 --sway-seed 11
       --sway-amp 0.04 --filter-type GAUSSIAN --filter-width 0.01 --sigma 0.1404 --sets nofog,fog,noise,bare
       --determinism]
Existing .npz outputs are skipped, so a stopped run resumes.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grass_fog_scene as gfs  # noqa: E402  (same folder; reuses make_grass, make_fog, diffuse unchanged)

log = logging.getLogger("grass_fog_flicker")

GRASS_INDEX = 1
GROUND_INDEX = 2
POST_INDEX = 3
POSTS = ((-2.0, 8.0), (3.0, 20.0))   # (x, y) of the post centres, m
POST_SIZE = (0.5, 0.5, 2.0)          # width, depth, height, m
POST_COLOR = (0.03, 0.03, 0.03)
VERTS_PER_BLADE = 8          # grass_fog_scene.make_grass: 4 rows x 2 vertices per blade
ROWS_PER_BLADE = 4
CAMERA_PITCH_DOWN_DEG = 7.0
CAMERA_FOV_DEG = 80.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="grass_fog_flicker.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eye", type=float, default=2.0, help="camera height over the ground, m")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1, help="Cycles sampling seed, the same for every frame")
    ap.add_argument("--seed2", type=int, default=2, help="second sampling seed for the noise floor")
    ap.add_argument("--grass-seed", type=int, default=7, help="grass layout seed (grass_fog_scene default 7)")
    ap.add_argument("--sway-seed", type=int, default=11)
    ap.add_argument("--sway-amp", type=float, default=0.04, help="blade tip sway amplitude, m")
    ap.add_argument("--filter-type", default="GAUSSIAN", choices=["BOX", "GAUSSIAN", "BLACKMAN_HARRIS"])
    ap.add_argument("--filter-width", type=float, default=0.01, help="px; 0.01 (Cycles minimum) = point sampling; BOX ignores it (1 px)")
    ap.add_argument("--sigma", type=float, default=0.18 * 0.78, help="extinction per m at unit mean density")
    ap.add_argument("--sets", default="nofog,fog,noise,bare", help="comma list of nofog,fog,noise,bare")
    ap.add_argument("--determinism", action="store_true", help="also render fog frame 0 again with --seed")
    args = ap.parse_args(argv)
    bad = set(args.sets.split(",")) - {"nofog", "fog", "noise", "bare"}
    if bad:
        raise SystemExit(f"unknown --sets entries {sorted(bad)}: use nofog, fog, noise, bare")
    if args.frames < 1 or args.samples < 1 or args.width < 8 or args.height < 8:
        raise SystemExit(f"frames {args.frames}, samples {args.samples}, size {args.width}x{args.height}: too small")
    return args


def halton(index: int, base: int) -> float:
    """Radical inverse of index in the given base (index >= 1)."""
    f, r = 1.0, 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r


def jitter_px(frame: int) -> tuple[float, float]:
    """Sub-pixel offset of frame k: Halton(2,3) at index k+1, centred on the pixel (-0.5..0.5 px)."""
    return halton(frame + 1, 2) - 0.5, halton(frame + 1, 3) - 0.5


class Sway:
    """Per-blade sideways sway, scaled by t^2 so the base stays put and the tip moves most."""

    def __init__(self, base_co: np.ndarray, seed: int, amp: float, frames: int) -> None:
        if base_co.shape[0] % VERTS_PER_BLADE:
            raise ValueError(f"grass mesh has {base_co.shape[0]} vertices, not a multiple of {VERTS_PER_BLADE}")
        self.base = base_co
        self.frames = frames
        blades = base_co.shape[0] // VERTS_PER_BLADE
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2.0 * math.pi, blades)
        self.phase = rng.uniform(0.0, 2.0 * math.pi, blades)
        amp_b = amp * rng.uniform(0.5, 1.0, blades)
        row = (np.arange(base_co.shape[0]) % VERTS_PER_BLADE) // 2
        t2 = (row / (ROWS_PER_BLADE - 1)) ** 2
        blade = np.arange(base_co.shape[0]) // VERTS_PER_BLADE
        self.blade = blade
        self.dx = (amp_b * np.cos(theta))[blade] * t2
        self.dy = (amp_b * np.sin(theta))[blade] * t2

    def coords(self, frame: int) -> np.ndarray:
        s = np.sin(self.phase + 2.0 * math.pi * frame / self.frames)[self.blade]
        co = self.base.copy()
        co[:, 0] += self.dx * s
        co[:, 1] += self.dy * s
        return co


def build_scene(args: argparse.Namespace) -> dict:
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 60.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.scale = (400.0, 400.0, 1.0)
    ground.data.materials.append(gfs.diffuse("Dirt", (0.07, 0.06, 0.05)))
    ground.pass_index = GROUND_INDEX
    grass = gfs.make_grass(args.grass_seed)
    grass.pass_index = GRASS_INDEX
    posts = []
    post_mat = gfs.diffuse("PostDark", POST_COLOR)
    for i, (px, py) in enumerate(POSTS):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(px, py, 0.5 * POST_SIZE[2]))
        post = bpy.context.active_object
        post.name = f"Post{i}"
        post.scale = POST_SIZE
        post.data.materials.append(post_mat)
        post.pass_index = POST_INDEX
        posts.append(post)
    gfs.make_fog("fog", args.sigma)
    fog_box = bpy.data.objects["FogBox"]

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
    cam_data.angle = math.radians(CAMERA_FOV_DEG)
    cam_data.clip_start, cam_data.clip_end = 0.1, 2000.0
    cam = bpy.data.objects.new("Eye", cam_data)
    cam.location = (0.0, 0.0, args.eye)
    cam.rotation_euler = (math.radians(90.0 - CAMERA_PITCH_DOWN_DEG), 0.0, 0.0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    setup_render(scene, args)
    base = np.empty(len(grass.data.vertices) * 3, dtype=np.float64)
    grass.data.vertices.foreach_get("co", base)
    base = base.reshape(-1, 3)
    log.info("grass: %d vertices, %d blades, max z %.3f m", base.shape[0], base.shape[0] // VERTS_PER_BLADE,
             float(base[:, 2].max()))
    dirt = next(n for n in ground.data.materials[0].node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    return {"dirt_spec": float(dirt.inputs["Specular IOR Level"].default_value), "posts": posts,
            "scene": scene, "grass": grass, "fog": fog_box, "cam": cam, "dirt": dirt,
            "sway": Sway(base, args.sway_seed, args.sway_amp, args.frames)}


def setup_render(scene: bpy.types.Scene, args: argparse.Namespace) -> None:
    scene.render.engine = "CYCLES"
    cyc = scene.cycles
    cyc.samples = args.samples
    cyc.use_denoising = False
    cyc.use_adaptive_sampling = False
    cyc.use_animated_seed = False
    cyc.seed = args.seed
    cyc.pixel_filter_type = args.filter_type
    cyc.filter_width = args.filter_width
    cyc.volume_step_rate = 0.25
    cyc.volume_max_steps = 512
    cyc.volume_bounces = 4
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = True
        cyc.device = "GPU"
    except TypeError:
        cyc.device = "CPU"
    scene.render.use_persistent_data = True
    scene.render.film_transparent = False
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    vl = scene.view_layers[0]
    vl.use_pass_combined = True
    vl.use_pass_z = True
    vl.use_pass_position = True
    vl.use_pass_object_index = True
    ims = scene.render.image_settings
    ims.media_type = "MULTI_LAYER_IMAGE"
    ims.file_format = "OPEN_EXR_MULTILAYER"   # the only format offered for MULTI_LAYER_IMAGE in 5.2
    ims.color_depth = "32"
    ims.exr_codec = "ZIP"
    if hasattr(ims, "use_exr_interleave"):
        ims.use_exr_interleave = True     # one part with every pass; exr_to_npz also reads all parts anyway


def set_jitter(cam: bpy.types.Object, frame: int, width: int) -> tuple[float, float]:
    """Lens shift is in units of the frame's larger side (the width here, sensor fit HORIZONTAL), for x and y."""
    jx, jy = jitter_px(frame)
    cam.data.shift_x = jx / width
    cam.data.shift_y = jy / width
    return jx, jy


def check_shift_units(ctx: dict, args: argparse.Namespace) -> None:
    """Self-test: a shift of 1/width must move a projected point by exactly 1 px in x and in y."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    scene, cam = ctx["scene"], ctx["cam"]
    pt = Vector((1.3, 20.0, 0.4))
    cam.data.shift_x = cam.data.shift_y = 0.0
    bpy.context.view_layer.update()
    p0 = world_to_camera_view(scene, cam, pt)
    cam.data.shift_x = cam.data.shift_y = 1.0 / args.width
    bpy.context.view_layer.update()
    p1 = world_to_camera_view(scene, cam, pt)
    dx, dy = (p1.x - p0.x) * args.width, (p1.y - p0.y) * args.height
    ok = abs(abs(dx) - 1.0) < 1e-3 and abs(abs(dy) - 1.0) < 1e-3
    log.info("SELFTEST shift units: 1/width shift moves the point %.4f px in x, %.4f px in y -> %s",
             dx, dy, "PASS" if ok else "FAIL")
    cam.data.shift_x = cam.data.shift_y = 0.0


PASS_ALIASES = {"Combined": ("Combined",), "Position": ("Position",), "Depth": ("Depth",),
                "Object Index": ("Object Index", "IndexOB")}


def find_channels(names: list[str], pass_name: str, comps: str) -> list[int]:
    """Indices of pass_name.<comp> among full channel names ('<part or layer>.<pass>.<comp>' or '<pass>.<comp>')."""
    out = []
    for c in comps:
        hits = [i for i, n in enumerate(names) for alias in PASS_ALIASES[pass_name]
                if n == f"{alias}.{c}" or n.endswith(f".{alias}.{c}")]
        if len(hits) != 1:
            raise RuntimeError(f"EXR channel {pass_name}.{c}: found {len(hits)} matches in {names}")
        out.append(hits[0])
    return out


def exr_to_npz(exr: str, npz: str, keep_data: bool) -> None:
    import OpenImageIO as oiio
    inp = oiio.ImageInput.open(exr)
    if inp is None:
        raise RuntimeError(f"cannot open {exr}: {oiio.geterror()}")
    names: list[str] = []
    planes: list[np.ndarray] = []
    try:
        sub = 0
        while inp.seek_subimage(sub, 0):          # multipart EXR: one subimage per part
            spec = inp.spec()
            part = spec.getattribute("name") or ""
            px = inp.read_image(sub, 0, 0, spec.nchannels, "float")
            if px is None:
                raise RuntimeError(f"cannot read part {sub} of {exr}: {inp.geterror()}")
            planes.append(np.asarray(px, dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels))
            names += [f"{part}.{c}" if part and not c.startswith(part + ".") else c for c in spec.channelnames]
            sub += 1
    finally:
        inp.close()
    data = np.concatenate(planes, axis=2)
    arrays = {"rgb": data[..., find_channels(names, "Combined", "RGB")]}
    if keep_data:
        arrays["position"] = data[..., find_channels(names, "Position", "XYZ")]
        arrays["depth"] = data[..., find_channels(names, "Depth", "Z")[0]]
        arrays["index"] = data[..., find_channels(names, "Object Index", "X")[0]]
    np.savez(npz, **arrays)
    log.info("EXR channels %s", names)


def render(ctx: dict, args: argparse.Namespace, name: str, *, fog: bool, grass: bool, frame: int,
           seed: int, keep_data: bool, black_ground: bool = False) -> None:
    npz = os.path.join(args.out_dir, f"{name}.npz")
    if os.path.exists(npz):
        log.info("skip %s (exists)", npz)
        return
    scene = ctx["scene"]
    ctx["fog"].hide_render = not fog
    ctx["grass"].hide_render = not grass
    for post in ctx["posts"]:
        post.hide_render = not grass
    ctx["grass"].data.vertices.foreach_set("co", ctx["sway"].coords(frame).astype(np.float32).ravel())
    ctx["grass"].data.update()
    jx, jy = set_jitter(ctx["cam"], frame, args.width)
    scene.cycles.seed = seed
    ctx["dirt"].inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0) if black_ground else (0.07, 0.06, 0.05, 1.0)
    ctx["dirt"].inputs["Specular IOR Level"].default_value = 0.0 if black_ground else ctx["dirt_spec"]
    exr = os.path.join(args.out_dir, f"{name}.exr")
    scene.render.filepath = exr
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    exr_to_npz(exr, npz, keep_data)
    os.remove(exr)                      # the .npz keeps every pass that is needed
    log.info("%s: %.1f s (jitter %+.3f %+.3f px, seed %d)", name, time.time() - t0, jx, jy, seed)
    print(f"RENDERED {name} {args.width}x{args.height} spp {args.samples} eye {args.eye} "
          f"device {scene.cycles.device} -> {npz}", flush=True)


META_KEYS = ("eye", "width", "height", "frames", "samples", "seed", "seed2", "grass_seed", "sway_seed", "sway_amp",
             "filter_type", "filter_width", "sigma")


def check_meta(args: argparse.Namespace) -> None:
    """Record the settings in meta.json; refuse to add renders to a folder made with other settings."""
    path = os.path.join(args.out_dir, "meta.json")
    meta = {k: getattr(args, k) for k in META_KEYS}
    meta["posts"] = [list(p) for p in POSTS]
    if os.path.exists(path):
        with open(path, encoding="ascii") as fh:
            old = json.load(fh)
        diff = {k: (old.get(k), meta[k]) for k in meta if old.get(k) != meta[k]}
        if diff:
            raise SystemExit(f"{path} was made with other settings {diff}: use a new --out-dir")
    else:
        with open(path, "w", encoding="ascii") as fh:
            json.dump(meta, fh, indent=1)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(args.out_dir, "render.log"), encoding="ascii")])
    log.info("args %s", vars(args))
    check_meta(args)
    ctx = build_scene(args)
    check_shift_units(ctx, args)
    sets = args.sets.split(",")
    for f in range(args.frames):
        log.info("frame %d jitter %+.4f %+.4f px", f, *jitter_px(f))
    if "nofog" in sets:
        for f in range(args.frames):
            render(ctx, args, f"nofog_f{f:02d}", fog=False, grass=True, frame=f, seed=args.seed, keep_data=True)
    if "fog" in sets:
        for f in range(args.frames):
            render(ctx, args, f"fog_f{f:02d}", fog=True, grass=True, frame=f, seed=args.seed, keep_data=f == 0)
    if "noise" in sets:
        render(ctx, args, f"noise_fog_s{args.seed2}", fog=True, grass=True, frame=0, seed=args.seed2,
               keep_data=False)
        render(ctx, args, f"noise_nofog_s{args.seed2}", fog=False, grass=True, frame=0, seed=args.seed2,
               keep_data=False)
        # temporal noise floor with the fixed seed: bare fogged ground at the frame-1 jitter (nothing else changes)
        render(ctx, args, "noise_bare_fog_f01", fog=True, grass=False, frame=1, seed=args.seed, keep_data=False)
    if args.determinism:
        render(ctx, args, "determinism_fog", fog=True, grass=True, frame=0, seed=args.seed, keep_data=False)
    if "bare" in sets:
        render(ctx, args, "bare_nofog", fog=False, grass=False, frame=0, seed=args.seed, keep_data=True)
        render(ctx, args, "bare_fog", fog=True, grass=False, frame=0, seed=args.seed, keep_data=False)
        render(ctx, args, "bare_fog_black", fog=True, grass=False, frame=0, seed=args.seed, keep_data=False,
               black_ground=True)
    log.info("done")


if __name__ == "__main__":
    main()
