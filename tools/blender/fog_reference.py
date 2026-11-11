"""Cycles reference for the GroundFog raymarch (Blender 5.2, run headless).

Usage:
  blender.exe -b --factory-startup --python tools/blender/fog_reference.py -- <data dir> [size] [samples] [step rate]

Reads <data dir>/columns.bin and cameras.json written by RaymarchExport.exe, builds the same fog as a Cycles
volume, renders each camera, and compares with gpu_<k>.bin.

Conventions (pinned): Z up, metres, 1 game unit = 1/70 m. Ground plane z = 0. The fog is pure absorption over a
white emissive ground and a white world, so a pixel value is the transmittance T and fog = 1 - T, the same
quantity the GPU march writes. DirectX LookToLH puts screen right at -X for a camera looking +Y with Z up, and
DirectX image row 0 is the top; Blender is right-handed with row 0 at the bottom, so the Blender image is flipped
in both axes before comparing.

Outputs in <data dir>: cycles_<k>.exr, compare_<k>.png (GPU | Cycles | |diff| x4), compare.json (statistics).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

UNITS_PER_METRE = 70.0


def parse_args() -> tuple[Path, int | None, int, float]:
    """size defaults to the GPU image size; a smaller size must divide it (GPU blocks are averaged to match)."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:
        raise SystemExit("usage: ... --python fog_reference.py -- <data dir> [size] [samples]")
    data = Path(argv[0])
    size = int(argv[1]) if len(argv) > 1 else None
    samples = int(argv[2]) if len(argv) > 2 else 64
    step_rate = float(argv[3]) if len(argv) > 3 else 0.1
    return data, size, samples, step_rate


def load_columns(data: Path, res: int) -> np.ndarray:
    cols = np.fromfile(data / "columns.bin", dtype=np.float32)
    if cols.size != res * res * 4:
        raise SystemExit(f"columns.bin has {cols.size} floats, expected {res * res * 4}")
    return cols.reshape(res, res, 4)  # [row = y from the field origin, x, rgba]


def make_columns_image(cols: np.ndarray) -> bpy.types.Image:
    res = cols.shape[0]
    img = bpy.data.images.new("FogColumns", width=res, height=res, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    img.alpha_mode = "CHANNEL_PACKED"  # keep RGB independent of A (no premultiply)
    # Blender rows start at the bottom (v = 0), the D3D texture row 0 is v = 0 too: same order.
    img.pixels.foreach_set(cols.astype(np.float32).ravel())
    img.pack()
    return img


def node(nodes, kind, location, **props):
    n = nodes.new(kind)
    n.location = location
    for k, v in props.items():
        setattr(n, k, v)
    return n


def make_fog_material(img: bpy.types.Image, origin: float, extent: float, opacity: float, steepness: float,
                      scatter: dict | None = None) -> bpy.types.Material:
    """Fog volume with the GroundFog density. scatter=None: black absorber (the comparison). Otherwise a Principled
    Volume for a picture, with keys color (RGB) and anisotropy; its density is the same extinction per metre."""
    mat = bpy.data.materials.new("GroundFogVolume")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    for n in list(nodes):
        nodes.remove(n)
    out = node(nodes, "ShaderNodeOutputMaterial", (1400, 0))
    geo = node(nodes, "ShaderNodeNewGeometry", (-1400, 0))
    sep = node(nodes, "ShaderNodeSeparateXYZ", (-1200, 0))
    links.new(geo.outputs["Position"], sep.inputs[0])

    def math_node(op, a, b=None, loc=(0, 0), clamp=False):
        m = node(nodes, "ShaderNodeMath", loc, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = float(v)
            else:
                links.new(v, m.inputs[i])
        return m.outputs[0]

    # uv = (P * 70 - origin) / extent
    u = math_node("DIVIDE", math_node("SUBTRACT", math_node("MULTIPLY", sep.outputs["X"], UNITS_PER_METRE, (-1000, 100)), origin, (-850, 100)), extent, (-700, 100))
    v = math_node("DIVIDE", math_node("SUBTRACT", math_node("MULTIPLY", sep.outputs["Y"], UNITS_PER_METRE, (-1000, -50)), origin, (-850, -50)), extent, (-700, -50))
    comb = node(nodes, "ShaderNodeCombineXYZ", (-550, 0))
    links.new(u, comb.inputs["X"])
    links.new(v, comb.inputs["Y"])
    tex = node(nodes, "ShaderNodeTexImage", (-380, 0), image=img, interpolation="Linear", extension="EXTEND")
    links.new(comb.outputs[0], tex.inputs["Vector"])
    rgb = node(nodes, "ShaderNodeSeparateColor", (-150, 0))
    links.new(tex.outputs["Color"], rgb.inputs[0])
    H = rgb.outputs[0]       # column height, game units
    W = rgb.outputs[1]       # optical thickness, metres at unit density
    known = tex.outputs["Alpha"]
    # z holds ground z * known (game units); the HLSL divides by known.
    ground = math_node("DIVIDE", rgb.outputs[2], math_node("MAXIMUM", known, 1e-3, (-150, -400)), (0, -400))

    h = math_node("SUBTRACT", math_node("MULTIPLY", sep.outputs["Z"], UNITS_PER_METRE, (0, -250)), ground, (150, -250))
    ratio = math_node("DIVIDE", h, math_node("MAXIMUM", H, 1e-3, (150, -100)), (300, -200))
    # Profile exp(-a h/H), a held to >= 1e-3 as RaymarchShaders kMinSteepness; ground factor a / (1 - exp(-a)) so the
    # vertical integral of the density is W.
    a = max(float(steepness), 1e-3)
    ground_factor = a / -math.expm1(-a)
    profile = math_node("EXPONENT", math_node("MULTIPLY", ratio, -a, (450, -200)), None, (600, -200))
    # Same tests as the HLSL skip: known >= 0.5, H >= 1, h >= 0, h < H.
    def at_least(value, threshold, loc):
        return math_node("SUBTRACT", 1.0, math_node("LESS_THAN", value, threshold, loc), (loc[0] + 100, loc[1]))
    inside = math_node("MULTIPLY",
                       math_node("MULTIPLY", at_least(h, 0.0, (400, -350)), math_node("LESS_THAN", ratio, 1.0, (450, -450)), (600, -400)),
                       math_node("MULTIPLY", at_least(H, 1.0, (400, -550)), at_least(known, 0.05, (400, -650)), (600, -600)),
                       (750, -500))
    # rho = W / (H / 70) * a / (1 - exp(-a)) * exp(-a h/H) ; extinction per metre = opacity * rho
    rho = math_node("MULTIPLY", math_node("DIVIDE", W, math_node("DIVIDE", math_node("MAXIMUM", H, 1e-3, (450, 100)), UNITS_PER_METRE, (600, 100)), (750, 50)),
                    ground_factor, (900, 50))
    density = math_node("MULTIPLY", math_node("MULTIPLY", math_node("MULTIPLY", rho, profile, (1050, 0)), inside, (1150, -100)),
                        math_node("MULTIPLY", known, opacity, (1150, -250)), (1250, -100))
    if scatter is None:
        shader = node(nodes, "ShaderNodeVolumeAbsorption", (1250, 100))
        shader.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # black absorber: T = exp(-integral of density)
    else:
        shader = node(nodes, "ShaderNodeVolumePrincipled", (1250, 100))
        shader.inputs["Color"].default_value = (*scatter["color"], 1.0)
        shader.inputs["Anisotropy"].default_value = float(scatter["anisotropy"])
    links.new(density, shader.inputs["Density"])
    links.new(shader.outputs[0], out.inputs["Volume"])
    return mat


def emissive_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = node(nt.nodes, "ShaderNodeEmission", (0, 0))
    em.inputs["Color"].default_value = (1, 1, 1, 1)
    em.inputs["Strength"].default_value = 1.0
    out = node(nt.nodes, "ShaderNodeOutputMaterial", (200, 0))
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def setup_cycles(scene: bpy.types.Scene, size: int, samples: int, step_rate: float) -> None:
    try:
        scene.render.engine = "CYCLES"
    except TypeError as err:
        raise SystemExit(f"Cycles not available: {err}")
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(d.type == backend for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                print(f"[fog_reference] Cycles device {backend}")
                break
        except TypeError:
            continue
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    # Biased ray marching with fine steps (5.x defaults to unbiased null scattering, whose noise would mix into the
    # comparison). Pixel filter narrowed so each pixel is close to one centre ray, like the GPU march (still an
    # approximation, not an exact point sample).
    if hasattr(scene.cycles, "volume_biased"):
        scene.cycles.volume_biased = True
    scene.cycles.volume_step_rate = step_rate
    scene.cycles.volume_max_steps = 4096
    scene.cycles.pixel_filter_type = "GAUSSIAN"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    world = bpy.data.worlds.new("White")
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world


def build_scene(cfg: dict, cols: np.ndarray, size: int, samples: int, step_rate: float) -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    setup_cycles(scene, size, samples, step_rate)

    origin = float(cfg["columns"]["origin"])
    extent = float(cfg["columns"]["extent"])
    half_m = extent / 2.0 / UNITS_PER_METRE
    centre_m = (origin + extent / 2.0) / UNITS_PER_METRE
    top_m = float(cols[..., 0].max()) * 1.02 / UNITS_PER_METRE + 0.01

    bpy.ops.mesh.primitive_plane_add(size=half_m * 8.0, location=(centre_m, centre_m, 0.0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(emissive_material("WhiteGround"))

    # Volume box: the field square, from just below the ground to the highest column top.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(centre_m, centre_m, top_m / 2.0 - 0.005))
    box = bpy.context.active_object
    box.name = "FogVolume"
    box.scale = (half_m * 2.0, half_m * 2.0, top_m + 0.01)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    img = make_columns_image(cols)
    box.data.materials.append(make_fog_material(img, origin, extent, float(cfg["opacity_per_metre"]), float(cfg["steepness"])))

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(float(cfg["fov_y_deg"]))
    cam_data.clip_start = 15.0 / UNITS_PER_METRE
    cam_data.clip_end = 1000.0
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return scene


def render_camera(scene: bpy.types.Scene, cfg: dict, k: int, data: Path) -> np.ndarray:
    cam = scene.camera
    c = cfg["cameras"][k]
    pos = Vector(c["pos"]) / UNITS_PER_METRE
    target = Vector(cfg["target"]) / UNITS_PER_METRE
    cam.location = pos
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (target - pos).to_track_quat("-Z", "Y")
    path = data / f"cycles_{k}.exr"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    t = px.reshape(h, w, 4)[..., 0]
    # Blender row 0 = bottom, DirectX row 0 = top; DirectX LH screen right = Blender screen left.
    return np.flip(np.flip(t, axis=0), axis=1)


def save_side_by_side(gpu: np.ndarray, ref: np.ndarray, path: Path) -> None:
    h, w = gpu.shape
    canvas = np.ones((h, w * 3 + 8, 4), dtype=np.float32)
    canvas[:, :w, :3] = (1.0 - gpu)[..., None]
    canvas[:, w + 4:2 * w + 4, :3] = (1.0 - ref)[..., None]
    canvas[:, 2 * w + 8:, :3] = (1.0 - np.clip(np.abs(gpu - ref) * 4.0, 0, 1))[..., None]
    img = bpy.data.images.new(path.stem, width=canvas.shape[1], height=h, alpha=True, float_buffer=False)
    img.pixels.foreach_set(np.flip(canvas, axis=0).ravel())  # back to Blender row order
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()


def main() -> None:
    data, size, samples, step_rate = parse_args()
    cfg = json.loads((data / "cameras.json").read_text(encoding="utf-8"))
    gpu_size = int(cfg["size"])
    size = size or gpu_size
    if gpu_size % size:
        raise SystemExit(f"size {size} must divide the GPU image size {gpu_size}")
    cols = load_columns(data, int(cfg["columns"]["res"]))
    scene = build_scene(cfg, cols, size, samples, step_rate)
    stats = []
    for k in range(len(cfg["cameras"])):
        t = render_camera(scene, cfg, k, data)
        ref = 1.0 - t
        gpu = np.fromfile(data / f"gpu_{k}.bin", dtype=np.float32).reshape(gpu_size, gpu_size)
        if gpu_size != size:  # block means: a block's centre is the Cycles pixel centre
            f = gpu_size // size
            gpu = gpu.reshape(size, f, size, f).mean(axis=(1, 3))
        diff = np.abs(gpu - ref)
        mask = (gpu > 0.01) | (ref > 0.01)
        s = {
            "camera": k,
            "gpu_mean_fog": float(gpu[mask].mean()) if mask.any() else 0.0,
            "cycles_mean_fog": float(ref[mask].mean()) if mask.any() else 0.0,
            "mean_abs_diff": float(diff[mask].mean()) if mask.any() else 0.0,
            "p95_abs_diff": float(np.percentile(diff[mask], 95)) if mask.any() else 0.0,
            "fog_pixels": int(mask.sum()),
        }
        stats.append(s)
        print("[fog_reference]", json.dumps(s))
        save_side_by_side(gpu, ref, data / f"compare_{k}.png")
    (data / "compare.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
