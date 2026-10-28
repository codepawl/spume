# SPDX-License-Identifier: Apache-2.0
"""Headless check of the Spume addon.

    blender -b -t 6 --python blender/test_addon.py -- <cache folder> <previews folder>

Builds an 8 x 8 x 10 m domain with two inflows (a vertical jet and a lateral torrent from the low-X wall) and a cube obstacle held above
the jet, bakes a preview with the real solver, loads it, and checks:
  - the cube deflects the jet: no liquid vertex lies inside the cube, the jet reaches its underside and nothing climbs past its top;
  - the second inflow is read from its real position;
  - whitewater kinds come from the solver (spray_kinds.bin);
  - progress comes from the `frame i/n` lines.
Then it renders three Eevee stills into the previews folder.
"""

import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import addon_utils  # noqa: E402

cache_folder, previews_folder = (Path(argument) for argument in sys.argv[sys.argv.index("--") + 1 :])
previews_folder.mkdir(parents=True, exist_ok=True)

JET_X = -1.0
CUBE_CENTRE = (JET_X, 0.0, 5.2)
CUBE_SIZE = 1.8
CUBE_BOTTOM = CUBE_CENTRE[2] - CUBE_SIZE / 2
CUBE_TOP = CUBE_CENTRE[2] + CUBE_SIZE / 2
STILL_FRAMES = (12, 24, 40)


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        raise SystemExit(1)


def principled(name, base_color, roughness, **inputs):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = base_color
    shader.inputs["Roughness"].default_value = roughness
    for key, value in inputs.items():
        shader.inputs[key].default_value = value
    return material


def build_world(scene):
    scene.world = bpy.data.worlds.new("Sky")
    scene.world.use_nodes = True
    nodes, links = scene.world.node_tree.nodes, scene.world.node_tree.links
    sky = nodes.new("ShaderNodeTexSky")
    sky_types = [item.identifier for item in sky.bl_rna.properties["sky_type"].enum_items]
    sky.sky_type = "MULTIPLE_SCATTERING" if "MULTIPLE_SCATTERING" in sky_types else "NISHITA"
    sky.sun_elevation = math.radians(24)
    sky.sun_rotation = math.radians(40)
    sky.sun_size = math.radians(1.2)
    links.new(sky.outputs["Color"], nodes["Background"].inputs["Color"])
    nodes["Background"].inputs["Strength"].default_value = 0.55


def build_scene():
    for stale in list(bpy.data.objects):
        bpy.data.objects.remove(stale)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, 40

    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 5))
    domain = bpy.context.object
    domain.name = "Domain"
    domain.scale = (4, 4, 5)  # 8 x 8 x 10 m
    domain.display_type = "WIRE"
    domain.hide_render = True
    domain.spume.role = "DOMAIN"
    domain.spume.cell_size = 0.08
    domain.spume.mesh_cell_size = 0.1
    domain.spume.frames = 40
    domain.spume.preview_frames = 40
    domain.spume.preview_scale = 1.0
    domain.spume.cache_folder = str(cache_folder)

    jet = bpy.data.objects.new("Jet", None)
    scene.collection.objects.link(jet)
    jet.location = (JET_X, 0, 0)
    jet.spume.role = "INFLOW"
    jet.spume.direction = (0, 0, 1)
    jet.spume.speed = 17.0
    jet.spume.opening_diameter = 0.7
    jet.spume.feed = 3.0

    torrent = bpy.data.objects.new("Torrent", None)
    scene.collection.objects.link(torrent)
    torrent.location = (-4, 0.2, 1.1)
    torrent.spume.role = "INFLOW"
    torrent.spume.direction = (1, 0, 0)
    torrent.spume.speed = 7.0
    torrent.spume.opening_diameter = 1.1
    torrent.spume.feed = 3.0

    bpy.ops.mesh.primitive_cube_add(size=CUBE_SIZE, location=CUBE_CENTRE)
    cube = bpy.context.object
    cube.name = "Obstacle"
    cube.spume.role = "OBSTACLE"
    cube.data.materials.append(principled("Block", (0.05, 0.055, 0.065, 1.0), 0.45))

    bpy.ops.mesh.primitive_plane_add(size=600, location=(0, 0, -0.01))
    bpy.context.object.name = "Ground"
    bpy.context.object.data.materials.append(principled("Ground", (0.035, 0.04, 0.045, 1.0), 0.3))

    build_world(scene)
    bpy.ops.object.light_add(type="SUN", location=(5, -5, 12))
    sun = bpy.context.object
    sun.rotation_euler = (math.radians(55), 0.0, math.radians(40))
    sun.data.energy = 3.0
    sun.data.angle = math.radians(1.5)

    bpy.ops.object.camera_add(location=(6.2, -7.8, 2.4))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((-1.0, 0.0, 3.7)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 22
    scene.camera = camera
    return scene, domain


def check_cancel(domain):
    bpy.context.view_layer.objects.active = domain
    check(bpy.ops.spume.bake(preview=True) == {"FINISHED"}, "second bake started for the cancel check")
    time.sleep(3)
    check(bpy.ops.spume.cancel() == {"FINISHED"}, "spume.cancel ran")
    while baking.poll() is not None:
        time.sleep(0.2)
    check(domain.spume.bake_message == "Cancelled" and not baking.is_running(), "cancel stopped the solver")


def run_bake(scene, domain):
    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)
    check(bpy.ops.spume.bake(preview=True) == {"FINISHED"}, "spume.bake(preview=True) started")
    started = time.time()
    messages = set()
    while baking.is_running():
        interval = baking.poll()
        messages.add(domain.spume.bake_message)
        if interval is None:
            break
        time.sleep(0.4)
        if time.time() - started > 900:
            baking.cancel()
    print(f"bake finished in {time.time() - started:.0f} s: {domain.spume.bake_message}", flush=True)
    check(domain.spume.bake_message.startswith("Baked"), "bake succeeded")
    check(any(message.startswith("Frame ") and message.endswith(" of 40") for message in messages), f"progress read from the frame i/n lines ({sorted(messages)[:3]})")
    return started


def world_vertices(obj):
    count = len(obj.data.vertices)
    local = np.empty(count * 3, np.float32)
    obj.data.vertices.foreach_get("co", local)
    local = local.reshape(count, 3)
    matrix = np.array(obj.matrix_world)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def liquid_at(scene, frame):
    scene.frame_set(frame)
    return world_vertices(bpy.data.objects["Domain Liquid"])


def render_stills(scene):
    engines = [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items]
    scene.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in engines else "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.eevee.taa_render_samples = 48
    for name, value in (("use_raytracing", True), ("use_gtao", True)):
        if hasattr(scene.eevee, name):
            setattr(scene.eevee, name, value)
    scene.view_settings.view_transform = "AgX" if "AgX" in [item.identifier for item in scene.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    scene.view_settings.look = "AgX - Punchy" if scene.view_settings.view_transform == "AgX" else "None"
    scene.view_settings.exposure = -0.3
    paths = []
    for name, frame in zip(("01_impact", "02_spread", "03_fall"), STILL_FRAMES):
        scene.frame_set(frame)
        scene.render.filepath = str(previews_folder / f"spume_{name}.png")
        bpy.ops.render.render(write_still=True)
        paths.append(scene.render.filepath)
    return paths


check(addon_utils.enable("spume_addon", default_set=True) is not None, "addon enabled through addon_utils")
from spume_addon import baking  # noqa: E402  (after the addon is enabled)

scene, domain = build_scene()
check_cancel(domain)
run_bake(scene, domain)
folder = baking.cache_folder(domain, True)

entries = json.loads((folder / "inflows.json").read_text())
check(len(entries) == 2, "both Inflow objects were passed to the solver")
by_mode = {entry["mode"]: entry for entry in entries}
check(set(by_mode) == {"floor", "side"}, "one floor inflow and one side inflow")
check(abs(by_mode["floor"]["position"][0] - (JET_X + 4.0)) < 0.01 and abs(by_mode["floor"]["position"][1] - 4.0) < 0.01, "the jet sits at its real position, not the domain middle")
check(abs(by_mode["side"]["position"][1] - 1.1) < 0.01 and abs(by_mode["side"]["position"][2] - 3.8) < 0.01, "the torrent sits at its real height and offset")
mask = np.load(folder / "obstacle.npy")
check(mask.shape == (100, 125, 100) and 0.9 * (CUBE_SIZE / 0.08) ** 3 < mask.sum() < 1.5 * (CUBE_SIZE / 0.08 + 2) ** 3, f"obstacle mask covers the cube ({int(mask.sum())} solid cells)")

liquid_object = bpy.data.objects["Domain Liquid"]
check(liquid_object.matrix_world.translation.z < 5.1, "liquid object sits in the domain frame")
cube = bpy.data.objects["Obstacle"]
to_cube = np.array(cube.matrix_world.inverted())
half = CUBE_SIZE / 2
summary = {}
for frame in (1, 12, 24, 40):
    vertices = liquid_at(scene, frame)
    local = vertices @ to_cube[:3, :3].T + to_cube[:3, 3]
    inside = int((np.abs(local) < half).all(axis=1).sum())
    column = np.hypot(vertices[:, 0] - JET_X, vertices[:, 1]) < 0.45
    under = int((column & (vertices[:, 2] > CUBE_BOTTOM - 1.0) & (vertices[:, 2] < CUBE_BOTTOM)).sum())
    over = int((column & (vertices[:, 2] > CUBE_TOP)).sum())
    near_wall = int(((vertices[:, 0] < -2.0) & (vertices[:, 2] < 2.5)).sum())
    summary[frame] = dict(vertices=len(vertices), inside_cube=inside, under_cube=under, over_cube_in_column=over, torrent_zone=near_wall)
print("per frame:", summary, flush=True)
check(all(item["inside_cube"] == 0 for item in summary.values()), "no liquid vertex falls inside the cube at frames 1, 12, 24, 40")
check(summary[12]["under_cube"] > 0 or summary[24]["under_cube"] > 0, "the jet reaches the underside of the cube")
check(all(summary[frame]["over_cube_in_column"] == 0 for frame in (12, 24)), "the cube stops the jet: nothing above its top in the jet's column")
check(summary[24]["torrent_zone"] > 0, "the second inflow puts liquid in the domain from the low-X wall")
check(len({item["vertices"] for item in summary.values()}) == 4, "mesh vertex count changes between frames")

meta = json.loads((folder / "meta.json").read_text())
check((folder / "spray_kinds.bin").stat().st_size == sum(count for _, count in meta["spray"]["index"]), "spray_kinds.bin has one byte per whitewater point")
scene.frame_set(24)
kinds = np.zeros(len(bpy.data.objects["Domain Whitewater"].data.vertices), np.int32)
bpy.data.objects["Domain Whitewater"].data.attributes["kind"].data.foreach_get("value", kinds)
print("whitewater points per kind at frame 24 (spray, foam, bubble):", np.bincount(kinds, minlength=3).tolist(), flush=True)
check(len(kinds) > 0 and (np.bincount(kinds, minlength=3) > 0).sum() >= 2, "whitewater has at least two kinds, read from the solver")
log = (folder / "bake.log").read_text()
check("frame 40/40" in log, "the solver logs one `frame i/n` line per frame")
scene.frame_set(1)
check(len(bpy.data.objects["Domain Liquid"].data.vertices) == summary[1]["vertices"], "scrubbing back restores frame 1")
for path in render_stills(scene):
    print("still", path, flush=True)
print("ALL PASSED", flush=True)
