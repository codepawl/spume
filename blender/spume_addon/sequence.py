# SPDX-License-Identifier: Apache-2.0
"""Mesh sequence objects: a liquid mesh and a whitewater point cloud that follow the scene frame through a handler."""

import bpy
import numpy as np
from bpy.app.handlers import persistent
from mathutils import Matrix, Vector

from . import cache, look

# Solver axes are x, y (up), z. Blender's are x, y, z (up): solver (x, y, z) is Blender (x, -z, y), a quarter turn about x.
SOLVER_TO_BLENDER = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))

SPRAY, FOAM, BUBBLE = 0, 1, 2
KIND_RADIUS = {SPRAY: 0.016, FOAM: 0.026, BUBBLE: 0.011}

_shown = {}


def domain_layout(domain):
    """Box size in solver axes (length x, height, width z) and the matrix taking solver metres to world.

    The solver floor is the domain's lowest face, solver z = 0 is its far (+Y) face, and the box is not scaled by the matrix because the
    scale is already folded into the size. Rotation is kept; shear is not supported.
    """
    corners = np.array([corner[:] for corner in domain.bound_box])
    low, high = corners.min(axis=0), corners.max(axis=0)
    scale = domain.matrix_world.to_scale()
    size = (high - low) * np.array(scale)
    origin = domain.matrix_world @ Vector((low[0], high[1], low[2]))
    rotation = domain.matrix_world.to_quaternion().to_matrix().to_4x4()
    return (float(size[0]), float(size[2]), float(size[1])), Matrix.Translation(origin) @ rotation @ SOLVER_TO_BLENDER


def opening_offset(meta):
    """The solver writes positions relative to the first opening's centre (x and z); this restores the box frame.
    Version 0.2 caches store that point as `origin`; older ones are reconstructed from the single-inflow settings."""
    if meta.get("origin"):
        return np.array(meta["origin"], np.float32)
    settings = meta["settings"]
    length = settings["length"] or settings["width"]
    if settings["inflow"] == "side":
        return np.array([0.0, 0.0, settings["width"] * 0.5], np.float32)
    return np.array([length * 0.5, 0.0, settings["width"] * 0.5], np.float32)


def classify(velocities):
    """Guess the kind of a point from its speed, for caches that have no spray_kinds.bin: fast is spray, rising and slow is a bubble, slow is foam."""
    speed = np.linalg.norm(velocities, axis=1)
    kind = np.full(len(speed), SPRAY, np.int32)
    kind[speed < 2.5] = FOAM
    kind[(speed < 3.0) & (velocities[:, 1] > 0.8)] = BUBBLE
    return kind


def set_mesh(mesh, vertices, triangles):
    mesh.clear_geometry()
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.ravel())
    mesh.loops.add(len(triangles) * 3)
    mesh.loops.foreach_set("vertex_index", triangles.ravel())
    mesh.polygons.add(len(triangles))
    mesh.polygons.foreach_set("loop_start", np.arange(0, len(triangles) * 3, 3, dtype=np.int32))
    mesh.update(calc_edges=True)
    mesh.shade_smooth()


def set_points(mesh, positions, velocities, ids, kinds=None):
    mesh.clear_geometry()
    count = len(positions)
    mesh.vertices.add(count)
    mesh.vertices.foreach_set("co", positions.ravel())
    kind = classify(velocities) if kinds is None else kinds.astype(np.int32)
    jitter = ((ids.astype(np.uint64) * 2654435761) % 4294967296).astype(np.float32) / 4294967296.0
    radius = np.array([KIND_RADIUS[SPRAY], KIND_RADIUS[FOAM], KIND_RADIUS[BUBBLE]], np.float32)[kind] * (0.6 + 0.8 * jitter)
    for name, data_type, values in (("radius", "FLOAT", radius), ("kind", "INT", kind), ("velocity", "FLOAT_VECTOR", velocities.astype(np.float32).ravel())):
        mesh.attributes.new(name, data_type, "POINT").data.foreach_set("value" if data_type != "FLOAT_VECTOR" else "vector", values)
    mesh.update()


def sequence_object(domain, suffix, material=None):
    name = f"{domain.name} {suffix}"
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        (domain.users_collection[0] if domain.users_collection else bpy.context.scene.collection).objects.link(obj)
        if material:
            obj.data.materials.append(material)
    return obj


def frame_index(domain, frame_count, scene_frame):
    return int(min(max(scene_frame - domain.spume.start_frame, 0), frame_count - 1))


def show_frame(domain, scene_frame, force=False):
    settings = domain.spume
    folder = bpy.path.abspath(settings.loaded_folder)
    liquid = bpy.data.objects.get(f"{domain.name} Liquid")
    if not folder or liquid is None:
        return
    meta = cache.read_meta(folder)
    index = frame_index(domain, meta["frameCount"], scene_frame)
    if not force and _shown.get(domain.name) == (folder, index):
        return
    offset = opening_offset(meta)
    vertices, triangles = cache.read_mesh_frame(folder, index)
    set_mesh(liquid.data, vertices + offset, triangles)
    whitewater = bpy.data.objects.get(f"{domain.name} Whitewater")
    if whitewater is not None:
        positions, velocities, ids, kinds = cache.read_spray_frame(folder, meta, index)
        set_points(whitewater.data, positions + offset, velocities, ids, kinds)
    _shown[domain.name] = (folder, index)


def load(domain, folder):
    """Create or refresh the liquid and whitewater objects for a finished cache and show the current frame."""
    settings = domain.spume
    meta = cache.read_meta(folder)
    settings.loaded_folder = folder
    settings.frames = meta["frameCount"]
    _, matrix = domain_layout(domain)
    liquid = sequence_object(domain, "Liquid", look.water_material())
    liquid.matrix_world = matrix
    if meta["settings"]["ww_export"] > 1 and not meta["settings"].get("no_whitewater"):
        whitewater = sequence_object(domain, "Whitewater")
        whitewater.matrix_world = matrix
        if "Spume Spheres" not in whitewater.modifiers:
            modifier = whitewater.modifiers.new("Spume Spheres", "NODES")
            modifier.node_group = look.whitewater_nodes()
    show_frame(domain, bpy.context.scene.frame_current, force=True)


def free_objects(domain):
    for suffix in ("Liquid", "Whitewater"):
        obj = bpy.data.objects.get(f"{domain.name} {suffix}")
        if obj is not None:
            mesh = obj.data
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)
    domain.spume.loaded_folder = ""
    _shown.pop(domain.name, None)


@persistent
def on_frame_change(scene, depsgraph=None):
    for domain in scene.objects:
        if domain.spume.role == "DOMAIN" and domain.spume.loaded_folder:
            show_frame(domain, scene.frame_current)


@persistent
def on_file_load(_unused=None):
    _shown.clear()


def register():
    bpy.app.handlers.frame_change_pre.append(on_frame_change)
    bpy.app.handlers.load_post.append(on_file_load)


def unregister():
    bpy.app.handlers.frame_change_pre.remove(on_frame_change)
    bpy.app.handlers.load_post.remove(on_file_load)
