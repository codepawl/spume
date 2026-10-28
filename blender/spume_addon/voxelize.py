# SPDX-License-Identifier: Apache-2.0
"""Turns Obstacle and Outflow objects into solver data: a solid-cell mask on the solver grid and boxes in solver metres."""

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# Sample points sit at multiples of half a cell, so every cell owns its corners, face centres and centre (27 points).
SAMPLES_PER_CELL = 2


def world_bvh(obj, depsgraph):
    """BVH of the object's evaluated triangles in world space."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
        triangles = [tuple(triangle.vertices) for triangle in mesh.loop_triangles]
    finally:
        evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(vertices, triangles)


def solver_bounds(obj, to_solver):
    """Axis-aligned bounds of the object in solver metres as (low, high) numpy vectors."""
    corners = np.array([(to_solver @ (obj.matrix_world @ Vector(corner)))[:] for corner in obj.bound_box])
    return corners.min(axis=0), corners.max(axis=0)


def inside_samples(tree, to_world, step, first_index, counts):
    """Inside test on a regular lattice of solver points: point (a, b, c) is at (first_index + (a, b, c)) * step.
    One ray per lattice column along solver x; a point is inside when an odd number of surface hits lies before it. Needs a closed mesh."""
    along = np.array([(first_index[0] + index) * step for index in range(counts[0])])
    start_x = along[0] - step
    direction = (to_world.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
    inside = np.zeros(counts, bool)
    for b in range(counts[1]):
        for c in range(counts[2]):
            # a tiny irregular offset keeps rays off the shared edges of triangles
            y = (first_index[1] + b) * step + 1.3e-4
            z = (first_index[2] + c) * step + 2.1e-4
            origin = to_world @ Vector((start_x, y, z))
            travelled = 0.0
            hits = []
            for _ in range(256):
                location, _normal, _index, distance = tree.ray_cast(origin, direction)
                if location is None:
                    break
                travelled += distance
                hits.append(start_x + travelled)
                origin = location + direction * 1.0e-5
                travelled += 1.0e-5
            if hits:
                inside[:, b, c] = np.searchsorted(np.array(hits), along) % 2 == 1
    return inside


def obstacle_mask(objects, depsgraph, to_solver, size, cell):
    """Boolean solid mask (nx, ny, nz) on the solver grid. A cell is solid when any of its sample points is inside an obstacle, so the voxels
    cover the mesh (they may stick out up to a cell). Returns None when no object touches the grid."""
    mask = np.zeros(size, bool)
    to_world = to_solver.inverted()
    step = cell / SAMPLES_PER_CELL
    for obj in objects:
        if obj.type != "MESH":
            continue
        low, high = solver_bounds(obj, to_solver)
        first_cell = np.maximum(np.floor(low / cell).astype(int) - 1, 0)
        last_cell = np.minimum(np.ceil(high / cell).astype(int) + 1, np.array(size) - 1)
        if np.any(last_cell < first_cell):
            continue
        cell_counts = last_cell - first_cell + 1
        lattice_first = first_cell * SAMPLES_PER_CELL
        lattice_counts = cell_counts * SAMPLES_PER_CELL + 1
        samples = inside_samples(world_bvh(obj, depsgraph), to_world, step, lattice_first, lattice_counts)
        for axis in range(3):
            samples = np.moveaxis(samples, axis, 0)
            samples = samples[0:-2:2] | samples[1:-1:2] | samples[2::2]
            samples = np.moveaxis(samples, 0, axis)
        mask[first_cell[0] : last_cell[0] + 1, first_cell[1] : last_cell[1] + 1, first_cell[2] : last_cell[2] + 1] |= samples
    return mask if mask.any() else None


def outflow_boxes(objects, to_solver):
    """Each Outflow object as (x0, y0, z0, x1, y1, z1) in solver metres: the bounds of the object."""
    boxes = []
    for obj in objects:
        low, high = solver_bounds(obj, to_solver)
        boxes.append((*low.tolist(), *high.tolist()))
    return boxes
