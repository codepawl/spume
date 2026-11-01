# SPDX-License-Identifier: Apache-2.0
"""Particles to a smoothed density field with anisotropic kernels, then GPU marching cubes.

After Yu and Turk 2013: each particle takes its kernel's shape from its neighbours, so it flattens into a disc on a sheet and stretches along a rope,
and its centre moves towards their mean, which smooths the surface. Kernels keep their volume, so the field deep inside the liquid does not change.
Vertex normals come from the field gradient.
"""

import math

import numpy as np
import warp as wp

from .flip import sample_field

# Share of the way a particle's kernel centre moves to the mean of its neighbours.
CENTRE_SMOOTHING = 0.9
# Longest kernel axis over the shortest.
MAX_KERNEL_RATIO = 4.0
# With fewer neighbours than this, a particle is a droplet: a small round kernel.
MIN_NEIGHBOURS = 8
DROPLET_SCALE = 0.75
# Passes of a small blur on the density field: it joins grains on a sheet and rounds droplets.
BLUR_PASSES = 4
# A particle with this many neighbours and axes this close in length counts as interior.
INTERIOR_NEIGHBOURS = 30
INTERIOR_ROUNDNESS = 1.5
# Neighbour search radius in simulation cells. At 8 particles per cell this finds about 45 neighbours.
NEIGHBOUR_RADIUS_CELLS = 1.1
# A vertex inside an obstacle moves out by half a solver cell per attempt.
SOLID_PUSH_ATTEMPTS = 16


@wp.kernel
def fit_kernels(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    neighbour_radius: float,
    centres: wp.array(dtype=wp.vec3),
    shapes: wp.array(dtype=wp.mat33),
    extents: wp.array(dtype=float),
):
    tid = wp.tid()
    if alive[tid] == 0:
        return
    position = positions[tid]
    total_weight = float(0.0)
    first_moment = wp.vec3(0.0, 0.0, 0.0)
    second_moment = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    neighbour_count = int(0)
    query = wp.hash_grid_query(grid, position, neighbour_radius)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if alive[index] == 0:
            continue
        offset = positions[index] - position
        distance = wp.length(offset) / neighbour_radius
        if distance >= 1.0:
            continue
        weight = 1.0 - distance * distance * distance
        total_weight += weight
        first_moment += offset * weight
        second_moment += wp.outer(offset, offset) * weight
        neighbour_count += 1
    if neighbour_count < MIN_NEIGHBOURS:
        centres[tid] = position
        shapes[tid] = wp.identity(n=3, dtype=float) / DROPLET_SCALE
        extents[tid] = DROPLET_SCALE
        return
    mean_offset = first_moment / total_weight
    centres[tid] = position + mean_offset * CENTRE_SMOOTHING
    covariance = second_moment / total_weight - wp.outer(mean_offset, mean_offset)
    axes = wp.mat33()
    variances = wp.vec3()
    unused = wp.mat33()
    wp.svd3(covariance, axes, variances, unused)
    # svd3 sorts the variances from largest to smallest.
    length_0 = wp.sqrt(wp.max(variances[0], 1.0e-12))
    shortest_allowed = length_0 / MAX_KERNEL_RATIO
    length_1 = wp.max(wp.sqrt(wp.max(variances[1], 0.0)), shortest_allowed)
    length_2 = wp.max(wp.sqrt(wp.max(variances[2], 0.0)), shortest_allowed)
    # Deep inside the liquid the shape never reaches the surface, and a round kernel splats to far fewer grid cells.
    if neighbour_count >= INTERIOR_NEIGHBOURS and length_0 < length_2 * INTERIOR_ROUNDNESS:
        shapes[tid] = wp.identity(n=3, dtype=float)
        extents[tid] = 1.0
        return
    volume_scale = wp.pow(length_0 * length_1 * length_2, 1.0 / 3.0)
    inverse_lengths = wp.mat33(volume_scale / length_0, 0.0, 0.0, 0.0, volume_scale / length_1, 0.0, 0.0, 0.0, volume_scale / length_2)
    shapes[tid] = axes * inverse_lengths * wp.transpose(axes)
    extents[tid] = length_0 / volume_scale


@wp.func
def add_kernel(field: wp.array3d(dtype=float), relative: wp.vec3, shape: wp.mat33, weight: float, extent: float, cell: float, radius: float):
    reach = int(radius * extent / cell) + 1
    centre_i = int(wp.floor(relative[0]))
    centre_j = int(wp.floor(relative[1]))
    centre_k = int(wp.floor(relative[2]))
    radius_squared = radius * radius
    for di in range(-reach, reach + 2):
        for dj in range(-reach, reach + 2):
            for dk in range(-reach, reach + 2):
                i = centre_i + di
                j = centre_j + dj
                k = centre_k + dk
                if i < 0 or j < 0 or k < 0 or i >= field.shape[0] or j >= field.shape[1] or k >= field.shape[2]:
                    continue
                offset = shape * ((wp.vec3(float(i), float(j), float(k)) - relative) * cell)
                distance_squared = wp.dot(offset, offset)
                if distance_squared < radius_squared:
                    falloff = 1.0 - distance_squared / radius_squared
                    wp.atomic_add(field, i, j, k, weight * falloff * falloff * falloff)


@wp.kernel
def splat_density(
    centres: wp.array(dtype=wp.vec3),
    shapes: wp.array(dtype=wp.mat33),
    extents: wp.array(dtype=float),
    alive: wp.array(dtype=int),
    origin: wp.vec3,
    cell: float,
    radius: float,
    droplets_only: bool,
    field: wp.array3d(dtype=float),
):
    tid = wp.tid()
    if alive[tid] == 0:
        return
    extent = extents[tid]
    # Only droplets have a kernel smaller than round.
    if (extent < 1.0) != droplets_only:
        return
    relative = (centres[tid] - origin) / cell
    shape = shapes[tid]
    weight = wp.determinant(shape)
    add_kernel(field, relative, shape, weight, extent, cell, radius)
    # The floor cuts off the lower half of the kernel, so a thin sheet on it would fall under the iso level.
    # A mirror image below the floor gives that half back.
    if relative[1] * cell < radius * extent:
        flip_height = wp.mat33(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
        add_kernel(field, wp.vec3(relative[0], -relative[1], relative[2]), flip_height * shape * flip_height, weight, extent, cell, radius)


@wp.kernel
def blur_field_axis(source: wp.array3d(dtype=float), step_i: int, step_j: int, step_k: int, target: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    below_i = wp.max(i - step_i, 0)
    below_j = wp.max(j - step_j, 0)
    below_k = wp.max(k - step_k, 0)
    above_i = wp.min(i + step_i, source.shape[0] - 1)
    above_j = wp.min(j + step_j, source.shape[1] - 1)
    above_k = wp.min(k + step_k, source.shape[2] - 1)
    target[i, j, k] = 0.25 * source[below_i, below_j, below_k] + 0.5 * source[i, j, k] + 0.25 * source[above_i, above_j, above_k]


@wp.kernel
def vertex_normals(
    field: wp.array3d(dtype=float),
    vertices: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    vertex = vertices[tid]
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]
    step = 1.0
    gradient = wp.vec3(
        sample_field(field, x + step, y, z) - sample_field(field, x - step, y, z),
        sample_field(field, x, y + step, z) - sample_field(field, x, y - step, z),
        sample_field(field, x, y, z + step) - sample_field(field, x, y, z - step),
    )
    length = wp.length(gradient)
    if length > 1.0e-9:
        normals[tid] = -gradient / length
    else:
        normals[tid] = wp.vec3(0.0, 1.0, 0.0)


@wp.kernel
def push_out_of_solid(
    vertices: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    solid: wp.array3d(dtype=int),
    origin: wp.vec3,
    mesh_cell: float,
    solver_cell: float,
):
    """Vertices that the particle splats left inside an obstacle move back along the opposite normal, towards the liquid."""
    tid = wp.tid()
    vertex = vertices[tid]
    step = normals[tid] * (0.5 * solver_cell / mesh_cell)
    for _attempt in range(SOLID_PUSH_ATTEMPTS):
        metres = vertex * mesh_cell + origin
        i = wp.clamp(int(wp.floor(metres[0] / solver_cell)), 0, solid.shape[0] - 1)
        j = wp.clamp(int(wp.floor(metres[1] / solver_cell)), 0, solid.shape[1] - 1)
        k = wp.clamp(int(wp.floor(metres[2] / solver_cell)), 0, solid.shape[2] - 1)
        if solid[i, j, k] == 0:
            break
        vertex = vertex - step
    vertices[tid] = vertex


def interior_density(particles_per_cell: float, sim_cell: float, radius: float) -> float:
    """Field value deep inside the liquid: particle density times the integral of the (1 - r^2/R^2)^3 kernel."""
    samples = np.linspace(0.0, radius, 2000)
    integral = np.trapezoid(4.0 * math.pi * samples**2 * (1.0 - (samples / radius) ** 2) ** 3, samples)
    return particles_per_cell / sim_cell**3 * integral


class SurfaceExtractor:
    """Owns the density field, particle kernels and marching-cubes buffers of one mesh grid (reused every frame)."""

    def __init__(self, size, origin, cell, sim_cell, device, radius_cells=2.2):
        self.size = size
        self.origin = origin
        self.cell = cell
        self.device = device
        self.radius = radius_cells * cell
        self.sim_cell = sim_cell
        self.neighbour_radius = NEIGHBOUR_RADIUS_CELLS * sim_cell
        self.field = wp.zeros(size, dtype=float, device=device)
        self.blurred_field = wp.zeros(size, dtype=float, device=device)
        self.marching = wp.MarchingCubes(size[0], size[1], size[2], device=device)
        self.grid = wp.HashGrid(128, 128, 128, device=device)
        self.centres = None
        self.shapes = None
        self.extents = None

    def reserve_kernels(self, particle_count):
        if self.centres is not None and self.centres.shape[0] >= particle_count:
            return
        capacity = int(particle_count * 1.5) + 1024
        self.centres = wp.zeros(capacity, dtype=wp.vec3, device=self.device)
        self.shapes = wp.zeros(capacity, dtype=wp.mat33, device=self.device)
        self.extents = wp.zeros(capacity, dtype=float, device=self.device)

    def blur_field(self):
        for _ in range(BLUR_PASSES):
            for step_i, step_j, step_k in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                wp.launch(blur_field_axis, dim=self.size, inputs=[self.field, step_i, step_j, step_k, self.blurred_field], device=self.device)
                self.field, self.blurred_field = self.blurred_field, self.field

    def extract(self, positions, alive, threshold, solid=None):
        """Returns (vertices in metres, normals, triangle indices) as numpy arrays.

        `solid` is the solver's int array of obstacle cells; vertices inside it are moved out on the device.
        """
        particle_count = positions.shape[0]
        self.reserve_kernels(particle_count)
        self.grid.build(positions, self.neighbour_radius)
        wp.launch(fit_kernels, dim=particle_count, inputs=[self.grid.id, positions, alive, self.neighbour_radius, self.centres, self.shapes, self.extents], device=self.device)
        self.field.zero_()
        splat_inputs = [self.centres, self.shapes, self.extents, alive, wp.vec3(*self.origin), self.cell, self.radius]
        wp.launch(splat_density, dim=particle_count, inputs=splat_inputs + [False, self.field], device=self.device)
        self.blur_field()
        # Droplets go in after the blur, which would melt them away.
        wp.launch(splat_density, dim=particle_count, inputs=splat_inputs + [True, self.field], device=self.device)
        self.marching.surface(self.field, threshold)
        vertex_count = self.marching.verts.shape[0]
        if vertex_count == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0, np.uint32)
        normals = wp.zeros(vertex_count, dtype=wp.vec3, device=self.device)
        wp.launch(vertex_normals, dim=vertex_count, inputs=[self.field, self.marching.verts, normals], device=self.device)
        if solid is not None:
            wp.launch(push_out_of_solid, dim=vertex_count, inputs=[self.marching.verts, normals, solid, wp.vec3(*self.origin), self.cell, self.sim_cell], device=self.device)
        vertices = self.marching.verts.numpy() * self.cell + np.asarray(self.origin, np.float32)
        # Warp's marching cubes winds its triangles inward. Viewers that cull back faces need them outward.
        outward_indices = self.marching.indices.numpy().astype(np.uint32).reshape(-1, 3)[:, [0, 2, 1]].ravel()
        return vertices.astype(np.float32), normals.numpy().astype(np.float32), outward_indices
