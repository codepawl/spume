# SPDX-License-Identifier: Apache-2.0
"""Particles to a smoothed density field with velocity-stretched kernels, then GPU marching cubes.

A particle moving fast is splatted as an ellipsoid elongated along its velocity (a cheap stand-in for the anisotropic kernels of
Yu and Turk 2013), so a fast rope stays a rope instead of a row of beads. Vertex normals come from the field gradient.
"""

import math

import numpy as np
import warp as wp

from flip import sample_field


@wp.kernel
def splat_density(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    origin: wp.vec3,
    cell: float,
    radius: float,
    stretch_per_speed: float,
    max_stretch: float,
    field: wp.array3d(dtype=float),
):
    tid = wp.tid()
    if alive[tid] == 0:
        return
    relative = (positions[tid] - origin) / cell
    speed = wp.length(velocities[tid])
    direction = wp.vec3(0.0, 1.0, 0.0)
    stretch = float(1.0)
    if speed > 0.5:
        direction = velocities[tid] / speed
        stretch = wp.min(1.0 + speed * stretch_per_speed, max_stretch)
    # the kernel reaches `reach` cells along the stretched axis
    reach = int(radius * stretch / cell) + 1
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
                offset = (wp.vec3(float(i), float(j), float(k)) - relative) * cell
                along = wp.dot(offset, direction)
                across_squared = wp.dot(offset, offset) - along * along
                distance_squared = across_squared + (along / stretch) * (along / stretch)
                if distance_squared < radius_squared:
                    falloff = 1.0 - distance_squared / radius_squared
                    wp.atomic_add(field, i, j, k, falloff * falloff * falloff)


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


def interior_density(particles_per_cell: float, sim_cell: float, radius: float) -> float:
    """Field value deep inside the liquid: particle density times the integral of the (1 - r^2/R^2)^3 kernel."""
    samples = np.linspace(0.0, radius, 2000)
    integral = np.trapezoid(4.0 * math.pi * samples**2 * (1.0 - (samples / radius) ** 2) ** 3, samples)
    return particles_per_cell / sim_cell**3 * integral


class SurfaceExtractor:
    """Owns the density field and marching-cubes buffers of one mesh grid (reused every frame)."""

    def __init__(self, size, origin, cell, device, radius_cells=2.2, stretch_per_speed=0.12, max_stretch=2.5):
        self.size = size
        self.origin = origin
        self.cell = cell
        self.device = device
        self.radius = radius_cells * cell
        self.stretch_per_speed = stretch_per_speed
        self.max_stretch = max_stretch
        self.field = wp.zeros(size, dtype=float, device=device)
        self.marching = wp.MarchingCubes(size[0], size[1], size[2], device=device)

    def extract(self, positions, velocities, alive, threshold):
        """Returns (vertices in metres, normals, triangle indices) as numpy arrays."""
        self.field.zero_()
        wp.launch(
            splat_density,
            dim=positions.shape[0],
            inputs=[positions, velocities, alive, wp.vec3(*self.origin), self.cell, self.radius, self.stretch_per_speed, self.max_stretch, self.field],
            device=self.device,
        )
        self.marching.surface(self.field, threshold)
        vertex_count = self.marching.verts.shape[0]
        if vertex_count == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0, np.uint32)
        normals = wp.zeros(vertex_count, dtype=wp.vec3, device=self.device)
        wp.launch(vertex_normals, dim=vertex_count, inputs=[self.field, self.marching.verts, normals], device=self.device)
        vertices = self.marching.verts.numpy() * self.cell + np.asarray(self.origin, np.float32)
        return vertices.astype(np.float32), normals.numpy().astype(np.float32), self.marching.indices.numpy().astype(np.uint32)
