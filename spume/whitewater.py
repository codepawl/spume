# SPDX-License-Identifier: Apache-2.0
"""Diffuse whitewater (spray, foam, bubbles) after Ihmsen, Akinci, Becker and Teschner, "Unified spray, foam and air bubbles
for particle-based fluids" (2012). Potentials are evaluated per liquid particle from grid fields:

  trapped air  - how strongly the velocity field varies around the particle (relative velocity of neighbours)
  wave crest   - convex surface (curvature of the density normal) that is moving outwards
  energy       - kinetic energy 0.5 |v|^2, which scales both

Emitted count per step = energy * (k_trapped * trapped + k_crest * crest) * dt. Each diffuse particle is then classified from
the local liquid density: spray (little liquid: ballistic), foam (surface: follows the grid velocity), bubble (inside: buoyant).
"""

import warp as wp

from .flip import GRAVITY, clamp_index, is_wall, resolve_solid, sample_field
from .sparse import cell_of_thread, outside

SPRAY, FOAM, BUBBLE = 0, 1, 2

# A diffuse particle that reaches a solid side wall or the ceiling keeps this share of its speed (bounced back) and slides on.
WALL_RESTITUTION = 0.3
WALL_MARGIN = 0.01


@wp.kernel
def cell_fields(
    blocks: wp.array(dtype=wp.vec3i),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    cell_count: wp.array3d(dtype=int),
    inverse_cell: float,
    particles_per_cell: float,
    density: wp.array3d(dtype=float),
    shear: wp.array3d(dtype=float),
):
    """Smoothed liquid density (1 = full) and velocity-gradient norm (1/s) per cell."""
    cell_index = cell_of_thread(blocks, wp.tid())
    if outside(density, cell_index):
        return
    i = cell_index[0]
    j = cell_index[1]
    k = cell_index[2]
    size_x = density.shape[0]
    size_y = density.shape[1]
    size_z = density.shape[2]
    total = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                total = total + float(cell_count[clamp_index(i + di, size_x), clamp_index(j + dj, size_y), clamp_index(k + dk, size_z)])
    density[i, j, k] = total / (27.0 * particles_per_cell)
    centre_velocity = wp.vec3(0.5 * (u[i, j, k] + u[i + 1, j, k]), 0.5 * (v[i, j, k] + v[i, j + 1, k]), 0.5 * (w[i, j, k] + w[i, j, k + 1]))
    low_velocity = wp.vec3(0.5 * (u[i, j, k] + u[wp.max(i - 1, 0), j, k]), 0.5 * (v[i, j, k] + v[i, wp.max(j - 1, 0), k]), 0.5 * (w[i, j, k] + w[i, j, wp.max(k - 1, 0)]))
    high_velocity = wp.vec3(0.5 * (u[i + 1, j, k] + u[wp.min(i + 2, size_x), j, k]), 0.5 * (v[i, j + 1, k] + v[i, wp.min(j + 2, size_y), k]), 0.5 * (w[i, j, k + 1] + w[i, j, wp.min(k + 2, size_z)]))
    gradient = (high_velocity - low_velocity) * (0.5 * inverse_cell)
    shear[i, j, k] = wp.length(gradient) + 0.0 * wp.length(centre_velocity)


@wp.kernel
def density_normals(
    blocks: wp.array(dtype=wp.vec3i),
    density: wp.array3d(dtype=float),
    normal_x: wp.array3d(dtype=float),
    normal_y: wp.array3d(dtype=float),
    normal_z: wp.array3d(dtype=float),
):
    """Outward surface normal = -gradient of the smoothed density."""
    cell_index = cell_of_thread(blocks, wp.tid())
    if outside(density, cell_index):
        return
    i = cell_index[0]
    j = cell_index[1]
    k = cell_index[2]
    size_x = density.shape[0]
    size_y = density.shape[1]
    size_z = density.shape[2]
    gx = density[clamp_index(i + 1, size_x), j, k] - density[clamp_index(i - 1, size_x), j, k]
    gy = density[i, clamp_index(j + 1, size_y), k] - density[i, clamp_index(j - 1, size_y), k]
    gz = density[i, j, clamp_index(k + 1, size_z)] - density[i, j, clamp_index(k - 1, size_z)]
    length = wp.sqrt(gx * gx + gy * gy + gz * gz)
    if length > 1.0e-5:
        normal_x[i, j, k] = -gx / length
        normal_y[i, j, k] = -gy / length
        normal_z[i, j, k] = -gz / length
    else:
        normal_x[i, j, k] = 0.0
        normal_y[i, j, k] = 0.0
        normal_z[i, j, k] = 0.0


@wp.func
def ramp(value: float, low: float, high: float) -> float:
    return (wp.min(value, high) - wp.min(value, low)) / (high - low)


@wp.kernel
def emit_whitewater(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    density: wp.array3d(dtype=float),
    shear: wp.array3d(dtype=float),
    normal_x: wp.array3d(dtype=float),
    normal_y: wp.array3d(dtype=float),
    normal_z: wp.array3d(dtype=float),
    inverse_cell: float,
    cell: float,
    trapped_rate: float,
    crest_rate: float,
    trapped_limits: wp.vec2,
    crest_limits: wp.vec2,
    energy_limits: wp.vec2,
    life_range: wp.vec2,
    delta_time: float,
    step_seed: int,
    seed: int,
    ww_positions: wp.array(dtype=wp.vec3),
    ww_velocities: wp.array(dtype=wp.vec3),
    ww_life: wp.array(dtype=float),
    ww_ids: wp.array(dtype=int),
    ww_counter: wp.array(dtype=int),
):
    tid = wp.tid()
    if alive[tid] == 0:
        return
    position = positions[tid]
    velocity = velocities[tid]
    g = position * inverse_cell
    i = clamp_index(int(g[0]), density.shape[0])
    j = clamp_index(int(g[1]), density.shape[1])
    k = clamp_index(int(g[2]), density.shape[2])
    speed = wp.length(velocity)
    energy = ramp(0.5 * speed * speed, energy_limits[0], energy_limits[1])
    if energy <= 0.0:
        return
    trapped = ramp(shear[i, j, k] * cell, trapped_limits[0], trapped_limits[1])
    crest = float(0.0)
    if density[i, j, k] < 0.8 and speed > 0.1:
        normal = wp.vec3(normal_x[i, j, k], normal_y[i, j, k], normal_z[i, j, k])
        if wp.dot(normal, velocity / speed) > 0.6:
            curvature = (
                normal_x[clamp_index(i + 1, density.shape[0]), j, k]
                - normal_x[clamp_index(i - 1, density.shape[0]), j, k]
                + normal_y[i, clamp_index(j + 1, density.shape[1]), k]
                - normal_y[i, clamp_index(j - 1, density.shape[1]), k]
                + normal_z[i, j, clamp_index(k + 1, density.shape[2])]
                - normal_z[i, j, clamp_index(k - 1, density.shape[2])]
            ) * (0.5 * inverse_cell)
            crest = ramp(curvature, crest_limits[0], crest_limits[1])
    expected = energy * (trapped_rate * trapped + crest_rate * crest) * delta_time
    state = wp.rand_init(seed + tid * 2654435 + step_seed * 40503, step_seed)
    count = int(expected)
    if wp.randf(state) < expected - float(count):
        count = count + 1
    for emitted in range(count):
        # a ring buffer: when full, the oldest slots are overwritten
        slot = wp.atomic_add(ww_counter, 0, 1) % ww_positions.shape[0]
        angle = wp.randf(state) * 6.2831853
        radius = cell * 0.5 * wp.sqrt(wp.randf(state))
        offset = wp.vec3(radius * wp.cos(angle), (wp.randf(state) - 0.5) * cell, radius * wp.sin(angle))
        jitter = wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5) * (0.5 * speed)
        ww_positions[slot] = position + offset
        ww_velocities[slot] = velocity + jitter
        ww_life[slot] = life_range[0] + (life_range[1] - life_range[0]) * wp.randf(state)
        ww_ids[slot] = tid * 31 + emitted * 7919 + step_seed * 104729


@wp.kernel
def advect_whitewater(
    ww_positions: wp.array(dtype=wp.vec3),
    ww_velocities: wp.array(dtype=wp.vec3),
    ww_life: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    inverse_cell: float,
    delta_time: float,
    spray_density: float,
    foam_density: float,
    buoyancy: float,
    drag: float,
    domain_size: wp.vec3,
    walls: int,
):
    tid = wp.tid()
    if ww_life[tid] <= 0.0:
        return
    position = ww_positions[tid]
    velocity = ww_velocities[tid]
    g = position * inverse_cell
    local_density = sample_field(density, g[0] - 0.5, g[1] - 0.5, g[2] - 0.5)
    grid_velocity = wp.vec3(
        sample_field(u, g[0], g[1] - 0.5, g[2] - 0.5),
        sample_field(v, g[0] - 0.5, g[1], g[2] - 0.5),
        sample_field(w, g[0] - 0.5, g[1] - 0.5, g[2]),
    )
    if local_density < spray_density:
        velocity = velocity + wp.vec3(0.0, -GRAVITY, 0.0) * delta_time
        velocity = velocity * (1.0 - 0.15 * delta_time)
        ww_life[tid] = ww_life[tid] - 0.35 * delta_time
    elif local_density < foam_density:
        velocity = grid_velocity
        ww_life[tid] = ww_life[tid] - delta_time
    else:
        velocity = velocity + (wp.vec3(0.0, buoyancy * GRAVITY, 0.0) + (grid_velocity - velocity) * drag) * delta_time
        ww_life[tid] = ww_life[tid] - 0.15 * delta_time
    target = position + velocity * delta_time
    position = resolve_solid(solid, position, target, inverse_cell)
    for axis in range(3):
        if position[axis] != target[axis]:
            velocity[axis] = 0.0
    if position[1] < 0.01:
        position = wp.vec3(position[0], 0.01, position[2])
        velocity = wp.vec3(velocity[0] * 0.5, 0.0, velocity[2] * 0.5)
    # solid side walls (x, z) and the ceiling bounce the particle back; the floor is handled above and open faces let it go
    for axis in range(3):
        if axis != 1 and position[axis] < WALL_MARGIN and is_wall(walls, axis * 2):
            position[axis] = WALL_MARGIN
            velocity[axis] = wp.abs(velocity[axis]) * WALL_RESTITUTION
        if position[axis] > domain_size[axis] - WALL_MARGIN and is_wall(walls, axis * 2 + 1):
            position[axis] = domain_size[axis] - WALL_MARGIN
            velocity[axis] = -wp.abs(velocity[axis]) * WALL_RESTITUTION
    if position[0] < 0.0 or position[2] < 0.0 or position[0] >= domain_size[0] or position[2] >= domain_size[2] or position[1] >= domain_size[1]:
        ww_life[tid] = 0.0
    ww_positions[tid] = position
    ww_velocities[tid] = velocity


@wp.kernel
def select_for_export(
    ww_positions: wp.array(dtype=wp.vec3),
    ww_velocities: wp.array(dtype=wp.vec3),
    ww_life: wp.array(dtype=float),
    ww_ids: wp.array(dtype=int),
    density: wp.array3d(dtype=float),
    inverse_cell: float,
    spray_density: float,
    foam_density: float,
    keep_fraction: float,
    out_positions: wp.array(dtype=wp.vec3),
    out_velocities: wp.array(dtype=wp.vec3),
    out_ids: wp.array(dtype=int),
    out_kinds: wp.array(dtype=wp.uint8),
    out_counter: wp.array(dtype=int),
):
    """Deterministic subsample: keeps a particle by a hash of its own id, so the chosen set does not depend on GPU thread order.
    The kind (spray, foam, bubble) is the one `advect_whitewater` just moved it by, from the local liquid density."""
    tid = wp.tid()
    if ww_life[tid] <= 0.0:
        return
    state = wp.rand_init(ww_ids[tid], 12345)
    if wp.randf(state) > keep_fraction:
        return
    slot = wp.atomic_add(out_counter, 0, 1)
    if slot >= out_positions.shape[0]:
        return
    out_positions[slot] = ww_positions[tid]
    out_velocities[slot] = ww_velocities[tid]
    out_ids[slot] = ww_ids[tid]
    g = ww_positions[tid] * inverse_cell
    local_density = sample_field(density, g[0] - 0.5, g[1] - 0.5, g[2] - 0.5)
    kind = int(BUBBLE)
    if local_density < spray_density:
        kind = int(SPRAY)
    elif local_density < foam_density:
        kind = int(FOAM)
    out_kinds[slot] = wp.uint8(kind)
