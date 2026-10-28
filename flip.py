# SPDX-License-Identifier: Apache-2.0
"""GPU FLIP/PIC liquid solver on a staggered (MAC) grid, written with NVIDIA Warp.

Coordinates are real metres, +Y up. The grid origin is the corner of the domain; the floor is the plane y = 0.
Boundaries: the floor is a solid wall, every other face of the box is open air (pressure 0, particles that leave are removed).
Obstacles are a boolean mask of solid cells: no flow through their faces, no pressure inside, particles slide along them.
Particles carry velocity. Each step: particles to grid, gravity, inflow, pressure projection, grid to particles, advect.
"""

import warp as wp

GRAVITY = 9.81


@wp.func
def clamp_index(value: int, size: int) -> int:
    return wp.max(0, wp.min(size - 1, value))


@wp.func
def sample_field(field: wp.array3d(dtype=float), x: float, y: float, z: float) -> float:
    """Trilinear sample; x, y, z are in the field's own index space (node = integer)."""
    size_x = field.shape[0]
    size_y = field.shape[1]
    size_z = field.shape[2]
    x = wp.clamp(x, 0.0, float(size_x - 1) - 1.0e-4)
    y = wp.clamp(y, 0.0, float(size_y - 1) - 1.0e-4)
    z = wp.clamp(z, 0.0, float(size_z - 1) - 1.0e-4)
    i = int(x)
    j = int(y)
    k = int(z)
    fx = x - float(i)
    fy = y - float(j)
    fz = z - float(k)
    c00 = field[i, j, k] * (1.0 - fx) + field[i + 1, j, k] * fx
    c10 = field[i, j + 1, k] * (1.0 - fx) + field[i + 1, j + 1, k] * fx
    c01 = field[i, j, k + 1] * (1.0 - fx) + field[i + 1, j, k + 1] * fx
    c11 = field[i, j + 1, k + 1] * (1.0 - fx) + field[i + 1, j + 1, k + 1] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def splat_field(field: wp.array3d(dtype=float), weights: wp.array3d(dtype=float), x: float, y: float, z: float, value: float):
    size_x = field.shape[0]
    size_y = field.shape[1]
    size_z = field.shape[2]
    x = wp.clamp(x, 0.0, float(size_x - 1) - 1.0e-4)
    y = wp.clamp(y, 0.0, float(size_y - 1) - 1.0e-4)
    z = wp.clamp(z, 0.0, float(size_z - 1) - 1.0e-4)
    i = int(x)
    j = int(y)
    k = int(z)
    fx = x - float(i)
    fy = y - float(j)
    fz = z - float(k)
    for corner in range(8):
        di = corner & 1
        dj = (corner >> 1) & 1
        dk = (corner >> 2) & 1
        weight = wp.where(di == 1, fx, 1.0 - fx) * wp.where(dj == 1, fy, 1.0 - fy) * wp.where(dk == 1, fz, 1.0 - fz)
        wp.atomic_add(field, i + di, j + dj, k + dk, weight * value)
        wp.atomic_add(weights, i + di, j + dj, k + dk, weight)


@wp.kernel
def particles_to_grid(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    inverse_cell: float,
    u: wp.array3d(dtype=float),
    u_weight: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    v_weight: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    w_weight: wp.array3d(dtype=float),
    cell_count: wp.array3d(dtype=int),
):
    tid = wp.tid()
    if alive[tid] == 0:
        return
    grid_position = positions[tid] * inverse_cell
    velocity = velocities[tid]
    splat_field(u, u_weight, grid_position[0], grid_position[1] - 0.5, grid_position[2] - 0.5, velocity[0])
    splat_field(v, v_weight, grid_position[0] - 0.5, grid_position[1], grid_position[2] - 0.5, velocity[1])
    splat_field(w, w_weight, grid_position[0] - 0.5, grid_position[1] - 0.5, grid_position[2], velocity[2])
    i = int(grid_position[0])
    j = int(grid_position[1])
    k = int(grid_position[2])
    wp.atomic_add(cell_count, clamp_index(i, cell_count.shape[0]), clamp_index(j, cell_count.shape[1]), clamp_index(k, cell_count.shape[2]), 1)


@wp.kernel
def normalise_face(field: wp.array3d(dtype=float), weight: wp.array3d(dtype=float), old: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    if weight[i, j, k] > 1.0e-6:
        field[i, j, k] = field[i, j, k] / weight[i, j, k]
    else:
        field[i, j, k] = 0.0
    old[i, j, k] = field[i, j, k]


@wp.kernel
def add_gravity(v: wp.array3d(dtype=float), delta_time: float):
    i, j, k = wp.tid()
    v[i, j, k] = v[i, j, k] - GRAVITY * delta_time


@wp.kernel
def extend_face(
    source: wp.array3d(dtype=float),
    source_valid: wp.array3d(dtype=float),
    target: wp.array3d(dtype=float),
    target_valid: wp.array3d(dtype=float),
):
    """One layer of velocity extrapolation into faces no particle touched."""
    i, j, k = wp.tid()
    if source_valid[i, j, k] > 0.0:
        target[i, j, k] = source[i, j, k]
        target_valid[i, j, k] = 1.0
        return
    total = float(0.0)
    count = float(0.0)
    size_x = source.shape[0]
    size_y = source.shape[1]
    size_z = source.shape[2]
    for axis in range(6):
        ni = i
        nj = j
        nk = k
        if axis == 0:
            ni = i - 1
        elif axis == 1:
            ni = i + 1
        elif axis == 2:
            nj = j - 1
        elif axis == 3:
            nj = j + 1
        elif axis == 4:
            nk = k - 1
        else:
            nk = k + 1
        if ni >= 0 and ni < size_x and nj >= 0 and nj < size_y and nk >= 0 and nk < size_z:
            if source_valid[ni, nj, nk] > 0.0:
                total = total + source[ni, nj, nk]
                count = count + 1.0
    if count > 0.0:
        target[i, j, k] = total / count
        target_valid[i, j, k] = 1.0
    else:
        target[i, j, k] = 0.0
        target_valid[i, j, k] = 0.0


SHAPE_CIRCLE = 0
SHAPE_RECTANGLE = 1

# Solid walls are a bit mask: bit 0 is the x = 0 face, 1 the far x face, 2 the floor, 3 the ceiling, 4 the z = 0 face, 5 the far z face.
# Faces that are not set are open air. A solid wall carries no pressure and no velocity through it, except where an opening imposes one.
WALL_FLOOR = 4
WALL_ALL = 63


@wp.func
def is_wall(walls: int, bit: int) -> bool:
    return ((walls >> bit) & 1) == 1


@wp.func
def solid_cell(solid: wp.array3d(dtype=int), position: wp.vec3, inverse_cell: float) -> bool:
    """True when the point lies inside a solid obstacle cell (points outside the grid are never solid)."""
    i = int(wp.floor(position[0] * inverse_cell))
    j = int(wp.floor(position[1] * inverse_cell))
    k = int(wp.floor(position[2] * inverse_cell))
    if i < 0 or j < 0 or k < 0 or i >= solid.shape[0] or j >= solid.shape[1] or k >= solid.shape[2]:
        return False
    return solid[i, j, k] == 1


@wp.func
def resolve_solid(solid: wp.array3d(dtype=int), old_position: wp.vec3, new_position: wp.vec3, inverse_cell: float) -> wp.vec3:
    """Moves from the old to the new position one axis at a time and stops an axis that would enter a solid cell, so a particle slides along a face.
    A point that already sits inside a solid (an emission on the surface) is left free to leave."""
    if not solid_cell(solid, new_position, inverse_cell):
        return new_position
    if solid_cell(solid, old_position, inverse_cell):
        return new_position
    moved = wp.vec3(old_position[0], old_position[1], old_position[2])
    for axis in range(3):
        trial = wp.vec3(moved[0], moved[1], moved[2])
        trial[axis] = new_position[axis]
        if not solid_cell(solid, trial, inverse_cell):
            moved = trial
    return moved


@wp.kernel
def enforce_solids(field: wp.array3d(dtype=float), axis: int, solid: wp.array3d(dtype=int)):
    """Zero the velocity of every face that touches a solid cell."""
    i, j, k = wp.tid()
    low = int(0)
    high = int(0)
    if axis == 0:
        if i > 0:
            low = solid[i - 1, j, k]
        if i < solid.shape[0]:
            high = solid[i, j, k]
    elif axis == 1:
        if j > 0:
            low = solid[i, j - 1, k]
        if j < solid.shape[1]:
            high = solid[i, j, k]
    else:
        if k > 0:
            low = solid[i, j, k - 1]
        if k < solid.shape[2]:
            high = solid[i, j, k]
    if low == 1 or high == 1:
        field[i, j, k] = 0.0


@wp.kernel
def enforce_walls(field: wp.array3d(dtype=float), axis: int, walls: int):
    """Zero the velocity component that points through a solid wall."""
    i, j, k = wp.tid()
    index = i
    if axis == 1:
        index = j
    elif axis == 2:
        index = k
    last = field.shape[axis] - 1
    low_bit = wp.where(axis == 0, 0, wp.where(axis == 1, 2, 4))
    if index == 0 and is_wall(walls, low_bit):
        field[i, j, k] = 0.0
    if index == last and is_wall(walls, low_bit + 1):
        field[i, j, k] = 0.0


@wp.kernel
def set_opening_velocity(
    field: wp.array3d(dtype=float),
    axis: int,
    cell: float,
    face_offset: wp.vec3,
    centre: wp.vec3,
    direction: wp.vec3,
    across: wp.vec3,
    up_across: wp.vec3,
    shape: int,
    half_across: float,
    half_up: float,
    speed: float,
    cone: float,
    turbulence: float,
    step_seed: int,
    seed: int,
):
    """Impose the inflow on the faces of one velocity component that lie inside the opening: a disc or a rectangle in the plane across `direction`, a few cells deep."""
    i, j, k = wp.tid()
    position = wp.vec3((float(i) + face_offset[0]) * cell, (float(j) + face_offset[1]) * cell, (float(k) + face_offset[2]) * cell)
    relative = position - centre
    along = wp.dot(relative, direction)
    sideways = wp.dot(relative, across)
    upwards = wp.dot(relative, up_across)
    if along < -0.6 * cell or along > 3.5 * cell:
        return
    inside = False
    if shape == SHAPE_CIRCLE:
        inside = (sideways * sideways + upwards * upwards) < half_across * half_across
    else:
        inside = wp.abs(sideways) < half_across and wp.abs(upwards) < half_up
    if not inside:
        return
    state = wp.rand_init(seed + i * 7919 + k * 104729 + j * 31, step_seed)
    noise = 1.0 + turbulence * (wp.randf(state) - 0.5) * 2.0
    spread = cone * speed * 0.5 * (sideways / half_across * across[axis] + upwards / half_up * up_across[axis])
    field[i, j, k] = speed * noise * direction[axis] + spread


@wp.kernel
def compute_divergence(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    cell_count: wp.array3d(dtype=int),
    solid: wp.array3d(dtype=int),
    inverse_cell: float,
    divergence: wp.array3d(dtype=float),
    pressure: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    if cell_count[i, j, k] == 0 or solid[i, j, k] == 1:
        divergence[i, j, k] = 0.0
        pressure[i, j, k] = 0.0
        return
    divergence[i, j, k] = (u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k] + w[i, j, k + 1] - w[i, j, k]) * inverse_cell


@wp.func
def neighbour_pressure(pressure: wp.array3d(dtype=float), cell_count: wp.array3d(dtype=int), solid: wp.array3d(dtype=int), walls: int, i: int, j: int, k: int):
    """Returns (pressure, counts) of a neighbour: air and open out-of-domain are p = 0 and count; a solid wall or obstacle cell does not count."""
    if i < 0:
        return wp.vec2(0.0, wp.where(is_wall(walls, 0), 0.0, 1.0))
    if i >= pressure.shape[0]:
        return wp.vec2(0.0, wp.where(is_wall(walls, 1), 0.0, 1.0))
    if j < 0:
        return wp.vec2(0.0, wp.where(is_wall(walls, 2), 0.0, 1.0))
    if j >= pressure.shape[1]:
        return wp.vec2(0.0, wp.where(is_wall(walls, 3), 0.0, 1.0))
    if k < 0:
        return wp.vec2(0.0, wp.where(is_wall(walls, 4), 0.0, 1.0))
    if k >= pressure.shape[2]:
        return wp.vec2(0.0, wp.where(is_wall(walls, 5), 0.0, 1.0))
    if solid[i, j, k] == 1:
        return wp.vec2(0.0, 0.0)
    if cell_count[i, j, k] == 0:
        return wp.vec2(0.0, 1.0)
    return wp.vec2(pressure[i, j, k], 1.0)


@wp.kernel
def relax_pressure(
    pressure: wp.array3d(dtype=float),
    divergence: wp.array3d(dtype=float),
    cell_count: wp.array3d(dtype=int),
    solid: wp.array3d(dtype=int),
    cell: float,
    omega: float,
    parity: int,
    walls: int,
):
    """One red-black successive over-relaxation sweep (in place, safe because a colour only reads the other colour)."""
    i, j, k = wp.tid()
    if (i + j + k) % 2 != parity:
        return
    if cell_count[i, j, k] == 0 or solid[i, j, k] == 1:
        return
    total = float(0.0)
    neighbours = float(0.0)
    for axis in range(6):
        ni = i
        nj = j
        nk = k
        if axis == 0:
            ni = i - 1
        elif axis == 1:
            ni = i + 1
        elif axis == 2:
            nj = j - 1
        elif axis == 3:
            nj = j + 1
        elif axis == 4:
            nk = k - 1
        else:
            nk = k + 1
        sample = neighbour_pressure(pressure, cell_count, solid, walls, ni, nj, nk)
        total = total + sample[0]
        neighbours = neighbours + sample[1]
    if neighbours < 1.0:
        return
    target = (total - divergence[i, j, k] * cell * cell) / neighbours
    pressure[i, j, k] = (1.0 - omega) * pressure[i, j, k] + omega * target


@wp.func
def cell_pressure(pressure: wp.array3d(dtype=float), cell_count: wp.array3d(dtype=int), i: int, j: int, k: int) -> float:
    if i < 0 or k < 0 or j < 0 or i >= pressure.shape[0] or j >= pressure.shape[1] or k >= pressure.shape[2]:
        return 0.0
    if cell_count[i, j, k] == 0:
        return 0.0
    return pressure[i, j, k]


@wp.kernel
def apply_pressure_u(u: wp.array3d(dtype=float), pressure: wp.array3d(dtype=float), cell_count: wp.array3d(dtype=int), solid: wp.array3d(dtype=int), inverse_cell: float, walls: int):
    i, j, k = wp.tid()
    if (i == 0 and is_wall(walls, 0)) or (i == pressure.shape[0] and is_wall(walls, 1)):
        return
    if (i > 0 and solid[i - 1, j, k] == 1) or (i < pressure.shape[0] and solid[wp.min(i, pressure.shape[0] - 1), j, k] == 1):
        return
    inside_low = i > 0 and cell_count[i - 1, j, k] > 0
    inside_high = i < pressure.shape[0] and cell_count[wp.min(i, pressure.shape[0] - 1), j, k] > 0
    if inside_low or inside_high:
        u[i, j, k] = u[i, j, k] - (cell_pressure(pressure, cell_count, i, j, k) - cell_pressure(pressure, cell_count, i - 1, j, k)) * inverse_cell


@wp.kernel
def apply_pressure_v(v: wp.array3d(dtype=float), pressure: wp.array3d(dtype=float), cell_count: wp.array3d(dtype=int), solid: wp.array3d(dtype=int), inverse_cell: float, walls: int):
    i, j, k = wp.tid()
    if (j == 0 and is_wall(walls, 2)) or (j == pressure.shape[1] and is_wall(walls, 3)):
        return
    if (j > 0 and solid[i, j - 1, k] == 1) or (j < pressure.shape[1] and solid[i, wp.min(j, pressure.shape[1] - 1), k] == 1):
        return
    inside_low = j > 0 and cell_count[i, j - 1, k] > 0
    inside_high = j < pressure.shape[1] and cell_count[i, wp.min(j, pressure.shape[1] - 1), k] > 0
    if inside_low or inside_high:
        v[i, j, k] = v[i, j, k] - (cell_pressure(pressure, cell_count, i, j, k) - cell_pressure(pressure, cell_count, i, j - 1, k)) * inverse_cell


@wp.kernel
def apply_pressure_w(w: wp.array3d(dtype=float), pressure: wp.array3d(dtype=float), cell_count: wp.array3d(dtype=int), solid: wp.array3d(dtype=int), inverse_cell: float, walls: int):
    i, j, k = wp.tid()
    if (k == 0 and is_wall(walls, 4)) or (k == pressure.shape[2] and is_wall(walls, 5)):
        return
    if (k > 0 and solid[i, j, k - 1] == 1) or (k < pressure.shape[2] and solid[i, j, wp.min(k, pressure.shape[2] - 1)] == 1):
        return
    inside_low = k > 0 and cell_count[i, j, k - 1] > 0
    inside_high = k < pressure.shape[2] and cell_count[i, j, wp.min(k, pressure.shape[2] - 1)] > 0
    if inside_low or inside_high:
        w[i, j, k] = w[i, j, k] - (cell_pressure(pressure, cell_count, i, j, k) - cell_pressure(pressure, cell_count, i, j, k - 1)) * inverse_cell


@wp.kernel
def residual_divergence(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    cell_count: wp.array3d(dtype=int),
    inverse_cell: float,
    worst: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    if cell_count[i, j, k] == 0:
        return
    value = (u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k] + w[i, j, k + 1] - w[i, j, k]) * inverse_cell
    wp.atomic_max(worst, 0, wp.abs(value))


@wp.kernel
def grid_to_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    old_u: wp.array3d(dtype=float),
    old_v: wp.array3d(dtype=float),
    old_w: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    inverse_cell: float,
    flip_ratio: float,
    delta_time: float,
    domain_size: wp.vec3,
    wall_margin: float,
    walls: int,
):
    tid = wp.tid()
    if alive[tid] == 0:
        return
    position = positions[tid]
    g = position * inverse_cell
    new_velocity = wp.vec3(
        sample_field(u, g[0], g[1] - 0.5, g[2] - 0.5),
        sample_field(v, g[0] - 0.5, g[1], g[2] - 0.5),
        sample_field(w, g[0] - 0.5, g[1] - 0.5, g[2]),
    )
    old_velocity = wp.vec3(
        sample_field(old_u, g[0], g[1] - 0.5, g[2] - 0.5),
        sample_field(old_v, g[0] - 0.5, g[1], g[2] - 0.5),
        sample_field(old_w, g[0] - 0.5, g[1] - 0.5, g[2]),
    )
    velocity = flip_ratio * (velocities[tid] + new_velocity - old_velocity) + (1.0 - flip_ratio) * new_velocity
    target = position + velocity * delta_time
    advanced = resolve_solid(solid, position, target, inverse_cell)
    for axis in range(3):
        if advanced[axis] != target[axis]:
            velocity[axis] = 0.0
    position = advanced
    for axis in range(3):
        if position[axis] < wall_margin and is_wall(walls, axis * 2):
            position[axis] = wall_margin
            velocity[axis] = wp.max(velocity[axis], 0.0)
        if position[axis] > domain_size[axis] - wall_margin and is_wall(walls, axis * 2 + 1):
            position[axis] = domain_size[axis] - wall_margin
            velocity[axis] = wp.min(velocity[axis], 0.0)
    if position[0] < 0.0 or position[2] < 0.0 or position[0] >= domain_size[0] or position[2] >= domain_size[2] or position[1] >= domain_size[1]:
        alive[tid] = 0
    positions[tid] = position
    velocities[tid] = velocity


@wp.kernel
def mark_valid(weight: wp.array3d(dtype=float), valid: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    if weight[i, j, k] > 1.0e-6:
        valid[i, j, k] = 1.0
    else:
        valid[i, j, k] = 0.0


@wp.kernel
def spawn_inflow(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    first: int,
    centre: wp.vec3,
    direction: wp.vec3,
    across: wp.vec3,
    up_across: wp.vec3,
    shape: int,
    half_across: float,
    half_up: float,
    speed: float,
    delta_time: float,
    cone: float,
    turbulence: float,
    step_seed: int,
    seed: int,
):
    """Particles filling the prism the inflow pushes out of the opening during one step (a disc or a rectangle times speed * dt)."""
    tid = wp.tid()
    state = wp.rand_init(seed + step_seed * 1315423911, tid)
    sideways = float(0.0)
    upwards = float(0.0)
    if shape == SHAPE_CIRCLE:
        angle = wp.randf(state) * 6.2831853
        distance = half_across * wp.sqrt(wp.randf(state))
        sideways = distance * wp.cos(angle)
        upwards = distance * wp.sin(angle)
    else:
        sideways = (wp.randf(state) * 2.0 - 1.0) * half_across
        upwards = (wp.randf(state) * 2.0 - 1.0) * half_up
    depth = wp.randf(state) * speed * delta_time + 0.004
    noise = 1.0 + turbulence * (wp.randf(state) - 0.5) * 2.0
    lateral = turbulence * speed * 0.25
    spread = across * (cone * speed * sideways / half_across) + up_across * (cone * speed * upwards / half_up)
    jitter = across * (lateral * (wp.randf(state) - 0.5)) + up_across * (lateral * (wp.randf(state) - 0.5))
    positions[first + tid] = centre + across * sideways + up_across * upwards + direction * depth
    velocities[first + tid] = direction * (speed * noise) + spread + jitter
    alive[first + tid] = 1


@wp.kernel
def remove_in_boxes(
    positions: wp.array(dtype=wp.vec3),
    box_low: wp.array(dtype=wp.vec3),
    box_high: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    life: wp.array(dtype=float),
    use_life: int,
):
    """Outflow regions: delete the particles inside any box (liquid particles through `alive`, whitewater through `life`)."""
    tid = wp.tid()
    if use_life == 1:
        if life[tid] <= 0.0:
            return
    else:
        if alive[tid] == 0:
            return
    position = positions[tid]
    for box in range(box_low.shape[0]):
        low = box_low[box]
        high = box_high[box]
        if position[0] >= low[0] and position[0] <= high[0] and position[1] >= low[1] and position[1] <= high[1] and position[2] >= low[2] and position[2] <= high[2]:
            if use_life == 1:
                life[tid] = 0.0
            else:
                alive[tid] = 0
            return
