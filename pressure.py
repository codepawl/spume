# SPDX-License-Identifier: Apache-2.0
"""Pressure projection solvers on the GPU, next to the red-black SOR sweep in `flip.py`.

The system is the 7-point Poisson problem over the liquid cells: for each liquid cell, n p - (sum of liquid neighbour pressures) = -div * cell^2,
where n counts every neighbour that is not a solid wall or obstacle (air neighbours have p = 0). It is symmetric positive definite, so
conjugate gradients apply. This is a matrix-free CG with a Jacobi (inverse diagonal) preconditioner; the dot products and the step lengths stay
on the GPU, so an iteration never waits for the host. The current pressure is the starting guess.
"""

import warp as wp

from flip import neighbour_pressure


@wp.func
def stencil(
    field: wp.array3d(dtype=float),
    cell_count: wp.array3d(dtype=int),
    solid: wp.array3d(dtype=int),
    walls: int,
    i: int,
    j: int,
    k: int,
) -> wp.vec2:
    """Returns (number of counted neighbours, sum of their `field` values) for the cell."""
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
        sample = neighbour_pressure(field, cell_count, solid, walls, ni, nj, nk)
        total = total + sample[0]
        neighbours = neighbours + sample[1]
    return wp.vec2(neighbours, total)


@wp.kernel
def cg_start(
    pressure: wp.array3d(dtype=float),
    divergence: wp.array3d(dtype=float),
    cell_count: wp.array3d(dtype=int),
    solid: wp.array3d(dtype=int),
    cell: float,
    walls: int,
    residual: wp.array3d(dtype=float),
    direction: wp.array3d(dtype=float),
    inverse_diagonal: wp.array3d(dtype=float),
    scalars: wp.array(dtype=float),
):
    """residual = b - A p, direction = M^-1 residual; scalars[0] accumulates residual . direction."""
    i, j, k = wp.tid()
    residual[i, j, k] = 0.0
    direction[i, j, k] = 0.0
    inverse_diagonal[i, j, k] = 0.0
    if cell_count[i, j, k] == 0 or solid[i, j, k] == 1:
        return
    around = stencil(pressure, cell_count, solid, walls, i, j, k)
    if around[0] < 1.0:
        return
    inverse = 1.0 / around[0]
    value = -divergence[i, j, k] * cell * cell - (around[0] * pressure[i, j, k] - around[1])
    residual[i, j, k] = value
    inverse_diagonal[i, j, k] = inverse
    direction[i, j, k] = value * inverse
    wp.atomic_add(scalars, 0, value * value * inverse)


@wp.kernel
def cg_multiply(
    direction: wp.array3d(dtype=float),
    cell_count: wp.array3d(dtype=int),
    solid: wp.array3d(dtype=int),
    inverse_diagonal: wp.array3d(dtype=float),
    walls: int,
    product: wp.array3d(dtype=float),
    scalars: wp.array(dtype=float),
):
    """product = A direction; scalars[2] accumulates direction . product."""
    i, j, k = wp.tid()
    product[i, j, k] = 0.0
    if inverse_diagonal[i, j, k] == 0.0:
        return
    around = stencil(direction, cell_count, solid, walls, i, j, k)
    value = around[0] * direction[i, j, k] - around[1]
    product[i, j, k] = value
    wp.atomic_add(scalars, 2, value * direction[i, j, k])


@wp.kernel
def cg_update(
    pressure: wp.array3d(dtype=float),
    residual: wp.array3d(dtype=float),
    direction: wp.array3d(dtype=float),
    product: wp.array3d(dtype=float),
    inverse_diagonal: wp.array3d(dtype=float),
    scalars: wp.array(dtype=float),
):
    """pressure += alpha direction; residual -= alpha product; scalars[1] accumulates residual . (M^-1 residual)."""
    i, j, k = wp.tid()
    if inverse_diagonal[i, j, k] == 0.0:
        return
    alpha = float(0.0)
    if scalars[2] > 1.0e-30:
        alpha = scalars[0] / scalars[2]
    pressure[i, j, k] = pressure[i, j, k] + alpha * direction[i, j, k]
    value = residual[i, j, k] - alpha * product[i, j, k]
    residual[i, j, k] = value
    wp.atomic_add(scalars, 1, value * value * inverse_diagonal[i, j, k])


@wp.kernel
def cg_direction(
    residual: wp.array3d(dtype=float),
    direction: wp.array3d(dtype=float),
    inverse_diagonal: wp.array3d(dtype=float),
    scalars: wp.array(dtype=float),
):
    """direction = M^-1 residual + beta direction."""
    i, j, k = wp.tid()
    if inverse_diagonal[i, j, k] == 0.0:
        return
    beta = float(0.0)
    if scalars[0] > 1.0e-30:
        beta = scalars[1] / scalars[0]
    direction[i, j, k] = residual[i, j, k] * inverse_diagonal[i, j, k] + beta * direction[i, j, k]


@wp.kernel
def cg_roll_scalars(scalars: wp.array(dtype=float)):
    scalars[0] = scalars[1]
    scalars[1] = 0.0
    scalars[2] = 0.0


class ConjugateGradient:
    """Preconditioned conjugate gradients for the pressure Poisson system. One instance holds the work arrays."""

    def __init__(self, size, device):
        self.device = device
        self.residual = wp.zeros(size, dtype=float, device=device)
        self.direction = wp.zeros(size, dtype=float, device=device)
        self.product = wp.zeros(size, dtype=float, device=device)
        self.inverse_diagonal = wp.zeros(size, dtype=float, device=device)
        self.scalars = wp.zeros(3, dtype=float, device=device)
        self.size = size

    def launch(self, kernel, inputs, dim=None):
        wp.launch(kernel, dim=dim or self.size, inputs=inputs, device=self.device)

    def solve(self, pressure, divergence, cell_count, solid, cell, walls, iterations):
        self.scalars.zero_()
        self.launch(cg_start, [pressure, divergence, cell_count, solid, cell, walls, self.residual, self.direction, self.inverse_diagonal, self.scalars])
        for _ in range(iterations):
            self.launch(cg_multiply, [self.direction, cell_count, solid, self.inverse_diagonal, walls, self.product, self.scalars])
            self.launch(cg_update, [pressure, self.residual, self.direction, self.product, self.inverse_diagonal, self.scalars])
            self.launch(cg_direction, [self.residual, self.direction, self.inverse_diagonal, self.scalars])
            self.launch(cg_roll_scalars, [self.scalars], dim=1)
