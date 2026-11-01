# SPDX-License-Identifier: Apache-2.0
"""Sparse dispatch: grid kernels run only over the blocks of cells that hold liquid, not over the whole box.

A jet fills a few percent of its domain, so most of a dense sweep touches air. The grid is cut into blocks of BLOCK_SIZE cells a side. Each step
the blocks holding live particles are marked, dilated by one block (a block is wider than the velocity extrapolation reaches), and listed on the GPU.
A kernel launched with dim = block count * BLOCK_CELLS turns its thread index back into a cell with `cell_of_thread`. Fields stay dense arrays;
only the work is sparse, so sampling at any position keeps working.

Fields a kernel writes only in active blocks must be cleared in full each step (a memset is about 50 microseconds for this grid).
"""

import warp as wp

BLOCK_SIZE = 8
BLOCK_CELLS = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE


@wp.func
def cell_of_thread(blocks: wp.array(dtype=wp.vec3i), thread: int) -> wp.vec3i:
    """The cell a sparse kernel's thread stands on. It can lie one layer past the cell grid: faces have one more layer than cells."""
    block = blocks[thread // 512]
    local = thread % 512
    return wp.vec3i(block[0] * 8 + (local & 7), block[1] * 8 + ((local >> 3) & 7), block[2] * 8 + (local >> 6))


@wp.func
def outside(field: wp.array3d(dtype=float), cell: wp.vec3i) -> bool:
    return cell[0] >= field.shape[0] or cell[1] >= field.shape[1] or cell[2] >= field.shape[2]


@wp.kernel
def mark_blocks(positions: wp.array(dtype=wp.vec3), alive: wp.array(dtype=int), inverse_cell: float, mask: wp.array3d(dtype=int)):
    tid = wp.tid()
    if alive[tid] == 0:
        return
    grid_position = positions[tid] * inverse_cell
    i = wp.clamp(int(grid_position[0]), 0, mask.shape[0] * 8 - 1)
    j = wp.clamp(int(grid_position[1]), 0, mask.shape[1] * 8 - 1)
    k = wp.clamp(int(grid_position[2]), 0, mask.shape[2] * 8 - 1)
    mask[i // 8, j // 8, k // 8] = 1


@wp.kernel
def list_dilated_blocks(mask: wp.array3d(dtype=int), blocks: wp.array(dtype=wp.vec3i), counter: wp.array(dtype=int)):
    """Appends every block that is marked or touches a marked block."""
    i, j, k = wp.tid()
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ni = i + di
                nj = j + dj
                nk = k + dk
                if ni >= 0 and nj >= 0 and nk >= 0 and ni < mask.shape[0] and nj < mask.shape[1] and nk < mask.shape[2] and mask[ni, nj, nk] == 1:
                    blocks[wp.atomic_add(counter, 0, 1)] = wp.vec3i(i, j, k)
                    return


@wp.kernel
def list_liquid_cells(
    blocks: wp.array(dtype=wp.vec3i),
    cell_count: wp.array3d(dtype=int),
    solid: wp.array3d(dtype=int),
    cells: wp.array(dtype=wp.vec3i),
    counter: wp.array(dtype=int),
):
    cell = cell_of_thread(blocks, wp.tid())
    if cell[0] >= cell_count.shape[0] or cell[1] >= cell_count.shape[1] or cell[2] >= cell_count.shape[2]:
        return
    if cell_count[cell[0], cell[1], cell[2]] > 0 and solid[cell[0], cell[1], cell[2]] == 0:
        cells[wp.atomic_add(counter, 0, 1)] = cell


class ActiveRegion:
    """The active block list and liquid cell list of one grid, rebuilt every step."""

    def __init__(self, size, device, launcher):
        self.launcher = launcher
        self.block_size = tuple(size[axis] // BLOCK_SIZE + 1 for axis in range(3))
        total_blocks = self.block_size[0] * self.block_size[1] * self.block_size[2]
        total_cells = size[0] * size[1] * size[2]
        self.mask = wp.zeros(self.block_size, dtype=int, device=device)
        self.blocks = wp.zeros(total_blocks, dtype=wp.vec3i, device=device)
        self.block_counter = wp.zeros(1, dtype=int, device=device)
        self.cells = wp.zeros(total_cells, dtype=wp.vec3i, device=device)
        self.cell_counter = wp.zeros(1, dtype=int, device=device)
        self.block_count = 0
        self.cell_count = 0

    def thread_count(self):
        return self.block_count * BLOCK_CELLS

    def update_blocks(self, positions, alive, particle_count, inverse_cell):
        self.mask.zero_()
        self.block_counter.zero_()
        self.launcher.launch(mark_blocks, particle_count, [positions, alive, inverse_cell, self.mask])
        self.launcher.launch(list_dilated_blocks, self.block_size, [self.mask, self.blocks, self.block_counter])
        self.block_count = int(self.block_counter.numpy()[0])

    def update_cells(self, cell_count, solid):
        self.cell_counter.zero_()
        self.launcher.launch(list_liquid_cells, self.thread_count(), [self.blocks, cell_count, solid, self.cells, self.cell_counter])
        self.cell_count = int(self.cell_counter.numpy()[0])
