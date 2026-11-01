# SPDX-License-Identifier: Apache-2.0
"""A pool fed by a faucet must rise by the volume pumped in, with the default --separation."""

import math
import sys

import numpy as np
import warp as wp

from spume import cli, flip

POOL_HEIGHT = 0.3
FRAMES = 90


def column_top_height(positions, cell, low, high):
    inside = (positions[:, 0] > low) & (positions[:, 0] < high) & (positions[:, 2] > low) & (positions[:, 2] < high)
    columns = np.floor(positions[inside][:, [0, 2]] / cell).astype(int)
    column_keys = columns[:, 0] * 1000 + columns[:, 1]
    heights = positions[inside][:, 1]
    top_by_column = {}
    for key, height in zip(column_keys, heights):
        top_by_column[key] = max(top_by_column.get(key, 0.0), height)
    tops = np.array(sorted(top_by_column.values()))
    keep = tops[int(0.05 * len(tops)): int(math.ceil(0.95 * len(tops)))]
    return float(keep.mean())


def test_a_pool_fed_by_a_faucet_rises_by_the_pumped_volume(monkeypatch):
    arguments = [
        "spume", "--device", "cpu", "--out", "unused", "--cell", "0.1", "--length", "2", "--width", "2", "--height", "1.6", "--fps", "30",
        "--frames", str(FRAMES), "--no-whitewater", "--flip", "0.97", "--turbulence", "0", "--max-particles", "400000",
        "--inflow", "side", "--inflow-position", "1", "1.4", "1", "--direction", "0", "-1", "0", "--opening", "0.2", "--speed", "2.5", "--rise", "0.05", "--feed", "30",
        "--obstacle-box", "0", "0", "0", "2", "0.4", "0.1",
        "--obstacle-box", "0", "0", "1.9", "2", "0.4", "2",
        "--obstacle-box", "0", "0", "0", "0.1", "0.4", "2",
        "--obstacle-box", "1.9", "0", "0", "2", "0.4", "2",
        "--liquid-box", "0.1", "0", "0.1", "1.9", str(POOL_HEIGHT), "1.9",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    options = cli.parse_arguments()
    wp.config.quiet = True
    wp.init()
    simulation = cli.Simulation(options)
    initial_count = simulation.particle_count
    start_height = column_top_height(simulation.positions.numpy()[:initial_count], options.cell, 0.1, 1.9)
    reference_velocity = max(max(inflow.speed for inflow in simulation.inflows) * 1.1, math.sqrt(2.0 * flip.GRAVITY * options.height))
    steps_per_frame = max(1, math.ceil((1.0 / options.fps) / (options.cfl * options.cell / reference_velocity)))
    delta_time = 1.0 / options.fps / steps_per_frame
    for frame in range(FRAMES):
        for substep in range(steps_per_frame):
            simulation.step((frame + substep / steps_per_frame) / options.fps, delta_time)
    count = simulation.particle_count
    alive = simulation.alive.numpy()[:count] == 1
    final_height = column_top_height(simulation.positions.numpy()[:count][alive], options.cell, 0.1, 1.9)
    pumped_volume = (int(alive.sum()) - initial_count) * options.cell**3 / options.particles_per_cell
    expected_rise = pumped_volume / (1.8 * 1.8)
    assert abs((final_height - start_height) - expected_rise) < 0.05 * (POOL_HEIGHT + expected_rise)
