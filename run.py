# SPDX-License-Identifier: Apache-2.0
"""Command line liquid jet: FLIP on the GPU, whitewater, marching cubes, export.

    uv run python run.py --out <folder> [options]

Every physical setting is an option, so other shots reuse it by changing numbers: a vertical jet out of the floor (default), a lateral torrent from a pipe
mouth (--inflow side), a closed box with walls and a ceiling (--closed), solid obstacles (--obstacle-mask, --obstacle-box), regions that drain
the liquid (--outflow-box) and any number of inflows (--inflows inflows.json).
"""

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import warp as wp

import flip
import preview
import pressure
import sparse
from launcher import Launcher
import surface
import whitewater
from export import SequenceWriter


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output folder (created)")
    parser.add_argument("--name", default="geyser")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cell", type=float, default=0.08, help="simulation cell size in metres")
    parser.add_argument("--width", type=float, default=9.0, help="domain width in z (and in x unless --length is given), metres")
    parser.add_argument("--length", type=float, default=None, help="domain length in x, metres (default: --width)")
    parser.add_argument("--height", type=float, default=20.0, help="domain height, metres")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--skip-seconds", type=float, default=0.0, help="simulate this long without writing frames (the run-up of a torrent), so the asset starts on the steady flow")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=18.5, help="peak jet speed at the opening, m/s")
    parser.add_argument("--inflow", choices=["floor", "side"], default="floor", help="floor: a vertical jet out of the floor at the middle; side: a lateral jet or pipe mouth in the x = 0 wall")
    parser.add_argument("--direction", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="inflow direction (default: up for floor, +x for side)")
    parser.add_argument("--inflow-position", type=float, nargs="+", default=None, metavar="COORD", help="opening centre in domain metres: X Z on the floor for --inflow floor; X Y Z for --inflow side (the opening may sit in any wall; X Z alone takes Y from --inflow-height). Default: the middle of the floor, or the x = 0 wall")
    parser.add_argument("--inflows", default=None, help="JSON file with a list of inflows {mode, position, direction, shape, size, speed, rise, feed, taper}; replaces the single-inflow flags (they remain the defaults of each entry)")
    parser.add_argument("--obstacle-mask", default=None, help=".npy bool array (nx, ny, nz) of solid cells on the solver grid")
    parser.add_argument("--obstacle-box", type=float, nargs=6, action="append", default=[], metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"), help="solid box in domain metres (repeatable)")
    parser.add_argument("--outflow-box", type=float, nargs=6, action="append", default=[], metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"), help="region in domain metres that deletes liquid and whitewater (repeatable)")
    parser.add_argument("--liquid-box", type=float, nargs=6, action="append", default=[], metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"), help="liquid at rest in this box at the start, in domain metres (repeatable): a pool or a tank")
    parser.add_argument("--inflow-height", type=float, default=1.5, help="side inflow: height of the opening centre above the floor, metres")
    parser.add_argument("--shape", choices=["circle", "rectangle"], default="circle", help="shape of the opening")
    parser.add_argument("--opening", type=float, default=0.6, help="circle opening diameter, metres")
    parser.add_argument("--opening-size", type=float, nargs=2, default=None, metavar=("WIDE", "HIGH"), help="rectangle opening: width across the flow and height, metres")
    parser.add_argument("--closed", action="store_true", help="solid walls on every face of the box and a ceiling (a culvert), not only the floor")
    parser.add_argument("--preview", default=None, help="save a point-cloud preview PNG of the liquid at three moments")
    parser.add_argument("--rise", type=float, default=0.2, help="seconds for the pressure pulse to reach full speed")
    parser.add_argument("--feed", type=float, default=2.2, help="seconds the jet is held at full speed")
    parser.add_argument("--taper", type=float, default=1.0, help="seconds the jet takes to die away after the feed")
    parser.add_argument("--cone", type=float, default=0.03, help="sideways speed at the rim as a fraction of the jet speed")
    parser.add_argument("--turbulence", type=float, default=0.08, help="slow waves in the inflow's speed and direction, fraction of the speed. Higher breaks the jet up sooner")
    parser.add_argument("--particles-per-cell", type=int, default=8)
    parser.add_argument("--separation", type=float, default=1.5, help="share of a cell's extra crowding the pressure solve pushes apart per step. Gives impacts their splash back")
    parser.add_argument("--flip", type=float, default=0.97, help="FLIP share of the PIC/FLIP blend")
    parser.add_argument("--cfl", type=float, default=1.0)
    parser.add_argument("--pressure", choices=["sor", "cg"], default="cg", help="pressure solver: Jacobi-preconditioned conjugate gradients, or red-black SOR")
    parser.add_argument("--iterations", type=int, default=20, help="pressure iterations per step (CG iterations or SOR sweeps; SOR needs about 60)")
    parser.add_argument("--omega", type=float, default=1.9, help="over-relaxation factor")
    parser.add_argument("--viscosity", type=float, default=0.0, help="kinematic viscosity, m^2/s (water 1e-6 is 0 here, honey about 0.01, thick cream about 0.1). Use a low --flip with it")
    parser.add_argument("--floor-friction", type=float, default=0.0, help="slows liquid in the cell row on the floor, per second. A slip floor makes a fast sheet thinner than a cell, which the mesh cannot show. Try 5 to 15")
    parser.add_argument("--viscosity-iterations", type=int, default=20, help="Jacobi sweeps of the viscosity solve per step (rounded down to even)")
    parser.add_argument("--max-particles", type=int, default=3_000_000)
    parser.add_argument("--mesh-cell", type=float, default=0.1, help="marching cubes cell size, metres")
    parser.add_argument("--mesh-radius", type=float, default=2.2, help="splat radius in mesh cells")
    parser.add_argument("--iso", type=float, default=0.35, help="surface level as a fraction of the interior density")
    parser.add_argument("--no-whitewater", action="store_true", help="skip spray, foam and bubbles entirely (no emission, no advection, empty spray files)")
    parser.add_argument("--ww-capacity", type=int, default=2_500_000)
    parser.add_argument("--ww-trapped-rate", type=float, default=60.0)
    parser.add_argument("--ww-crest-rate", type=float, default=120.0)
    parser.add_argument("--ww-life", type=float, nargs=2, default=[1.0, 3.0])
    parser.add_argument("--ww-export", type=int, default=22000, help="most whitewater points kept per frame")
    parser.add_argument("--ww-buoyancy", type=float, default=1.6)
    parser.add_argument("--ww-drag", type=float, default=8.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def video_memory_used_mib():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(result.stdout.strip().splitlines()[0])


@dataclass
class Inflow:
    """One opening: where it sits, which way it pushes, its shape and its pressure pulse."""

    mode: str
    centre: tuple
    direction: np.ndarray
    shape: str
    half_across: float
    half_up: float
    area: float
    speed: float
    rise: float
    feed: float
    taper: float
    delay: float = 0.0
    across: np.ndarray = None
    up_across: np.ndarray = None

    def __post_init__(self):
        self.direction = self.direction / np.linalg.norm(self.direction)
        across = np.cross([0.0, 1.0, 0.0], self.direction)
        self.across = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(across) < 1e-6 else across / np.linalg.norm(across)
        self.up_across = np.cross(self.direction, self.across)
        self.shape_code = flip.SHAPE_RECTANGLE if self.shape == "rectangle" else flip.SHAPE_CIRCLE
        self.gpu_centre = wp.vec3(*self.centre)
        self.gpu_direction = wp.vec3(*self.direction)
        self.gpu_across = wp.vec3(*self.across)
        self.gpu_up_across = wp.vec3(*self.up_across)
        self.spawn_remainder = 0.0

    def speed_at(self, seconds):
        """Pressure pulse: smooth rise, a held feed, then a smooth collapse to zero."""
        seconds -= self.delay
        if seconds <= 0.0:
            return 0.0
        rise = min(1.0, seconds / self.rise)
        gain = rise * rise * (3.0 - 2.0 * rise)
        after_feed = (seconds - self.feed) / self.taper
        if after_feed > 0.0:
            fade = min(1.0, after_feed)
            gain *= 1.0 - fade * fade * (3.0 - 2.0 * fade)
        return self.speed * gain

    def settings(self):
        return {
            "mode": self.mode, "position": list(self.centre), "direction": self.direction.tolist(), "shape": self.shape,
            "size": [self.half_across * 2.0, self.half_up * 2.0], "speed": self.speed, "rise": self.rise, "feed": self.feed, "taper": self.taper, "delay": self.delay,
        }


def make_inflow(entry, options, length):
    """Build an inflow from one JSON entry; every missing key falls back to the single-inflow command line flag."""
    mode = entry.get("mode", options.inflow)
    position = entry.get("position", options.inflow_position if not options.inflows else None)
    if mode == "side":
        if position is None:
            position = (0.0, options.inflow_height, options.width * 0.5)
        elif len(position) == 2:
            position = (position[0], options.inflow_height, position[1])
        default_direction = (1.0, 0.0, 0.0)
    else:
        if position is None:
            position = (length * 0.5, options.width * 0.5)
        position = (position[0], 0.0, position[1])
        default_direction = (0.0, 1.0, 0.0)
    direction = np.array(entry.get("direction") or options.direction or default_direction, np.float64)
    shape = entry.get("shape", options.shape)
    size = entry.get("size", options.opening_size if shape == "rectangle" else options.opening)
    if shape == "rectangle":
        wide, high = (size, size) if isinstance(size, (int, float)) else (size or (options.opening, options.opening))
        half_across, half_up, area = wide * 0.5, high * 0.5, wide * high
    else:
        diameter = size if isinstance(size, (int, float)) else size[0]
        half_across = half_up = diameter * 0.5
        area = math.pi * half_across**2
    return Inflow(
        mode=mode, centre=tuple(float(value) for value in position), direction=direction, shape=shape, half_across=half_across, half_up=half_up, area=area,
        speed=float(entry.get("speed", options.speed)), rise=float(entry.get("rise", options.rise)), feed=float(entry.get("feed", options.feed)), taper=float(entry.get("taper", options.taper)), delay=float(entry.get("delay", 0.0)),
    )


def load_inflows(options, length):
    if options.inflows:
        entries = json.loads(Path(options.inflows).read_text())
        if not isinstance(entries, list) or not entries:
            raise SystemExit("--inflows needs a JSON list with at least one inflow")
    else:
        entries = [{}]
    return [make_inflow(entry, options, length) for entry in entries]


def side_wall_bits(inflows, size, cell):
    """Solid wall bits for the faces that carry a side inflow (the pipe wall). A mid-air inflow adds none."""
    bits = 0
    extent = [size[0] * cell, size[1] * cell, size[2] * cell]
    for inflow in inflows:
        if inflow.mode != "side":
            continue
        for axis in range(3):
            if abs(inflow.centre[axis]) < cell:
                bits |= 1 << (axis * 2)
            if abs(inflow.centre[axis] - extent[axis]) < cell:
                bits |= 1 << (axis * 2 + 1)
    return bits


def paint_box(mask, box, cell):
    """Mark every cell the box overlaps."""
    low = np.floor(np.array(box[:3]) / cell).astype(int).clip(0, None)
    high = np.ceil(np.array(box[3:]) / cell).astype(int)
    mask[low[0] : high[0], low[1] : high[1], low[2] : high[2]] = True


def load_obstacles(options, size, cell):
    """Boolean solid-cell mask on the solver grid from --obstacle-mask and --obstacle-box, or None when there is no obstacle."""
    mask = np.zeros(size, bool)
    if options.obstacle_mask:
        loaded = np.load(options.obstacle_mask)
        if loaded.shape != tuple(size):
            raise SystemExit(f"--obstacle-mask has shape {loaded.shape}, the solver grid is {tuple(size)} (nx, ny, nz)")
        mask |= loaded.astype(bool)
    for box in options.obstacle_box:
        paint_box(mask, box, cell)
    return mask if mask.any() else None


def push_out_of_solid(vertices, normals, mask, cell):
    """Surface vertices that the particle splats left inside an obstacle move back along the opposite surface normal, towards the liquid."""
    upper = np.array(mask.shape) - 1
    for _ in range(16):
        index = np.clip(np.floor(vertices / cell).astype(int), 0, upper)
        inside = mask[index[:, 0], index[:, 1], index[:, 2]]
        if not inside.any():
            break
        vertices[inside] -= normals[inside] * (0.5 * cell)
    return vertices


class Simulation:
    def __init__(self, options):
        self.options = options
        self.device = options.device
        self.cell = options.cell
        self.length = options.length if options.length else options.width
        self.size = tuple(int(round(extent / options.cell)) for extent in (self.length, options.height, options.width))
        nx, ny, nz = self.size
        self.domain = wp.vec3(nx * self.cell, ny * self.cell, nz * self.cell)
        self.inflows = load_inflows(options, self.length)
        self.walls = flip.WALL_ALL if options.closed else flip.WALL_FLOOR | side_wall_bits(self.inflows, self.size, self.cell)
        device = self.device
        self.launcher = Launcher(device)
        self.solid_host = load_obstacles(options, self.size, self.cell)
        self.solid = wp.array(self.solid_host.astype(np.int32) if self.solid_host is not None else np.zeros(self.size, np.int32), dtype=int, device=device)
        self.outflow_low, self.outflow_high = self.build_outflow_boxes()
        self.cg = pressure.ConjugateGradient(self.size, device, self.launcher) if options.pressure == "cg" else None
        self.active = sparse.ActiveRegion(self.size, device, self.launcher)
        shapes = {"u": (nx + 1, ny, nz), "v": (nx, ny + 1, nz), "w": (nx, ny, nz + 1)}
        self.velocity = {name: wp.zeros(shape, dtype=float, device=device) for name, shape in shapes.items()}
        self.old = {name: wp.zeros(shape, dtype=float, device=device) for name, shape in shapes.items()}
        self.weight = {name: wp.zeros(shape, dtype=float, device=device) for name, shape in shapes.items()}
        self.valid = {name: wp.zeros(shape, dtype=float, device=device) for name, shape in shapes.items()}
        self.scratch = {name: wp.zeros(shape, dtype=float, device=device) for name, shape in shapes.items()}
        self.scratch_valid = {name: wp.zeros(shape, dtype=float, device=device) for name, shape in shapes.items()}
        self.viscous_source = {name: wp.zeros(shape, dtype=float, device=device) for name, shape in shapes.items()} if options.viscosity > 0.0 else None
        self.cell_count = wp.zeros(self.size, dtype=int, device=device)
        self.pressure = wp.zeros(self.size, dtype=float, device=device)
        self.divergence = wp.zeros(self.size, dtype=float, device=device)
        self.density = wp.zeros(self.size, dtype=float, device=device)
        self.shear = wp.zeros(self.size, dtype=float, device=device)
        self.normals = [wp.zeros(self.size, dtype=float, device=device) for _ in range(3)]
        self.worst_divergence = wp.zeros(1, dtype=float, device=device)

        capacity = options.max_particles
        self.positions = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.velocities = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.alive = wp.zeros(capacity, dtype=int, device=device)
        self.particle_count = 0
        self.spawn_remainder = 0.0
        self.fill_liquid_boxes()

        ww = 1 if options.no_whitewater else options.ww_capacity
        export_capacity = 1 if options.no_whitewater else options.ww_export
        self.ww_positions = wp.zeros(ww, dtype=wp.vec3, device=device)
        self.ww_velocities = wp.zeros(ww, dtype=wp.vec3, device=device)
        self.ww_life = wp.zeros(ww, dtype=float, device=device)
        self.ww_ids = wp.zeros(ww, dtype=int, device=device)
        self.ww_counter = wp.zeros(1, dtype=int, device=device)
        self.export_positions = wp.zeros(export_capacity, dtype=wp.vec3, device=device)
        self.export_velocities = wp.zeros(export_capacity, dtype=wp.vec3, device=device)
        self.export_ids = wp.zeros(export_capacity, dtype=int, device=device)
        self.export_kinds = wp.zeros(export_capacity, dtype=wp.uint8, device=device)
        self.export_counter = wp.zeros(1, dtype=int, device=device)

        mesh_size = tuple(int(math.ceil(extent / options.mesh_cell)) + 1 for extent in (self.length, options.height, options.width))
        self.surface = surface.SurfaceExtractor(mesh_size, (0.0, 0.0, 0.0), options.mesh_cell, self.cell, device, radius_cells=options.mesh_radius)
        self.iso_threshold = options.iso * surface.interior_density(options.particles_per_cell, self.cell, self.surface.radius)
        self.step_index = 0
        self.timers = {"sim": 0.0, "whitewater": 0.0, "mesh": 0.0, "export": 0.0}

    def fill_liquid_boxes(self):
        """Particles at rest in every --liquid-box, skipping solid cells, jittered inside their cells."""
        options = self.options
        random = np.random.default_rng(options.seed)
        batches = []
        for box in options.liquid_box:
            mask = np.zeros(self.size, bool)
            paint_box(mask, box, self.cell)
            if self.solid_host is not None:
                mask &= ~self.solid_host
            cells = np.argwhere(mask).repeat(options.particles_per_cell, axis=0)
            batches.append((cells + random.random(cells.shape)) * self.cell)
        if not batches:
            return
        positions = np.concatenate(batches).astype(np.float32)
        if len(positions) > self.positions.shape[0]:
            raise SystemExit(f"--liquid-box needs {len(positions)} particles, --max-particles is {self.positions.shape[0]}")
        count = len(positions)
        wp.copy(self.positions, wp.array(positions, dtype=wp.vec3, device=self.device), count=count)
        wp.copy(self.alive, wp.array(np.ones(count, np.int32), dtype=int, device=self.device), count=count)
        self.particle_count = count

    def build_outflow_boxes(self):
        boxes = self.options.outflow_box
        if not boxes:
            return None, None
        low = wp.array([wp.vec3(*box[:3]) for box in boxes], dtype=wp.vec3, device=self.device)
        high = wp.array([wp.vec3(*box[3:]) for box in boxes], dtype=wp.vec3, device=self.device)
        return low, high

    def drain_outflow(self, positions, count, alive, life, use_life):
        if self.outflow_low is not None and count > 0:
            self.launch(flip.remove_in_boxes, count, [positions, self.outflow_low, self.outflow_high, alive, life, use_life])

    def launch(self, kernel, dim, inputs):
        self.launcher.launch(kernel, dim, inputs)

    def extend(self, passes=2):
        """Velocity extrapolation into faces no particle touched, for whitewater: liquid particles only read faces they splatted to, which keep their value.
        An even pass count ends back in the original array, so nothing is copied."""
        for name in ("u", "v", "w"):
            original = self.velocity[name]
            field, scratch = original, self.scratch[name]
            valid, scratch_valid = self.valid[name], self.scratch_valid[name]
            scratch.zero_()
            valid.zero_()
            scratch_valid.zero_()
            self.launch(flip.mark_valid, self.active.thread_count(), [self.active.blocks, self.weight[name], valid])
            for _ in range(passes):
                self.launch(flip.extend_face, self.active.thread_count(), [self.active.blocks, field, valid, scratch, scratch_valid])
                field, scratch = scratch, field
                valid, scratch_valid = scratch_valid, valid
            if field is not original:
                wp.copy(original, field)

    def diffuse(self, delta_time):
        """Implicit viscosity on the liquid faces, by Jacobi sweeps from the current velocity. An even sweep count ends back in the velocity array."""
        options = self.options
        blend = options.viscosity * delta_time / (self.cell * self.cell)
        threads = self.active.thread_count()
        for axis, name in enumerate(("u", "v", "w")):
            field, scratch = self.velocity[name], self.scratch[name]
            source = self.viscous_source[name]
            wp.copy(source, field)
            for _ in range(options.viscosity_iterations // 2 * 2):
                self.launch(flip.diffuse_face, threads, [self.active.blocks, field, source, self.weight[name], self.solid, axis, blend, self.walls, scratch])
                field, scratch = scratch, field

    def inflow_phase(self, inflow, seconds):
        """A jet flaps at a Strouhal number of about 0.3: speed over diameter sets how fast the waves at its opening change."""
        flaps_per_second = 0.3 * inflow.speed / (2.0 * max(inflow.half_across, inflow.half_up))
        return 2.0 * math.pi * flaps_per_second * seconds + 1.7 * self.inflows.index(inflow)

    def spawn(self, index, inflow, speed, delta_time, seconds):
        options = self.options
        particle_volume = self.cell**3 / options.particles_per_cell
        wanted = inflow.area * speed * delta_time / particle_volume + inflow.spawn_remainder
        count = int(wanted)
        inflow.spawn_remainder = wanted - count
        count = min(count, self.positions.shape[0] - self.particle_count)
        if count > 0:
            self.launch(
                flip.spawn_inflow,
                count,
                [
                    self.positions, self.velocities, self.alive, self.particle_count,
                    inflow.gpu_centre, inflow.gpu_direction, inflow.gpu_across, inflow.gpu_up_across, inflow.shape_code, inflow.half_across, inflow.half_up,
                    speed, delta_time, options.cone, options.turbulence, self.inflow_phase(inflow, seconds), self.step_index, options.seed + index * 1000003,
                ],
            )
            self.particle_count += count

    def step(self, seconds, delta_time):
        options = self.options
        inverse_cell = 1.0 / self.cell
        speeds = [inflow.speed_at(seconds) for inflow in self.inflows]
        started = time.perf_counter()
        for index, inflow in enumerate(self.inflows):
            self.spawn(index, inflow, speeds[index], delta_time, seconds)
        for name in ("u", "v", "w"):
            self.velocity[name].zero_()
            self.weight[name].zero_()
        self.cell_count.zero_()
        self.active.update_blocks(self.positions, self.alive, self.particle_count, inverse_cell)
        threads = self.active.thread_count()
        blocks = self.active.blocks
        u, v, w = self.velocity["u"], self.velocity["v"], self.velocity["w"]
        self.launch(
            flip.particles_to_grid,
            self.particle_count,
            [self.positions, self.velocities, self.alive, inverse_cell, u, self.weight["u"], v, self.weight["v"], w, self.weight["w"], self.cell_count],
        )
        for name in ("u", "v", "w"):
            self.launch(flip.normalise_face, threads, [blocks, self.velocity[name], self.weight[name], self.old[name]])
        self.active.update_cells(self.cell_count, self.solid)
        self.launch(flip.add_gravity, threads, [blocks, v, delta_time])
        for axis, (field, face_offset) in enumerate(((u, (0.0, 0.5, 0.5)), (v, (0.5, 0.0, 0.5)), (w, (0.5, 0.5, 0.0)))):
            self.launch(flip.enforce_walls, threads, [blocks, field, axis, self.walls])
            if self.solid_host is not None:
                self.launch(flip.enforce_solids, threads, [blocks, field, axis, self.solid])
            for index, inflow in enumerate(self.inflows):
                if speeds[index] <= 0.0:
                    continue
                self.launch(
                    flip.set_opening_velocity,
                    threads,
                    [
                        blocks, field, axis, self.cell, wp.vec3(*face_offset), inflow.gpu_centre, inflow.gpu_direction, inflow.gpu_across, inflow.gpu_up_across,
                        inflow.shape_code, inflow.half_across, inflow.half_up, speeds[index], options.cone, options.turbulence, self.inflow_phase(inflow, seconds),
                    ],
                )
        if self.viscous_source is not None:
            self.diffuse(delta_time)
        if options.floor_friction > 0.0:
            keep = math.exp(-options.floor_friction * delta_time)
            self.launch(flip.damp_floor_layer, threads, [blocks, u, keep])
            self.launch(flip.damp_floor_layer, threads, [blocks, w, keep])
        self.launch(flip.compute_divergence, threads, [blocks, u, v, w, self.cell_count, self.solid, inverse_cell, self.divergence, self.pressure, float(options.particles_per_cell), options.separation / delta_time])
        self.project()
        self.launch(flip.apply_pressure_u, threads, [blocks, u, self.pressure, self.cell_count, self.solid, inverse_cell, self.walls])
        self.launch(flip.apply_pressure_v, threads, [blocks, v, self.pressure, self.cell_count, self.solid, inverse_cell, self.walls])
        self.launch(flip.apply_pressure_w, threads, [blocks, w, self.pressure, self.cell_count, self.solid, inverse_cell, self.walls])
        self.extend()
        if self.solid_host is not None:
            for axis, field in enumerate((u, v, w)):
                self.launch(flip.enforce_solids, threads, [blocks, field, axis, self.solid])
        self.launch(
            flip.grid_to_particles,
            self.particle_count,
            [self.positions, self.velocities, self.alive, u, v, w, self.old["u"], self.old["v"], self.old["w"], self.solid, inverse_cell, options.flip, delta_time, self.domain, 0.004, self.walls],
        )
        self.drain_outflow(self.positions, self.particle_count, self.alive, self.ww_life, 0)
        wp.synchronize_device(self.device)
        middle = time.perf_counter()
        self.timers["sim"] += middle - started
        if not options.no_whitewater:
            self.update_whitewater(delta_time)
            self.drain_outflow(self.ww_positions, self.ww_positions.shape[0], self.alive, self.ww_life, 1)
        wp.synchronize_device(self.device)
        self.timers["whitewater"] += time.perf_counter() - middle
        self.step_index += 1

    def project(self):
        """Solve for the pressure that makes the velocity divergence free."""
        options = self.options
        if self.cg is not None:
            self.cg.solve(self.active.cells, self.active.cell_count, self.pressure, self.divergence, self.cell_count, self.solid, self.cell, self.walls, options.iterations)
            return
        for _ in range(options.iterations):
            for parity in (0, 1):
                self.launch(flip.relax_pressure, self.active.cell_count, [self.active.cells, self.pressure, self.divergence, self.cell_count, self.solid, self.cell, options.omega, parity, self.walls])

    def update_whitewater(self, delta_time):
        options = self.options
        inverse_cell = 1.0 / self.cell
        u, v, w = self.velocity["u"], self.velocity["v"], self.velocity["w"]
        for field in (self.density, self.shear, *self.normals):
            field.zero_()
        threads = self.active.thread_count()
        blocks = self.active.blocks
        self.launch(whitewater.cell_fields, threads, [blocks, u, v, w, self.cell_count, inverse_cell, float(options.particles_per_cell), self.density, self.shear])
        self.launch(whitewater.density_normals, threads, [blocks, self.density, self.normals[0], self.normals[1], self.normals[2]])
        self.launch(
            whitewater.emit_whitewater,
            self.particle_count,
            [
                self.positions, self.velocities, self.alive, self.density, self.shear, self.normals[0], self.normals[1], self.normals[2],
                inverse_cell, self.cell, options.ww_trapped_rate, options.ww_crest_rate,
                wp.vec2(0.5, 6.0), wp.vec2(1.0, 6.0), wp.vec2(4.0, 80.0), wp.vec2(options.ww_life[0], options.ww_life[1]),
                delta_time, self.step_index, options.seed,
                self.ww_positions, self.ww_velocities, self.ww_life, self.ww_ids, self.ww_counter,
            ],
        )
        self.launch(
            whitewater.advect_whitewater,
            self.ww_positions.shape[0],
            [
                self.ww_positions, self.ww_velocities, self.ww_life, self.density, self.solid, u, v, w, inverse_cell, delta_time,
                0.12, 0.55, options.ww_buoyancy, options.ww_drag, self.domain, self.walls,
            ],
        )

    def export_whitewater(self):
        if self.options.no_whitewater:
            empty = np.zeros((0, 3), np.float32)
            return empty, empty, np.zeros(0, np.int64), np.zeros(0, np.uint8), 0
        life = self.ww_life.numpy()
        alive_count = int((life > 0.0).sum())
        keep = min(1.0, self.options.ww_export / max(alive_count, 1) * 0.98)
        self.export_counter.zero_()
        self.launch(
            whitewater.select_for_export,
            self.ww_positions.shape[0],
            [
                self.ww_positions, self.ww_velocities, self.ww_life, self.ww_ids, self.density, 1.0 / self.cell, 0.12, 0.55, keep,
                self.export_positions, self.export_velocities, self.export_ids, self.export_kinds, self.export_counter,
            ],
        )
        count = min(int(self.export_counter.numpy()[0]), self.options.ww_export)
        ids = self.export_ids.numpy()[:count]
        order = np.argsort(ids, kind="stable")
        return self.export_positions.numpy()[:count][order], self.export_velocities.numpy()[:count][order], ids[order], self.export_kinds.numpy()[:count][order], alive_count

    def snapshot(self, offset, limit=60000):
        """Liquid particle positions (relative to the opening, metres) and speeds, thinned to `limit`, for the preview."""
        alive = self.alive.numpy()[: self.particle_count] == 1
        positions = self.positions.numpy()[: self.particle_count][alive] - offset
        speeds = np.linalg.norm(self.velocities.numpy()[: self.particle_count][alive], axis=1)
        step = max(1, len(positions) // limit)
        return positions[::step], speeds[::step]

    def head_height(self):
        alive = self.alive[: self.particle_count].numpy() == 1
        heights = self.positions[: self.particle_count].numpy()[alive][:, 1]
        return float(np.percentile(heights, 99.8)) if len(heights) else 0.0

    def residual(self):
        self.worst_divergence.zero_()
        self.launch(flip.residual_divergence, self.active.cell_count, [self.active.cells, self.velocity["u"], self.velocity["v"], self.velocity["w"], self.cell_count, 1.0 / self.cell, self.worst_divergence])
        return float(self.worst_divergence.numpy()[0])


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    options = parse_arguments()
    if options.inflow_position is not None and len(options.inflow_position) not in (2, 3):
        raise SystemExit("--inflow-position takes X Z (floor, or side with --inflow-height) or X Y Z (side)")
    wp.config.quiet = True
    wp.init()
    baseline_mib = video_memory_used_mib()
    peak_mib = baseline_mib
    simulation = Simulation(options)
    writer = SequenceWriter(Path(options.out), options.name, options.fps)
    # the fastest of the inflows and a free fall from the top of the domain
    reference_velocity = max(max(inflow.speed for inflow in simulation.inflows) * 1.1, math.sqrt(2.0 * flip.GRAVITY * options.height))
    steps_per_frame = max(1, math.ceil((1.0 / options.fps) / (options.cfl * options.cell / reference_velocity)))
    delta_time = 1.0 / options.fps / steps_per_frame
    first_centre = simulation.inflows[0].centre
    offset = np.array([first_centre[0], 0.0, first_centre[2]], np.float32)
    residuals = []
    snapshots = []
    snapshot_frames = {options.frames // 3, 2 * options.frames // 3, options.frames - 1}
    print(f"grid {simulation.size} cells of {options.cell} m, {steps_per_frame} steps per frame, dt {delta_time * 1000:.2f} ms, iso {simulation.iso_threshold:.3f}", flush=True)
    wall_start = time.perf_counter()
    skipped_frames = int(round(options.skip_seconds * options.fps))
    for frame in range(options.frames):
        for substep in range(steps_per_frame):
            simulation.step((frame + substep / steps_per_frame) / options.fps, delta_time)
        print(f"frame {frame + 1}/{options.frames}", flush=True)
        if frame < skipped_frames:
            continue
        started = time.perf_counter()
        vertices, normals, indices = simulation.surface.extract(simulation.positions[: simulation.particle_count], simulation.alive[: simulation.particle_count], simulation.iso_threshold)
        wp.synchronize_device(options.device)
        if simulation.solid_host is not None and len(vertices):
            vertices = push_out_of_solid(vertices, normals, simulation.solid_host, options.cell)
        simulation.timers["mesh"] += time.perf_counter() - started
        started = time.perf_counter()
        spray_positions, spray_velocities, spray_ids, spray_kinds, ww_alive = simulation.export_whitewater()
        head = simulation.head_height()
        writer.add_frame(vertices - offset, normals, indices, spray_positions - offset, spray_velocities, spray_ids, head, spray_kinds)
        simulation.timers["export"] += time.perf_counter() - started
        if frame in snapshot_frames:
            snapshots.append(simulation.snapshot(offset))
        if frame % 30 == 0:
            peak_mib = max(peak_mib, video_memory_used_mib())
        residual = simulation.residual()
        residuals.append(residual)
        if frame % 10 == 0 or frame == options.frames - 1:
            print(f"frame {frame} liquid {simulation.particle_count} whitewater alive {ww_alive} exported {len(spray_positions)} triangles {len(indices) // 3} head {head:.2f} m residual {residual:.3f} /s elapsed {time.perf_counter() - wall_start:.0f} s", flush=True)
    if options.preview:
        domain_min = -offset
        domain_max = np.array(simulation.size, np.float32) * options.cell - offset
        preview.save_preview(Path(options.preview), snapshots, domain_min, domain_max, simulation.walls, options.fps, options.frames)
    started = time.perf_counter()
    settings = {key: value for key, value in vars(options).items() if key != "out"}
    settings["resolved_inflows"] = [inflow.settings() for inflow in simulation.inflows]
    meta = writer.finish(settings, origin=offset)
    simulation.timers["export"] += time.perf_counter() - started
    report = {
        "timers_seconds": {key: round(value, 1) for key, value in simulation.timers.items()},
        "total_seconds": round(time.perf_counter() - wall_start, 1),
        "vram_peak_mib_total": peak_mib,
        "vram_baseline_mib_other_apps": baseline_mib,
        "bytes": meta["bytes"],
        "peak_head_m": max(meta["headHeights"]),
        "liquid_particles": simulation.particle_count,
        "pressure": options.pressure,
        "residual_mean_per_s": round(float(np.mean(residuals)), 4) if residuals else 0.0,
        "residual_max_per_s": round(float(np.max(residuals)), 4) if residuals else 0.0,
    }
    print(json.dumps(report), flush=True)
    Path(options.out, "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
