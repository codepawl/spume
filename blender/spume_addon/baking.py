# SPDX-License-Identifier: Apache-2.0
"""Runs the Spume solver in its own process. The UI never waits: output goes to a log file and a timer polls it."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from . import prefs, sequence, voxelize

_job = {}
CACHE_FILES = ("frame_*.glb", "spray.bin", "spray_ids.bin", "spray_kinds.bin", "meta.json", "report.json", "inflows.json", "obstacle.npy")
PROGRESS_LINE = re.compile(r"frame (\d+)/(\d+)")


def cache_root(domain):
    base = domain.spume.cache_folder
    if base.startswith("//") and not bpy.data.filepath:
        return Path(tempfile.gettempdir()) / "spume_cache" / domain.name
    return Path(bpy.path.abspath(base)) / domain.name


def cache_folder(domain, preview):
    return cache_root(domain) / ("preview" if preview else "full")


def objects_with_role(scene, role):
    return sorted((obj for obj in scene.objects if obj.spume.role == role), key=lambda obj: obj.name)


def inflow_entry(inflow, to_solver, cell, length, width):
    """One Inflow object as a solver inflow: its real position and direction in solver metres."""
    position = to_solver @ inflow.matrix_world.translation
    direction = (to_solver.to_3x3() @ Vector(inflow.spume.direction)).normalized()
    pulse = inflow.spume
    entry = {
        "direction": [round(value, 5) for value in direction],
        "shape": pulse.opening_shape,
        "size": round(pulse.opening_diameter, 4) if pulse.opening_shape == "circle" else [round(pulse.opening_size[0], 4), round(pulse.opening_size[1], 4)],
        "speed": pulse.speed, "rise": pulse.rise, "feed": pulse.feed, "taper": pulse.taper,
    }
    if direction.y > 0.7 and position.y < 2 * cell:
        entry.update(mode="floor", position=[round(position.x, 4), round(position.z, 4)])
    else:
        entry.update(mode="side", position=[round(position.x, 4), round(position.y, 4), round(position.z, 4)])
    return entry


def core_arguments(context, domain, preview, folder):
    """Translate the scene into solver options and write the files they point at. Returns (arguments, warnings)."""
    scene = context.scene
    settings = domain.spume
    inflows = objects_with_role(scene, "INFLOW")
    if not inflows:
        raise ValueError("Add an object and set its Spume type to Inflow.")
    warnings = []
    (length, height, width), matrix = sequence.domain_layout(domain)
    scale = settings.preview_scale if preview else 1.0
    cell = settings.cell_size * scale
    mesh_cell = settings.mesh_cell_size * scale
    frames = min(settings.frames, settings.preview_frames) if preview else settings.frames
    to_solver = matrix.inverted()

    folder.mkdir(parents=True, exist_ok=True)
    entries = [inflow_entry(inflow, to_solver, cell, length, width) for inflow in inflows]
    (folder / "inflows.json").write_text(json.dumps(entries, indent=1))
    arguments = [
        "--out", str(folder), "--name", domain.name, "--fps", str(scene.render.fps),
        "--cell", f"{cell:.5f}", "--mesh-cell", f"{mesh_cell:.5f}", "--frames", str(frames),
        "--length", f"{length:.4f}", "--width", f"{width:.4f}", "--height", f"{height:.4f}",
        "--inflows", str(folder / "inflows.json"),
    ]
    if settings.use_whitewater and settings.whitewater_amount > 0.0:
        amount = settings.whitewater_amount
        arguments += ["--ww-trapped-rate", f"{60.0 * amount:.3f}", "--ww-crest-rate", f"{120.0 * amount:.3f}", "--ww-export", str(max(2, int(22000 * amount)))]
    else:
        arguments += ["--no-whitewater"]
    if preview:
        arguments += ["--iterations", "40"]

    size = tuple(int(round(extent / cell)) for extent in (length, height, width))
    obstacles = [obj for obj in objects_with_role(scene, "OBSTACLE") if obj.type == "MESH"]
    if obstacles:
        mask = voxelize.obstacle_mask(obstacles, context.evaluated_depsgraph_get(), to_solver, size, cell)
        if mask is None:
            warnings.append("No Obstacle object overlaps the domain.")
        else:
            np.save(folder / "obstacle.npy", mask)
            arguments += ["--obstacle-mask", str(folder / "obstacle.npy")]
    for outflow in objects_with_role(scene, "OUTFLOW"):
        for box in voxelize.outflow_boxes([outflow], to_solver):
            arguments += ["--outflow-box", *(f"{value:.4f}" for value in box)]
    return arguments, warnings


def start(context, domain, preview):
    if _job:
        raise ValueError("A Spume bake is already running.")
    command, working_folder = prefs.solver_command(context)
    folder = cache_folder(domain, preview)
    clear_folder(folder)
    arguments, warnings = core_arguments(context, domain, preview, folder)
    log = open(folder / "bake.log", "w", encoding="utf-8")
    flags = (subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    process = subprocess.Popen(command + arguments, cwd=working_folder, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, start_new_session=os.name != "nt")
    frames = int(arguments[arguments.index("--frames") + 1])
    _job.update(process=process, log=log, domain=domain.name, folder=folder, frames=frames, preview=preview)
    domain.spume.bake_progress = 0.0
    domain.spume.bake_message = "Starting the solver"
    return warnings


def cancel():
    process = _job.get("process")
    if process and process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
        else:
            os.killpg(process.pid, 15)
        process.wait()
    _job["cancelled"] = True


def is_running():
    return bool(_job)


def logged_progress(folder):
    """(frames done, frames total) from the last `frame i/n` line of the solver log, or None before the first one."""
    log = Path(folder, "bake.log")
    if not log.exists():
        return None
    with open(log, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 4096))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = PROGRESS_LINE.findall(tail)
    return (int(matches[-1][0]), int(matches[-1][1])) if matches else None


def last_log_line(folder):
    lines = Path(folder, "bake.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


def poll():
    """Timer callback. Returns the seconds until the next call, or None when the job is over."""
    if not _job:
        return None
    domain = bpy.data.objects.get(_job["domain"])
    folder = _job["folder"]
    process = _job["process"]
    progress = logged_progress(folder)
    if domain is not None and progress:
        done, total = progress
        domain.spume.bake_progress = min(1.0, done / max(total, 1))
        domain.spume.bake_message = f"Frame {done} of {total}"
    code = process.poll()
    if bpy.context.window_manager:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    if code is None and not _job.get("cancelled"):
        return 0.5
    _job["log"].close()
    if _job.get("cancelled"):
        message = "Cancelled"
    elif code != 0:
        message = f"Solver failed ({code}): {last_log_line(folder)}"
    else:
        message = f"Baked {_job['frames']} frames"
        if domain is not None:
            sequence.load(domain, str(folder))
    if domain is not None:
        domain.spume.bake_message = message
        if code == 0 and not _job.get("cancelled"):
            domain.spume.bake_progress = 1.0
    _job.clear()
    return None


def clear_folder(folder, include_log=False):
    if Path(folder).exists():
        for pattern in CACHE_FILES + (("bake.log",) if include_log else ()):
            for path in Path(folder).glob(pattern):
                path.unlink()


def free(domain):
    sequence.free_objects(domain)
    root = cache_root(domain)
    for name in ("preview", "full"):
        clear_folder(root / name, include_log=True)
        if (root / name).exists() and not any((root / name).iterdir()):
            shutil.rmtree(root / name)
    domain.spume.bake_progress = 0.0
    domain.spume.bake_message = ""
