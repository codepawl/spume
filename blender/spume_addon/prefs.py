# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path

import bpy
from bpy.props import StringProperty


def source_tree_core():
    """When the addon runs from a checkout (blender/spume_addon) the solver is two folders up."""
    candidate = Path(__file__).resolve().parents[2]
    return str(candidate) if (candidate / "run.py").exists() else ""


class SpumePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    core_folder: StringProperty(name="Solver Folder", description="The Spume folder that holds run.py and pyproject.toml", subtype="DIR_PATH", default=source_tree_core())
    uv_path: StringProperty(name="uv", description="uv executable. Empty uses the one on PATH. Ignored when a Python path is set", subtype="FILE_PATH")
    python_path: StringProperty(name="Python", description="Python with the solver's requirements installed (warp-lang, numpy, psutil). Empty runs the solver through uv", subtype="FILE_PATH")

    def draw(self, context):
        for name in ("core_folder", "uv_path", "python_path"):
            self.layout.prop(self, name)


def get(context):
    return context.preferences.addons[__package__].preferences


def solver_command(context):
    """Returns (command prefix, working folder) or raises ValueError with something the user can fix."""
    preferences = get(context)
    core = Path(bpy.path.abspath(preferences.core_folder))
    if not (core / "run.py").exists():
        raise ValueError("Set the Solver Folder in the Spume add-on preferences (the folder with run.py).")
    if preferences.python_path:
        return [bpy.path.abspath(preferences.python_path), str(core / "run.py")], core
    uv = bpy.path.abspath(preferences.uv_path) if preferences.uv_path else shutil.which("uv")
    if not uv:
        raise ValueError("uv was not found. Install it or set a uv or Python path in the Spume add-on preferences.")
    return [uv, "run", "--project", str(core), "python", str(core / "run.py")], core
