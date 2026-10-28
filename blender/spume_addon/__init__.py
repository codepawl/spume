# SPDX-License-Identifier: Apache-2.0
bl_info = {
    "name": "Spume",
    "author": "CodePawl",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "Properties > Physics > Spume",
    "description": "GPU liquid simulation (FLIP, whitewater, marching cubes) baked by the Spume solver",
    "doc_url": "https://github.com/codepawl/spume",
    "category": "Physics",
}

import bpy

from . import prefs, properties, sequence, ui


def register():
    properties.register()
    bpy.utils.register_class(prefs.SpumePreferences)
    for cls in ui.CLASSES:
        bpy.utils.register_class(cls)
    sequence.register()


def unregister():
    sequence.unregister()
    for cls in reversed(ui.CLASSES):
        bpy.utils.unregister_class(cls)
    bpy.utils.unregister_class(prefs.SpumePreferences)
    properties.unregister()
