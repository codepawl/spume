# SPDX-License-Identifier: Apache-2.0
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty


class SpumeSettings(bpy.types.PropertyGroup):
    role: EnumProperty(
        name="Type",
        items=[
            ("NONE", "None", "Not part of a Spume simulation"),
            ("DOMAIN", "Domain", "The object's bounds become the simulation grid"),
            ("INFLOW", "Inflow", "An opening that pushes liquid into the domain"),
            ("OBSTACLE", "Obstacle", "A static collider: its mesh is voxelized into solid cells. Needs a closed mesh"),
            ("OUTFLOW", "Outflow", "A region that deletes liquid and whitewater: the bounding box of the object"),
        ],
        default="NONE",
    )

    # Domain
    cell_size: FloatProperty(name="Cell Size", description="Simulation cell in metres", default=0.08, min=0.01, max=2.0, unit="LENGTH")
    mesh_cell_size: FloatProperty(name="Mesh Cell Size", description="Marching cubes cell in metres", default=0.1, min=0.01, max=2.0, unit="LENGTH")
    frames: IntProperty(name="Frames", default=180, min=1)
    start_frame: IntProperty(name="Start Frame", description="Scene frame that shows the first cached frame", default=1)
    use_whitewater: BoolProperty(name="Whitewater", description="Spray, foam and bubble points", default=True)
    whitewater_amount: FloatProperty(name="Amount", description="Scales how much whitewater is emitted and kept", default=1.0, min=0.0, soft_max=4.0)
    cache_folder: StringProperty(name="Cache", subtype="DIR_PATH", default="//spume_cache/")
    preview_scale: FloatProperty(name="Preview Cell Scale", description="Bake Preview multiplies the cell sizes by this", default=2.5, min=1.0, max=8.0)
    preview_frames: IntProperty(name="Preview Frames", default=40, min=1)
    loaded_folder: StringProperty(name="Loaded Cache", description="Cache folder the sequence objects currently read", default="")
    bake_progress: FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0, subtype="FACTOR")
    bake_message: StringProperty(name="Status", default="")

    # Inflow
    speed: FloatProperty(name="Speed", description="Peak speed at the opening, m/s", default=18.5, min=0.0, unit="VELOCITY")
    direction: FloatVectorProperty(name="Direction", description="Flow direction in world space", subtype="DIRECTION", default=(0.0, 0.0, 1.0), size=3)
    opening_shape: EnumProperty(name="Shape", items=[("circle", "Circle", ""), ("rectangle", "Rectangle", "")], default="circle")
    opening_diameter: FloatProperty(name="Diameter", default=0.6, min=0.01, unit="LENGTH")
    opening_size: FloatVectorProperty(name="Size", description="Width across the flow and height", default=(1.0, 1.0), size=2, min=0.01, unit="LENGTH")
    rise: FloatProperty(name="Rise", description="Seconds for the pulse to reach full speed", default=0.2, min=0.01, unit="TIME")
    feed: FloatProperty(name="Hold", description="Seconds the flow is held at full speed", default=2.2, min=0.0, unit="TIME")
    taper: FloatProperty(name="Taper", description="Seconds the flow takes to die away after the hold", default=1.0, min=0.01, unit="TIME")


def register():
    bpy.utils.register_class(SpumeSettings)
    bpy.types.Object.spume = PointerProperty(type=SpumeSettings)


def unregister():
    del bpy.types.Object.spume
    bpy.utils.unregister_class(SpumeSettings)
