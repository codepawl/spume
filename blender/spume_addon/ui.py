# SPDX-License-Identifier: Apache-2.0
import bpy

from . import baking, sequence


class SpumeOperator(bpy.types.Operator):
    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.spume.role == "DOMAIN"


class SPUME_OT_bake(SpumeOperator):
    bl_idname = "spume.bake"
    bl_label = "Bake"
    bl_description = "Run the Spume solver in a separate process and load the result"

    preview: bpy.props.BoolProperty(name="Preview", description="Low resolution, fewer frames", default=False)

    def execute(self, context):
        try:
            warnings = baking.start(context, context.object, self.preview)
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        for warning in warnings:
            self.report({"WARNING"}, warning)
        bpy.app.timers.register(baking.poll, first_interval=0.5)
        return {"FINISHED"}


class SPUME_OT_cancel(bpy.types.Operator):
    bl_idname = "spume.cancel"
    bl_label = "Cancel"

    def execute(self, context):
        baking.cancel()
        return {"FINISHED"}


class SPUME_OT_free(SpumeOperator):
    bl_idname = "spume.free"
    bl_label = "Free"
    bl_description = "Delete the baked cache and the liquid and whitewater objects"

    def execute(self, context):
        baking.free(context.object)
        return {"FINISHED"}


class SPUME_OT_load(SpumeOperator):
    bl_idname = "spume.load"
    bl_label = "Load Cache"
    bl_description = "Load the last full bake, or the preview bake when there is no full one"

    def execute(self, context):
        for preview in (False, True):
            folder = baking.cache_folder(context.object, preview)
            if (folder / "meta.json").exists():
                sequence.load(context.object, str(folder))
                return {"FINISHED"}
        self.report({"ERROR"}, "No baked cache found")
        return {"CANCELLED"}


class SPUME_PT_physics(bpy.types.Panel):
    bl_label = "Spume"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def draw(self, context):
        settings = context.object.spume
        self.layout.use_property_split = True
        self.layout.prop(settings, "role")
        if settings.role == "INFLOW":
            self.draw_inflow(settings)
        elif settings.role == "OBSTACLE":
            self.layout.label(text="Solid for the liquid (closed mesh)", icon="MOD_PHYSICS")
        elif settings.role == "OUTFLOW":
            self.layout.label(text="Deletes liquid inside its bounds", icon="TRASH")

    def draw_inflow(self, settings):
        layout = self.layout
        layout.prop(settings, "speed")
        layout.prop(settings, "direction")
        layout.prop(settings, "opening_shape")
        layout.prop(settings, "opening_diameter" if settings.opening_shape == "circle" else "opening_size")
        column = layout.column(align=True)
        column.prop(settings, "rise")
        column.prop(settings, "feed")
        column.prop(settings, "taper")


class SPUME_PT_domain(bpy.types.Panel):
    bl_label = "Domain"
    bl_parent_id = "SPUME_PT_physics"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.spume.role == "DOMAIN"

    def draw(self, context):
        settings = context.object.spume
        layout = self.layout
        layout.use_property_split = True
        running = baking.is_running()
        if running:
            layout.progress(factor=settings.bake_progress, text=settings.bake_message)
            layout.operator("spume.cancel", icon="CANCEL")
        else:
            row = layout.row(align=True)
            row.operator("spume.bake", text="Bake", icon="PLAY").preview = False
            row.operator("spume.bake", text="Bake Preview", icon="FF").preview = True
            row = layout.row(align=True)
            row.operator("spume.load", icon="FILE_REFRESH")
            row.operator("spume.free", icon="TRASH")
            if settings.bake_message:
                layout.label(text=settings.bake_message)
        column = layout.column(align=True)
        column.prop(settings, "cell_size")
        column.prop(settings, "mesh_cell_size")
        layout.prop(settings, "frames")
        layout.prop(settings, "start_frame")
        layout.prop(settings, "use_whitewater")
        if settings.use_whitewater:
            layout.prop(settings, "whitewater_amount")
        layout.prop(settings, "cache_folder")
        column = layout.column(align=True)
        column.prop(settings, "preview_scale")
        column.prop(settings, "preview_frames")


CLASSES = (SPUME_OT_bake, SPUME_OT_cancel, SPUME_OT_free, SPUME_OT_load, SPUME_PT_physics, SPUME_PT_domain)
