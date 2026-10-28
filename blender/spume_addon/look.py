# SPDX-License-Identifier: Apache-2.0
"""Default water and whitewater materials and the Geometry Nodes group that draws whitewater points as small spheres."""

import bpy


def water_material():
    material = bpy.data.materials.get("Spume Water")
    if material:
        return material
    material = bpy.data.materials.new("Spume Water")
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.55, 0.82, 0.95, 1.0)
    shader.inputs["Roughness"].default_value = 0.03
    shader.inputs["IOR"].default_value = 1.33
    shader.inputs["Transmission Weight"].default_value = 1.0
    if hasattr(material, "use_raytrace_refraction"):
        material.use_raytrace_refraction = True
    return material


def foam_material():
    material = bpy.data.materials.get("Spume Whitewater")
    if material:
        return material
    material = bpy.data.materials.new("Spume Whitewater")
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.96, 0.98, 1.0, 1.0)
    shader.inputs["Roughness"].default_value = 0.45
    shader.inputs["Emission Color"].default_value = (0.9, 0.95, 1.0, 1.0)
    shader.inputs["Emission Strength"].default_value = 0.08
    return material


def whitewater_nodes():
    """Mesh points -> small ico spheres sized by the `radius` point attribute."""
    name = "Spume Whitewater Spheres"
    existing = bpy.data.node_groups.get(name)
    if existing:
        return existing
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    sphere = nodes.new("GeometryNodeMeshIcoSphere")
    sphere.inputs["Radius"].default_value = 1.0
    sphere.inputs["Subdivisions"].default_value = 1
    radius = nodes.new("GeometryNodeInputNamedAttribute")
    radius.data_type = "FLOAT"
    radius.inputs["Name"].default_value = "radius"
    scale = nodes.new("ShaderNodeCombineXYZ")
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = foam_material()
    links.new(group_input.outputs["Geometry"], instance.inputs["Points"])
    links.new(sphere.outputs["Mesh"], instance.inputs["Instance"])
    for axis in ("X", "Y", "Z"):
        links.new(radius.outputs["Attribute"], scale.inputs[axis])
    links.new(scale.outputs["Vector"], instance.inputs["Scale"])
    links.new(instance.outputs["Instances"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree
