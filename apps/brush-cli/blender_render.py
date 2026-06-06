#!/usr/bin/env python3
"""Unlit mesh render via Blender CLI.

Invoked as:
  blender --background --factory-startup --python blender_render.py -- \\
    --ply <path> --out <png> --w <int> --h <int> \\
    --pos x,y,z --target x,y,z --up x,y,z --fov-y <deg>

The output is a flat (emission-shaded) rendering of the mesh's per-vertex
colors — no lighting, no ambient occlusion, no tone mapping — so the
rendered RGB is the interpolated vertex color directly. Designed to be
a faithful, fast input for PSNR comparison against the dataset GT
images.

The brush camera convention is +X right, +Y down, +Z forward in view
space, with world-up = +Y. Blender's camera looks down its local -Z with
local +Y as the camera-up. The argv `--up` is the world-space direction
that should map to the camera's up (i.e. brush's `rotation * (-Y_view)`).

The mesh is loaded as PLY (with vertex colors). A material is built with
a Color Attribute node feeding an Emission shader at strength 1, output
to material output surface. That gives true unlit shading regardless of
world lighting.
"""

import argparse
import math
import sys

import bpy  # type: ignore
import bmesh  # type: ignore
import mathutils  # type: ignore


def parse_vec3(s: str) -> "mathutils.Vector":
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"expected x,y,z, got {s!r}")
    return mathutils.Vector([float(p) for p in parts])


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--w", type=int, required=True)
    p.add_argument("--h", type=int, required=True)
    p.add_argument("--pos", required=True, help="x,y,z world-space camera position")
    p.add_argument(
        "--target", required=True, help="x,y,z world-space focal point"
    )
    p.add_argument(
        "--up", required=True, help="x,y,z world direction that maps to camera up"
    )
    p.add_argument(
        "--fov-y", type=float, required=True, help="vertical field of view, degrees"
    )
    return p.parse_args(argv)


def setup_scene(args):
    # Start from an empty scene (factory-startup gave us the default cube etc).
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Eevee renderer — fast and supports emission shading well.
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {
        e.identifier for e in scene.render.bl_rna.properties["engine"].enum_items
    } else "BLENDER_EEVEE"
    scene.render.resolution_x = args.w
    scene.render.resolution_y = args.h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = True  # alpha = 0 background → PSNR-friendly
    # Disable any post-processing / tone-mapping that would distort vertex
    # colours. View transform "Standard" + look "None" gives an as-is
    # write of linear sRGB to the output PNG.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def import_ply(path: str):
    # Modern blender (>=4.x) ships the PLY importer as `bpy.ops.wm.ply_import`.
    bpy.ops.wm.ply_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    return obj


def make_unlit_material(obj):
    """Build an Emission(strength=1) material driven by the PLY's per-
    vertex Color Attribute. Returns the material; caller assigns it.
    """
    mat = bpy.data.materials.new(name="UnlitVertexColor")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out_node = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    color_attr = nodes.new("ShaderNodeVertexColor")
    # Blender's PLY import names the vertex color attribute "Col"
    # (legacy) or "Color"; pick whichever exists on this mesh.
    me = obj.data
    if me.color_attributes:
        color_attr.layer_name = me.color_attributes.active_color.name
    links.new(color_attr.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out_node.inputs["Surface"])
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    return mat


def setup_camera(args):
    """Create a camera at the requested world-space pose with the
    requested vertical FoV. Blender's camera looks down its local -Z;
    local +Y is camera-up. Brush's view frame has +X right, +Y down,
    +Z forward (so brush's camera-up in world = brush.rot * (-Y_view)).
    To make a Blender camera that matches a brush camera, we point its
    -Z at the focal point with its +Y in the brush-up direction.
    """
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens_unit = "FOV"
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(args.fov_y)
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    pos = parse_vec3(args.pos)
    target = parse_vec3(args.target)
    up = parse_vec3(args.up).normalized()

    # Build a rotation that points -Z towards (target-pos) and aligns +Y with up.
    forward_world = (target - pos).normalized()  # direction camera looks at
    # Camera's -Z = forward_world  ⇒  +Z = -forward_world
    z_axis = -forward_world
    # Camera's +Y = up (after orthonormalising against z)
    y_axis = (up - up.dot(z_axis) * z_axis).normalized()
    x_axis = y_axis.cross(z_axis).normalized()
    # Compose a 3×3 rotation matrix whose columns are the world-space
    # camera basis vectors, then build the transform.
    rot = mathutils.Matrix(
        [
            [x_axis.x, y_axis.x, z_axis.x],
            [x_axis.y, y_axis.y, z_axis.y],
            [x_axis.z, y_axis.z, z_axis.z],
        ]
    )
    cam_obj.location = pos
    cam_obj.rotation_euler = rot.to_euler()


def main():
    args = parse_args()
    setup_scene(args)
    obj = import_ply(args.ply)
    make_unlit_material(obj)
    setup_camera(args)
    bpy.context.scene.render.filepath = args.out
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
