import bpy

from mathutils import Vector


def project_3d_point(camera: bpy.types.Object,
                     p: Vector,
                     render: bpy.types.RenderSettings = bpy.context.scene.render) -> Vector:
    """
    Given a camera and its projection matrix M;
    given p, a 3d point to project:

    Compute P’ = M * P
    P’= (x’, y’, z’, w')

    Ignore z'
    Normalize in:
    x’’ = x’ / w’
    y’’ = y’ / w’

    x’’ is the screen coordinate in normalised range -1 (left) +1 (right)
    y’’ is the screen coordinate in  normalised range -1 (bottom) +1 (top)

    :param camera: The camera for which we want the projection
    :param p: The 3D point to project
    :param render: The render settings associated to the scene.
    :return: The 2D projected point in normalized range [-1, 1] (left to right, bottom to top)
    """

    if camera.type != 'CAMERA':
        raise Exception("Object {} is not a camera.".format(camera.name))

    if len(p) != 3:
        raise Exception("Vector {} is not three-dimensional".format(p))

    # Get the two components to calculate M
    modelview_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        bpy.data.scenes["Scene"].view_layers["View Layer"].depsgraph,
        x = render.resolution_x,
        y = render.resolution_y,
        scale_x = render.pixel_aspect_x,
        scale_y = render.pixel_aspect_y,
    )

    # print(projection_matrix * modelview_matrix)

    # Compute P’ = M * P
    p1 = projection_matrix @ modelview_matrix @ Vector((p.x, p.y, p.z, 1))

    # Normalize in: x’’ = x’ / w’, y’’ = y’ / w’
    p2 = Vector(((p1.x/p1.w, p1.y/p1.w)))

    return p2

import numpy as np

def Get_Vertices_From_Object(name):
    return [bpy.data.objects[name].matrix_world @ vertex_i.co for vertex_i in bpy.data.objects[name].data.vertices]

def Get_Min_Max(vertices):
    min_vec3 = np.array([vertices[0, 0], vertices[0, 1], vertices[0, 2]], dtype=np.float32)
    max_vec3 = min_vec3.copy()
    
    for _, verts_i in enumerate(vertices[1::]):
        for j, verts_ij in enumerate(verts_i):
            if verts_ij < min_vec3[j]:
                min_vec3[j] = verts_ij

            if verts_ij > max_vec3[j]:
                max_vec3[j] = verts_ij
                
    return (min_vec3, max_vec3)

camera = bpy.data.objects['Camera']  # or bpy.context.active_object
render = bpy.context.scene.render

P = Vector((-0.002170146, 0.409979939, 0.162410125))

verts = Get_Vertices_From_Object('Cube')
(min, max) = Get_Min_Max(np.array(verts))

# X
bpy.data.objects['Sphere_Min_X'].location = [min[0], (max[1] + min[1])/2.0, max[2]]
bpy.data.objects['Sphere_Max_X'].location = [max[0], (max[1] + min[1])/2.0, max[2]]
# Y
bpy.data.objects['Sphere_Min_Y'].location = [(max[0] + min[0])/2.0, min[1], max[2]]
bpy.data.objects['Sphere_Max_Y'].location = [(max[0] + min[0])/2.0, max[1], max[2]]

P1 = Vector((min[0], (max[1] + min[1])/2.0, max[2]))
P2 = Vector((max[0], (max[1] + min[1])/2.0, max[2]))
P3 = Vector(((max[0] + min[0])/2.0, min[1], max[2]))
P4 = Vector(((max[0] + min[0])/2.0, max[1], max[2]))

proj_p = project_3d_point(camera=camera, p=P1, render=render)
print(proj_p)
proj_p = project_3d_point(camera=camera, p=P2, render=render)
print(proj_p)
proj_p = project_3d_point(camera=camera, p=P3, render=render)
print(proj_p)
proj_p = project_3d_point(camera=camera, p=P4, render=render)
print(proj_p)

"""
print("Projecting point {} for camera '{:s}' into resolution {:d}x{:d}..."
      .format(P, camera.name, render.resolution_x, render.resolution_y))

proj_p = project_3d_point(camera=camera, p=P, render=render)
print("Projected point (homogeneous coords): {}.".format(proj_p))

proj_p_pixels = Vector(((render.resolution_x-1) * (proj_p.x + 1) / 2, (render.resolution_y - 1) * (proj_p.y - 1) / (-2)))
print("Projected point (pixel coords): {}.".format(proj_p_pixels))

print("Done.")
"""