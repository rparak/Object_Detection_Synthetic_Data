import bpy
from mathutils import Matrix

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K

def Get_Vertices_From_Object(name):
    """
    Description:
        Get (x, y, z) positions of the vertices of the mesh object.

    Args:
        (1) name [string]: Name of the mesh object.

    Returns:
        (1) parameter [Vector<float>]: Vector (list) of given vertices.
    """

    return [bpy.data.objects[name].matrix_world @ vertex_i.co for vertex_i in bpy.data.objects[name].data.vertices]


K = get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)
print(K)

verts = Get_Vertices_From_Object('Cube')

for verts_i in verts:
    print(K @ verts_i)

    
"""
cam_name = "Camera" #or whatever it is
obj_name = "Cube" #or whatever it is

cam = bpy.data.objects[cam_name]
obj = bpy.data.objects[obj_name]

mat_rel = cam.matrix_world.inverted() @ obj.matrix_world

# location
relative_location = mat_rel.translation
# rotation
relative_rotation_euler = mat_rel.to_euler() #must be converted from radians to degrees
relative_rotation_quat = mat_rel.to_quaternion()


mesh = obj.to_mesh(preserve_all_data_layers=True)
mesh.transform(mat_rel)
"""
    
"""
for v in mesh.vertices:
    print(v.co)
"""
