import bpy
from mathutils import Matrix, Vector
import numpy as np

# References:
# https://blender.stackexchange.com/questions/213602/getting-edges-of-the-cameras-view-on-the-scene-xy-plane
# https://blender.stackexchange.com/questions/45146/how-to-find-all-objects-in-the-cameras-view-with-python
# https://github.com/federicoarenasl/Data-Generation-with-Blender/blob/master/Resources/main_script.py
# https://federicoarenasl.github.io/Data-Generation-with-Blender/
# https://www.immersivelimit.com/tutorials/synthetic-datasets-with-blender
# https://github.com/ku6ryo/synthetic-dataset-torus/blob/master/blender/script.py
# https://www.youtube.com/watch?v=J2C7QbUZL6Y
# https://github.com/georg-wolflein/chesscog/blob/master/scripts/synthesize_data.py
# https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
# https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html#details

# References
# https://gist.github.com/autosquid/8e1cddbc0336a49c6f84591d35371c4d
# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
# https://visp-doc.inria.fr/doxygen/visp-3.4.0/tutorial-tracking-mb-generic-rgbd-Blender.html
# https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
# https://learnopencv.com/camera-calibration-using-opencv/
# https://blender.stackexchange.com/questions/16472/how-can-i-get-the-cameras-projection-matrix
# https://gist.github.com/autosquid/8e1cddbc0336a49c6f84591d35371c4d
# https://visp-doc.inria.fr/doxygen/visp-3.4.0/tutorial-tracking-mb-generic-rgbd-Blender.html
# https://github.com/vsitzmann/shapenet_renderer/blob/master/util.py
# https://mcarletti.github.io/articles/blenderintrinsicparams/

# Conversion Pascal to .. etc.
# https://christianbernecker.medium.com/convert-bounding-boxes-from-coco-to-pascal-voc-to-yolo-and-back-660dc6178742

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    # focal length
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

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

def Get_Min_Max(vertices):
    """
    Description:
        Get the minimum and maximum X, Y, Z values of the input vertices.

    Args:
        (1) vertices [Vector<float> 8x3]: Vertices of the bounding box (AABB, OBB).

    Returns:
        (1) parameter [Vector<float> 1x3]: Minimum X, Y, Z values of the input vertices.
        (2) parameter [Vector<float> 1x3]: Maximum X, Y, Z values of the input vertices.
    """

    min_vec3 = np.array([vertices[0, 0], vertices[0, 1], vertices[0, 2]], dtype=np.float32)
    max_vec3 = min_vec3.copy()
    
    for _, verts_i in enumerate(vertices[1::]):
        for j, verts_ij in enumerate(verts_i):
            if verts_ij < min_vec3[j]:
                min_vec3[j] = verts_ij

            if verts_ij > max_vec3[j]:
                max_vec3[j] = verts_ij
                
    return (min_vec3, max_vec3)

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

# ----------------------------------------------------------
if __name__ == "__main__":
    # Insert your camera name here
    cam = bpy.data.objects['Camera']
    P, K, RT = get_3x4_P_matrix_from_blender(cam)
    #print("K")
    #print(K)
    #print("RT")
    #print(RT)
    #print("P")
    #print(P)

    #print("==== 3D Cursor projection ====")
    pc = P @ bpy.context.scene.cursor.location
    pc /= pc[2]
    #print("Projected cursor location")
    #print(pc)
    
    verts = Get_Vertices_From_Object('Cube')
    (min, max) = Get_Min_Max(np.array(verts))

    # X
    bpy.data.objects['Sphere_Min_X'].location = [min[0], (max[1] + min[1])/2.0, max[2]]
    bpy.data.objects['Sphere_Max_X'].location = [max[0], (max[1] + min[1])/2.0, max[2]]
    # Y
    bpy.data.objects['Sphere_Min_Y'].location = [(max[0] + min[0])/2.0, min[1], max[2]]
    bpy.data.objects['Sphere_Max_Y'].location = [(max[0] + min[0])/2.0, max[1], max[2]]

    b = np.zeros(4)
    
    #print("Max Y")
    pc = P @ bpy.data.objects['Sphere_Min_X'].location
    print('Res', pc)
    pc /= pc[2]
    b[3] = pc[1]
    #print(pc[1])
    
    #print("Min Y")
    pc = P @ bpy.data.objects['Sphere_Max_X'].location
    pc /= pc[2]
    b[2] = pc[1]
    #print(pc[1])
    
    #print("Max X")
    pc = P @ bpy.data.objects['Sphere_Min_Y'].location
    pc /= pc[2]
    b[1] = pc[0]
    #print(pc[0])
    
    #print("Min X")
    pc = P @ bpy.data.objects['Sphere_Max_Y'].location
    pc /= pc[2]
    b[0] = pc[0]
    #print(pc[0])
    
    w = 2064
    h = 1544
    
    print(b)
    print('Result 1')
    print(convert((w,h), b))
    print('Result 2')
    print(pascal_voc_to_yolo(b[0], b[2], b[1], b[3], w, h))