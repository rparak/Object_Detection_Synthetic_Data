# BPY (Blender as a python) [pip3 install bpy]
import bpy
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp

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

# CAMERA: ROT Y 11.75,
# LOC:
# X- 0.133691 m
# Y- -0.170359 m
# Z- 0.881605 m
"""
Description:
    Initialization of constants.
"""
CONST_NULL = 0.0
# Mathematical constants
CONST_MATH_PI = 3.141592653589793
    
# OBJ: X (-0.1 .. 0.45) = 0.55 / 2 = 0.275
# OBJ: Y (-0.5 .. 0.5) = 1.0 / 2 = 0.5

# Center_{Obj_1} = [0.175, 0.0, 0.0193]
# Render Region for random objects:
#   Location:
#       X_{+} = C[0] + 0.4
#       X_{-} = C[0] - 0.3
#       Y_{+} = C[1] + 0.2
#       Y_{-} = C[1] - 0.225
#   Rotation:
#       Z_{+-} = 180 -> 3.14159

# T_Joint_Type_0_ID_0
# Location: [0.175, 0, 0.0193]
# Rotation: [0,2,0]
# Random: Location (X, Y), Rotation (Z)

# T_Joint_Type_0_ID_1
# Location: [0.175, 0, 0.0302]
# Rotation: [90, 0, -90]
# Random: Location (X, Y), Rotation (Y)

# T_Joint_Type_0_ID_1
# Location: [0.175, 0, 0.0302]
# Rotation: [90,0,0]
# Random: Location (X, Y), Rotation (Y)

def Object_Exist(name: str) -> bool:
    """
    Description:
        Check if the object exists within the scene.
        
    Args:
        (1) name [string]: Object name.

    Returns:
        (1) parameter [bool]: 'True' if it exists, otherwise 'False'.
    """
    
    return True if bpy.context.scene.objects.get(name) else False

def Get_Vertices_From_Object(name: str) -> tp.List[float]:
    """
    Description:
        Get (x, y, z) positions of the vertices of the mesh object.

    Args:
        (1) name [string]: Name of the mesh object.

    Returns:
        (1) parameter [Vector<float>]: Vector (list) of given vertices.
    """

    return [bpy.data.objects[name].matrix_world @ vertex_i.co for vertex_i in bpy.data.objects[name].data.vertices]

def Get_Min_Max(vertices: tp.List[float]) -> tp.Tuple[tp.List[float], tp.List[float]]:
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

obj = bpy.data.objects['Cube']
#print(obj.data.vertices[0].co)

""" Get the inverse transformation matrix. """
matrix = bpy.data.objects['Camera'].matrix_world.normalized().inverted()

""" Create a new mesh data block, using the inverse transform matrix to undo any transformations. """
mesh = obj.to_mesh(preserve_all_data_layers=True)
#print(mesh.vertices[0].co)
mesh.transform(obj.matrix_world)
#print(mesh.vertices[0].co)
mesh.transform(matrix)
#print(mesh.vertices[0].co)
#print(f'Location: {mesh}')


""" Get the world coordinates for the camera frame bounding box, before any transformations. """
frame = [-v for v in bpy.data.objects['Camera'].data.view_frame(scene=bpy.data.scenes['Scene'])[:3]]

lx = []
ly = []

for v in mesh.vertices:
    #print(v.co)
    co_local = v.co
    z = -co_local.z

    if z <= 0.0:
        """ Vertex is behind the camera; ignore it. """
        continue
    else:
        """ Perspective division """
        frame = [(v / (v.z / z)) for v in frame]

    min_x, max_x = frame[1].x, frame[2].x
    min_y, max_y = frame[0].y, frame[1].y

    x = (co_local.x - min_x) / (max_x - min_x)
    y = (co_local.y - min_y) / (max_y - min_y)

    lx.append(x)
    ly.append(y)


""" Image is not in view if all the mesh verts were ignored """
if not lx or not ly:
    print('None')

min_x = np.clip(min(lx), 0.0, 1.0)
min_y = np.clip(min(ly), 0.0, 1.0)
max_x = np.clip(max(lx), 0.0, 1.0)
max_y = np.clip(max(ly), 0.0, 1.0)


""" Image is not in view if both bounding points exist on the same side """
if min_x == max_x or min_y == max_y:
    print('None')

""" Figure out the rendered image size """
render = bpy.data.scenes['Scene'].render
fac = render.resolution_percentage * 0.01
dim_x = render.resolution_x * fac
dim_y = render.resolution_y * fac

## Verify there's no coordinates equal to zero
coord_list = [min_x, min_y, max_x, max_y]
if min(coord_list) == 0.0:
    indexmin = coord_list.index(min(coord_list))
    coord_list[indexmin] = coord_list[indexmin] + 0.0000001
    
coordinates = ((min_x, min_y), (max_x, max_y))

"""
verts = Get_Vertices_From_Object('Cube')
(min, max) = Get_Min_Max(np.array(verts))
print(min,max)
"""

"""
for vertex_i in bpy.data.objects['Cube'].data.vertices:
    print((bpy.data.objects['Cube'].matrix_world @ matrix) @ vertex_i.co)
"""


## Change coordinates reference frame
x1 = (coordinates[0][0])
x2 = (coordinates[1][0])
y1 = (1 - coordinates[1][1])
y2 = (1 - coordinates[0][1])

## Get final bounding box information
width = (x2-x1)  # Calculate the absolute width of the bounding box
height = (y2-y1) # Calculate the absolute height of the bounding box
# Calculate the absolute center of the bounding box
cx = x1 + (width/2) 
cy = y1 + (height/2)

## Formulate line corresponding to the bounding box of one class
txt_coordinates = str(0) + ' ' + str(cx) + ' ' + str(cy) + ' ' + str(width) + ' ' + str(height) + '\n'
print('Result')
print(txt_coordinates)

         
"""
rnd_p = False

print(np.random.uniform(-1,0,1))

print(bpy.data.objects['T_Joint_Type_0_ID_0'].rotation_euler.x)

if rnd_p == True:
    bpy.data.objects['T_Joint_Type_0_ID_0'].location = [0.0, 0.0, 0.0]
    # radians ...
    #bpy.data.objects['T_Joint_Type_0_ID_0'].rotation_euler = [0.0, 0.0, 0.0]
else:
    bpy.data.objects['T_Joint_Type_0_ID_0'].location = [0.175, 0.0, 0.0193]
    # radians ..
    #bpy.data.objects['T_Joint_Type_0_ID_0'].rotation_euler = [0.0, 2.0, 0.0]
    

verts = Get_Vertices_From_Object('T_Joint_Type_0_ID_0')

(min, max) = Get_Min_Max(np.array(verts))

print(min, max)
"""

# X
#bpy.data.objects['Sphere_Min_X'].location = [min[0], (max[1] + min[1])/2.0, max[2]]
#bpy.data.objects['Sphere_Max_X'].location = [max[0], (max[1] + min[1])/2.0, max[2]]
# Y
#bpy.data.objects['Sphere_Min_Y'].location = [(max[0] + min[0])/2.0, min[1], max[2]]
#bpy.data.objects['Sphere_Max_Y'].location = [(max[0] + min[0])/2.0, max[1], max[2]]