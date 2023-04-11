# BPY (Blender as a python) [pip3 install bpy]
import bpy
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Library:
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation

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

def Remove_Object(name: str) -> None:
    """
    Description:
        Remove the object (hierarchy) from the scene, if it exists. 
    Args:
        (1) name [string]: The name of the object.
    """

    # Find the object with the desired name in the scene.
    object_name = None
    for obj in bpy.data.objects:
        if name in obj.name and Object_Exist(obj.name) == True:
            object_name = obj.name
            break

    # If the object exists, remove it, as well as the other objects in the hierarchy.
    if object_name is not None:
        bpy.data.objects[object_name].select_set(True)
        for child in bpy.data.objects[object_name].children:
            child.select_set(True)
        bpy.ops.object.delete()
        bpy.context.view_layer.update()

def Set_Object_Material_Transparency(name: str, alpha: float) -> None:
    """
    Description:
        Set the transparency of the object material and/or the object hierarchy (if exists).
        
        Note: 
            alpha = 1.0: Render surface without transparency.
            
    Args:
        (1) name [string]: The name of the object.
        (2) alpha [float]: Transparency information.
                           (total transparency is 0.0 and total opacity is 1.0)
    """

    for obj in bpy.data.objects:
        if bpy.data.objects[name].parent == True:
            if obj.parent == bpy.data.objects[name]:
                for material in obj.material_slots:
                    if alpha == 1.0:
                        material.material.blend_method  = 'OPAQUE'
                    else:
                        material.material.blend_method  = 'BLEND'
                    
                    material.material.shadow_method = 'OPAQUE'
                    material.material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = alpha
                
                # Recursive call.
                return Set_Object_Material_Transparency(obj.name, alpha)
        else:
            if obj == bpy.data.objects[name]:
                for material in obj.material_slots:
                    if alpha == 1.0:
                        material.material.blend_method  = 'OPAQUE'
                    else:
                        material.material.blend_method  = 'BLEND'
                    
                    material.material.shadow_method = 'OPAQUE'
                    material.material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = alpha

def __Add_Primitive(type: str, properties: tp.Tuple[float, tp.List[float], tp.List[float]]) -> bpy.ops.mesh:
    """
    Description:
        Add a primitive three-dimensional object.
    Args:
        (1) type [string]: Type of the object. 
                            Primitives: ['Plane', 'Cube', 'Sphere', 'Capsule']
        (2) properties [Dictionary {'Size/Radius': float, 'Scale/Size/None': Vector<float>, 
                                    'Location': Vector<float>]: Transformation properties of the created object. The structure depends 
                                                                on the specific object.
    
    Returns:
        (1) parameter [bpy.ops.mesh]: Individual three-dimensional object (primitive).
    """
        
    return {
        'Plane': lambda x: bpy.ops.mesh.primitive_plane_add(size=x['Size'], scale=x['Scale'], location=x['Location']),
        'Cube': lambda x: bpy.ops.mesh.primitive_cube_add(size=x['Size'], scale=x['Scale'], location=x['Location']),
        'Sphere': lambda x: bpy.ops.mesh.primitive_uv_sphere_add(radius=x['Radius'], location=x['Location']),
        'Capsule': lambda x: bpy.ops.mesh.primitive_round_cube_add(radius=x['Radius'], size=x['Size'], location=x['Location'], arc_div=10)
    }[type](properties)

def Create_Primitive(type: str, name: str, properties: tp.Tuple[tp.Tuple[float, tp.List[float]], tp.Tuple[float]]) -> None:
    """
    Description:
        Create a primitive three-dimensional object with additional properties.
    Args:
        (1) type [string]: Type of the object. 
                            Primitives: ['Plane', 'Cube', 'Sphere', 'Capsule']
        (2) name [string]: The name of the created object.
        (3) properties [{'transformation': {'Size/Radius': float, 'Scale/Size/None': Vector<float>, Location': Vector<float>}, 
                         'material': {'RGBA': Vector<float>, 'alpha': float}}]: Properties of the created object. The structure depends on 
                                                                                on the specific object.
    """

    # Create a new material and set the material color of the object.
    material = bpy.data.materials.new(f'{name}_mat')
    material.use_nodes = True
    material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = properties['material']['RGBA']

    # Add a primitive three-dimensional object.
    __Add_Primitive(type, properties['transformation'])

    # Change the name and material of the object.
    bpy.context.active_object.name = name
    bpy.context.active_object.active_material = material

    # Set the transparency of the object material.
    if properties['material']['alpha'] < 1.0:
        Set_Object_Material_Transparency(name, properties['material']['alpha'])

    # Deselect all objects in the current scene.
    Deselect_All()

    # Update the scene.
    bpy.context.view_layer.update()

def Set_Object_Origin(name: str, location: tp.List[float]) -> None:
    """
    Description:
        Set the origin of the individual objects.

    Args:
        (1) name [string]: Name of the mesh object. 
        (2) location [Vector<float> 1x3]: The origin of the object (location in x, y, z coordinates).
    """

    # Select an object.
    bpy.data.objects[name].select_set(True)

    # Set the position of the cursor and the origin of the 
    # object.
    bpy.context.scene.cursor.location = location
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    # Deselect an object and update the layer.
    bpy.data.objects[name].select_set(False)
    bpy.context.view_layer.update()

def Object_Visibility(name: str, state: bool) -> None:
    """
    Description:
        Function to hide and unhide the visibility of objects.
    
    Args:
        (1) name [string]: Name of the main object.
        (2) state [bool]: Unhide (True) / Hide (False).  
    """
    
    cmd = not state; obj = bpy.data.objects[name]
    
    if Object_Exist(name):
        obj.hide_viewport = cmd; obj.hide_render = cmd
        for obj_i in obj.children:
            obj_i.hide_viewport = cmd; obj_i.hide_render = cmd

def Deselect_All() -> None:
    """
    Description:
        Deselect all objects in the current scene.
    """
    
    for obj in bpy.context.selected_objects:
        bpy.data.objects[obj.name].select_set(False)

def Set_Object_Transformation(name: str, T: tp.List[tp.List[float]]) -> None:
    """
    Description:
        Set the object transformation.
    Args:
        (1) name [string]: Name of the main object.
        (2) T [Matrix<float> 4x4]: Homogeneous transformation matrix (access to location, rotation and scale).
    """

    if isinstance(T, (list, np.ndarray)):
        T = Transformation.Homogeneous_Transformation_Matrix_Cls(T, np.float32)
    
    bpy.data.objects[name].matrix_basis = T.Transpose().all().copy()

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