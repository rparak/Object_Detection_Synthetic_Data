# BPY (Blender as a python) [pip3 install bpy]
import bpy
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Time (Time access and conversions)
import time
# Custom Library:
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO

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
        Function to enable and disable the visibility of an object.
    
    Args:
        (1) name [string]: Name of the main object.
        (2) state [bool]: Enable (True) / Disable (False).  
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

def Get_Vertices_From_Object(name: str) -> tp.List[tp.List[float]]:
    """
    Description:
        Get (x, y, z) positions of the vertices of the mesh object.

    Args:
        (1) name [string]: Name of the mesh object.

    Returns:
        (1) parameter [Matrix<float> 3xn]: The vector (list) of given vertices.
                                           Note:
                                            Where n is the number of vertices.
    """

    return [bpy.data.objects[name].matrix_world @ vertex_i.co for vertex_i in bpy.data.objects[name].data.vertices]

def Save_Synthetic_Data(file_path: str, partition_name: str, iteration: int, object_id: int, bounding_box: tp.List[tp.Union[int, float]], label_format: str, 
                        image_format: str) -> None:
    """
    Description:
        Function to save data generated from the Blender. More precisely, an image in the required format 
        with corresponding labelling.

    Args:
        (1) file_path [string]: The specified path of the file without extension (format).
        (2) partition_name [string]: The name of the partition of the dataset.
        (2) iteration [int]: The current iteration of the process.
        (3) object_id [int]: The identification number of the scanned object.
        (4) bounding_box [Vector<float> 1x4]: The bounding box data (2D) in the specific format (YOLO, PASCAL_VOC, etc.)
        (5) label_format [string]: The format of the saved file.
                                   Note:
                                    'pkl' : Pickle file; 'txt' : Text file.
        (6) image_format [string]: The format of the saved image.
                                   Note:
                                    'png', jpeg', etc.
    """

    # Start the timer.
    t_0 = time.time()

    # Save the label data (bounding box) to a file.
    label_data = list(np.hstack((object_id, bounding_box))); label_data[0] = int(label_data[0])
    File_IO.Save(f'{file_path}/labels/{partition_name}/Object_ID_{object_id}_{iteration}', label_data, label_format.lower(), ' ')
    
    # Save the image to a file.
    bpy.context.scene.render.filepath = f'{file_path}/images/{partition_name}/Object_ID_{object_id}_{iteration}.{image_format.lower()}'
    bpy.ops.render.render(animation=False, write_still=True)

    # Display information.
    print(f'[INFO] The data in iteration {int(iteration)} was successfully saved to the folder {file_path}.')
    print(f'[INFO]  - Image: /images/{partition_name}/Object_ID_{object_id}_{iteration}.{image_format.lower()}')
    print(f'[INFO]  - Label: /labels/{partition_name}/Object_ID_{object_id}_{iteration}.txt')
    print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')